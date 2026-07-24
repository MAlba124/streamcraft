//! Dynamic element properties (spec: Dynamic element properties — config, not caps,
//! not signals). A property value is a [`Value`] from the format algebra and a
//! [`PropDesc.allowed`](crate::element::PropDesc) is a [`Constraint`](crate::format::Constraint)
//! from that same algebra, so `pipeline.set(el, "bitrate", Value::Int(320_000))`
//! validates with the same closed, no-backtracking check that fixes a format on a
//! link — reused, not reinvented.
//!
//! The mechanism is a per-element mailbox: the set is validated on the *app's*
//! thread and parked in this table; the scheduler observes it at the next **batch
//! boundary** (the safe point it already halts at between `process()` calls) and
//! delivers [`Event::PropChanged`](crate::event::Event::PropChanged). So there is no
//! hot-path lock and no mid-buffer change: a set issued while batch N is in flight
//! takes effect no earlier than N+1, deterministically.
//!
//! Storage per property is a **seqlock** over the `Value` (the spec's experimental
//! lean: one uniform slot for any property; drop to a typed atomic only where a
//! bench shows the per-batch read matters). Writers — rare, app-thread — serialize
//! on a `Mutex` the readers never touch; the reader side is wait-free in practice
//! (a retry only overlaps an in-flight write, whose critical section is three
//! stores). All seqlock atomics are `SeqCst`: reads happen at most once per batch,
//! so the fence cost is noise; relax behind a bench if it ever shows up.

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::element::ElementDesc;
use crate::error::Error;
use crate::format::Value;
use crate::id::{ElementId, ValueId};

// `Value` packed into POD words for the seqlock: a tag word and a payload word.
const TAG_UNSET: u32 = 0;
const TAG_INT: u32 = 1;
const TAG_RAT: u32 = 2;
const TAG_ID: u32 = 3;

fn encode(v: Value) -> (u32, u64) {
    match v {
        Value::Int(i) => (TAG_INT, i as u64),
        Value::Rat(n, d) => (TAG_RAT, ((n as u32 as u64) << 32) | d as u32 as u64),
        Value::Id(ValueId(id)) => (TAG_ID, id as u64),
    }
}

fn decode(tag: u32, bits: u64) -> Option<Value> {
    match tag {
        TAG_INT => Some(Value::Int(bits as i64)),
        TAG_RAT => Some(Value::Rat((bits >> 32) as u32 as i32, bits as u32 as i32)),
        TAG_ID => Some(Value::Id(ValueId(bits as u32))),
        _ => None,
    }
}

/// One property's slot: a seqlock over `(tag, bits)`. Readers never block; writers
/// (already serialized by [`PropTable`]'s writer lock) bump `seq` to odd, store the
/// words, and bump back to even.
struct PropCell {
    seq: AtomicU32,
    tag: AtomicU32,
    bits: AtomicU64,
}

impl PropCell {
    fn new() -> Self {
        Self {
            seq: AtomicU32::new(0),
            tag: AtomicU32::new(TAG_UNSET),
            bits: AtomicU64::new(0),
        }
    }

    /// Store a value. Caller holds the table's writer lock (one writer at a time).
    fn store(&self, v: Value) {
        let (tag, bits) = encode(v);
        let s = self.seq.load(Ordering::SeqCst);
        self.seq.store(s.wrapping_add(1), Ordering::SeqCst); // odd: write in progress
        self.tag.store(tag, Ordering::SeqCst);
        self.bits.store(bits, Ordering::SeqCst);
        self.seq.store(s.wrapping_add(2), Ordering::SeqCst); // even: stable
    }

    /// Lock-free read; `None` while unset. Retries only across an in-flight write.
    fn load(&self) -> Option<Value> {
        loop {
            let s1 = self.seq.load(Ordering::SeqCst);
            if s1 & 1 == 1 {
                std::hint::spin_loop();
                continue;
            }
            let tag = self.tag.load(Ordering::SeqCst);
            let bits = self.bits.load(Ordering::SeqCst);
            if self.seq.load(Ordering::SeqCst) == s1 {
                return decode(tag, bits);
            }
        }
    }
}

/// A per-element property mailbox: one [`PropCell`] per declared property (indexed
/// by position in `desc().props`), a dirty bitmask the scheduler polls once per
/// pass (one relaxed load when idle), and the writer lock that serializes the rare
/// app-side sets. Shared `Arc` between the pipeline/handles (writers) and the
/// element's `Ctx` (reader).
pub struct PropTable {
    cells: Box<[PropCell]>,
    /// Bit per property, set on write, swapped out by the scheduler at the batch
    /// boundary to know *which* props to deliver [`Event::PropChanged`] for.
    ///
    /// [`Event::PropChanged`]: crate::event::Event::PropChanged
    dirty: AtomicU64,
    writer: Mutex<()>,
}

impl PropTable {
    pub(crate) fn new(nprops: usize) -> Self {
        // The dirty mask is one u64; an element declaring >64 properties is a design
        // smell we fail loudly on at `add()` time, not a case to engineer for.
        assert!(nprops <= 64, "an element may declare at most 64 properties");
        Self {
            cells: (0..nprops).map(|_| PropCell::new()).collect(),
            dirty: AtomicU64::new(0),
            writer: Mutex::new(()),
        }
    }

    /// Store `v` into property `idx` and mark it dirty. Caller has validated.
    pub(crate) fn set(&self, idx: usize, v: Value) {
        let _g = self.writer.lock().unwrap_or_else(|e| e.into_inner());
        self.cells[idx].store(v);
        self.dirty.fetch_or(1 << idx, Ordering::Release);
    }

    /// The current value of property `idx`; `None` while never set.
    pub(crate) fn get(&self, idx: usize) -> Option<Value> {
        self.cells.get(idx).and_then(PropCell::load)
    }

    /// Take the dirty bitmask (scheduler, once per pass). The idle path is a single
    /// relaxed load — no RMW unless something actually changed.
    pub(crate) fn take_dirty(&self) -> u64 {
        if self.dirty.load(Ordering::Relaxed) == 0 {
            0
        } else {
            self.dirty.swap(0, Ordering::Acquire)
        }
    }
}

/// Validate a set against the element's declared properties and park it in the
/// table (spec: Dynamic element properties — validated at the call site, loudly).
/// `live_only` is the mid-`run()` restriction: a structural (`live: false`)
/// property re-opens resources, which needs the element-restart micro-transition
/// that doesn't exist yet — setting one while playing is a loud error, not a
/// silent glitch. Between runs anything goes (the element re-reads in `start()`).
pub(crate) fn validate_and_set(
    el: ElementId,
    desc: &'static ElementDesc,
    table: &PropTable,
    prop: &str,
    v: Value,
    live_only: bool,
) -> Result<(), Error> {
    let idx = desc
        .props
        .iter()
        .position(|p| p.name == prop)
        .ok_or_else(|| Error::Resource(format!("set: element '{}' has no property '{prop}'", desc.name)))?;
    let pd = &desc.props[idx];
    if live_only && !pd.live {
        return Err(Error::Element {
            element: el,
            message: format!(
                "set: property '{prop}' on '{}' is structural (live: false) — \
                 setting it while playing needs the element-restart machinery (not yet built); \
                 set it between runs",
                desc.name
            ),
        });
    }
    if !pd.allowed.accepts(v) {
        return Err(Error::Resource(format!(
            "set: value {v:?} rejected by '{}' property '{prop}' constraint {:?}",
            desc.name, pd.allowed
        )));
    }
    table.set(idx, v);
    Ok(())
}

/// A cloneable handle to set **live** properties on a running pipeline from another
/// thread (spec: Dynamic element properties), the same shape as
/// [`StopHandle`](crate::pipeline::StopHandle) / [`SeekHandle`](crate::pipeline::SeekHandle).
/// Sets are validated here, on the caller's thread, against the element's declared
/// constraints; the element observes the change at its next batch boundary via
/// [`Event::PropChanged`](crate::event::Event::PropChanged) and
/// [`Ctx::prop`](crate::ctx::Ctx::prop). Structural (`live: false`) properties are
/// rejected — set those between runs via [`Pipeline::set`](crate::pipeline::Pipeline::set).
///
/// Obtain from [`Pipeline::prop_handle`](crate::pipeline::Pipeline::prop_handle)
/// *after* the topology is built (it snapshots the elements present at that point).
#[derive(Clone)]
pub struct PropHandle {
    entries: Arc<[(&'static ElementDesc, Arc<PropTable>)]>,
}

impl PropHandle {
    pub(crate) fn new(entries: Vec<(&'static ElementDesc, Arc<PropTable>)>) -> Self {
        Self { entries: entries.into() }
    }

    /// Validate and park a live-property set. Cheap; never blocks a streaming thread.
    pub fn set(&self, el: ElementId, prop: &str, v: Value) -> Result<(), Error> {
        let (desc, table) = self
            .entries
            .get(el.0 as usize)
            .ok_or(Error::Todo("set: unknown element"))?;
        validate_and_set(el, desc, table, prop, v, true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::Value;

    #[test]
    fn encode_decode_roundtrip() {
        for v in [
            Value::Int(0),
            Value::Int(-1),
            Value::Int(i64::MAX),
            Value::Int(i64::MIN),
            Value::Rat(30000, 1001),
            Value::Rat(-3, 4),
            Value::Rat(i32::MIN, i32::MAX),
            Value::Id(ValueId(0)),
            Value::Id(ValueId(u32::MAX)),
        ] {
            let (tag, bits) = encode(v);
            assert_eq!(decode(tag, bits), Some(v), "{v:?} must survive the cell");
        }
        assert_eq!(decode(TAG_UNSET, 0), None);
    }

    #[test]
    fn unset_reads_none_then_set_reads_back() {
        let t = PropTable::new(2);
        assert_eq!(t.get(0), None);
        assert_eq!(t.get(1), None);
        t.set(1, Value::Int(42));
        assert_eq!(t.get(0), None, "sibling slot untouched");
        assert_eq!(t.get(1), Some(Value::Int(42)));
        t.set(1, Value::Rat(1, 2));
        assert_eq!(t.get(1), Some(Value::Rat(1, 2)), "overwrite wins");
    }

    #[test]
    fn dirty_mask_tracks_which_and_clears_on_take() {
        let t = PropTable::new(3);
        assert_eq!(t.take_dirty(), 0, "clean at rest");
        t.set(0, Value::Int(1));
        t.set(2, Value::Int(2));
        assert_eq!(t.take_dirty(), 0b101, "bits name the changed props");
        assert_eq!(t.take_dirty(), 0, "take clears");
        t.set(1, Value::Int(3));
        assert_eq!(t.take_dirty(), 0b010);
    }

    #[test]
    fn out_of_range_index_reads_none() {
        let t = PropTable::new(1);
        assert_eq!(t.get(5), None);
    }

    #[test]
    fn concurrent_reader_always_sees_a_written_value() {
        // Seqlock sanity: a reader racing a writer must only ever decode one of the
        // values actually written — never a torn tag/bits combination. The two
        // values are chosen so any mix decodes to something neither thread wrote.
        let t = Arc::new(PropTable::new(1));
        let a = Value::Int(-1); // bits = all ones
        let b = Value::Rat(7, 9);
        t.set(0, a);
        let w = {
            let t = Arc::clone(&t);
            std::thread::spawn(move || {
                for i in 0..20_000u32 {
                    t.set(0, if i & 1 == 0 { b } else { a });
                }
            })
        };
        for _ in 0..20_000 {
            let v = t.get(0).expect("set at least once");
            assert!(v == a || v == b, "torn read: {v:?}");
        }
        w.join().unwrap();
    }
}
