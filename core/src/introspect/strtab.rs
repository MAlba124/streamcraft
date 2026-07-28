//! Per-connection string table (spec: Strings). Every name on the wire is a `u32`
//! into this table; the server sends a [`StrDef`](super::wire::kind::STR_DEF) frame the
//! first time it references a new id, and the client caches it. Ids are dense from 1;
//! 0 means "none".
//!
//! Interning is by **pointer identity** for `&'static str` — the common case, since
//! element/pad/event names are all process-'static, so two references to the same
//! `&'static str` share an id without hashing the bytes. Owned strings (a family name
//! resolved from an interner into a `String` at snapshot time, or a runtime prop
//! string) fall back to a **content** key so equal contents dedup. The two key spaces
//! never collide: a static key is `(ptr, len)`, a content key is the owned bytes.

// Cold diagnostic path: this per-connection interner lives on the client thread and
// allocates only while serializing a snapshot to a poll/attach, not per media buffer.
#![allow(clippy::disallowed_methods)]

use std::collections::HashMap;

/// A key that distinguishes a `&'static str` (by address) from an owned string (by
/// content), so the same text interned via both paths still dedups on content but a
/// static reference avoids hashing its bytes on the hot path.
#[derive(Clone, PartialEq, Eq, Hash)]
enum Key {
    /// `&'static str` identity: its data pointer + length. Two `&'static str` with the
    /// same address and length are the same string (the compiler dedups equal string
    /// literals, and interned names are stored once).
    Static(usize, usize),
    /// An owned string, keyed by content so equal contents share an id.
    Owned(String),
}

/// The per-connection string interning table. Not `Sync`; lives on one client thread.
#[derive(Default)]
pub struct StrTab {
    map: HashMap<Key, u32>,
    /// The string each id resolves to (for tests / debugging); index `id - 1`.
    names: Vec<String>,
}

impl StrTab {
    pub fn new() -> Self {
        Self { map: HashMap::new(), names: Vec::new() }
    }

    /// Number of distinct strings interned so far.
    pub fn len(&self) -> usize {
        self.names.len()
    }

    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }

    /// Resolve an id (`1..=len`) back to its string, for tests. `None` for 0/out-of-range.
    pub fn resolve(&self, id: u32) -> Option<&str> {
        if id == 0 {
            return None;
        }
        self.names.get((id - 1) as usize).map(String::as_str)
    }

    /// Intern a `&'static str` by pointer identity. Returns `(id, is_new)`; when
    /// `is_new`, the caller must emit a `StrDef(id, s)` before first referencing `id`.
    /// An empty string is the "none" sentinel: id 0, never a StrDef.
    pub fn intern_static(&mut self, s: &'static str) -> (u32, bool) {
        if s.is_empty() {
            return (0, false);
        }
        let key = Key::Static(s.as_ptr() as usize, s.len());
        if let Some(&id) = self.map.get(&key) {
            return (id, false);
        }
        let id = self.assign(s.to_owned());
        self.map.insert(key, id);
        (id, true)
    }

    /// Intern an owned string by content. Returns `(id, is_new)`; when `is_new`, emit a
    /// `StrDef` before first reference. An empty string is id 0.
    pub fn intern_owned(&mut self, s: &str) -> (u32, bool) {
        if s.is_empty() {
            return (0, false);
        }
        let key = Key::Owned(s.to_owned());
        if let Some(&id) = self.map.get(&key) {
            return (id, false);
        }
        let id = self.assign(s.to_owned());
        self.map.insert(key, id);
        (id, true)
    }

    fn assign(&mut self, s: String) -> u32 {
        self.names.push(s);
        self.names.len() as u32 // dense from 1
    }
}
