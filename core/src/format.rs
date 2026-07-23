//! The closed constraint algebra + whole-graph solver (spec: Formats and
//! negotiation). Open vocabulary (interned ids), closed value/constraint set —
//! intersection is a page of integer code, no backtracking search.

use crate::id::{FieldId, FormatId, ValueId};

/// The entire value universe. No strings, no heap — memcmp-comparable.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Value {
    Int(i64),
    /// Exact rational (num, den) — rates, aspect ratios.
    Rat(i32, i32),
    /// Interned categorical value (pixel format, codec profile).
    Id(ValueId),
}

/// Ordering is *partial*: defined within a numeric kind, `None` across kinds and for
/// categorical [`Value::Id`]. This is why `Value` is `Eq` but not `Ord`.
impl PartialOrd for Value {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        match (*self, *other) {
            (Value::Int(a), Value::Int(b)) => Some(a.cmp(&b)),
            (Value::Rat(an, ad), Value::Rat(bn, bd)) => {
                if ad == 0 || bd == 0 {
                    return None;
                }
                // Normalise to positive denominators (via i64 to dodge i32::MIN
                // negation overflow), then cross-multiply in i128.
                let (an, ad) = if ad < 0 {
                    (-(an as i64), -(ad as i64))
                } else {
                    (an as i64, ad as i64)
                };
                let (bn, bd) = if bd < 0 {
                    (-(bn as i64), -(bd as i64))
                } else {
                    (bn as i64, bd as i64)
                };
                let lhs = an as i128 * bd as i128;
                let rhs = bn as i128 * ad as i128;
                Some(lhs.cmp(&rhs))
            }
            _ => None,
        }
    }
}

/// The entire constraint language a pad may declare per field.
#[derive(Clone, Debug)]
pub enum Constraint {
    Any,
    Eq(Value),
    Range { min: Value, max: Value, step: Value },
    Set(&'static [Value]),
}

impl Constraint {
    /// Does this constraint admit `v`? The building block of the solver.
    pub fn accepts(&self, v: Value) -> bool {
        use core::cmp::Ordering;
        match self {
            Constraint::Any => true,
            Constraint::Eq(x) => *x == v,
            Constraint::Set(vs) => vs.iter().any(|x| *x == v),
            Constraint::Range { min, max, step } => {
                let ge_min = matches!(
                    v.partial_cmp(min),
                    Some(Ordering::Greater | Ordering::Equal)
                );
                let le_max =
                    matches!(v.partial_cmp(max), Some(Ordering::Less | Ordering::Equal));
                if !(ge_min && le_max) {
                    return false;
                }
                // Step alignment for the common Int case (width/height/rate). For
                // Rat/Id ranges the step is ignored for now (TODO: rational steps).
                match (v, *min, *step) {
                    (Value::Int(vi), Value::Int(mi), Value::Int(si)) => {
                        si <= 0 || (vi - mi).rem_euclid(si) == 0
                    }
                    _ => true,
                }
            }
        }
    }
}

pub struct FieldConstraint {
    pub field: FieldId,
    pub allowed: Constraint,
    /// Fixation hint (native resolution, display rate).
    pub preferred: Option<Value>,
}

pub struct FormatOffer {
    pub family: FormatId,
    pub fields: &'static [FieldConstraint],
}

/// The solve's per-edge output: flat POD, memcmp-comparable, cheap to copy into
/// batch/edge metadata. Up to 16 fixed fields inline (spill-to-blob is TODO).
#[derive(Clone)]
pub struct FixedFormat {
    pub family: FormatId,
    len: u8,
    fields: [(FieldId, Value); 16],
    // TODO(step 2): optional refcounted config blob (SPS/PPS, STREAMINFO) —
    // equality by (len, hash) so formats stay cheaply comparable.
}

impl FixedFormat {
    pub fn new(family: FormatId) -> Self {
        Self {
            family,
            len: 0,
            fields: [(FieldId(0), Value::Int(0)); 16],
        }
    }

    pub fn get(&self, field: FieldId) -> Option<Value> {
        self.fields[..self.len as usize]
            .iter()
            .find(|(f, _)| *f == field)
            .map(|(_, v)| *v)
    }

    /// Insert or overwrite `field`. Returns `false` only if full (16 fields).
    pub fn set(&mut self, field: FieldId, value: Value) -> bool {
        for slot in &mut self.fields[..self.len as usize] {
            if slot.0 == field {
                slot.1 = value;
                return true;
            }
        }
        if (self.len as usize) < self.fields.len() {
            self.fields[self.len as usize] = (field, value);
            self.len += 1;
            true
        } else {
            false
        }
    }

    pub fn len(&self) -> usize {
        self.len as usize
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

const ANY: Constraint = Constraint::Any;

fn find_field<'a>(offer: &'a FormatOffer, field: FieldId) -> Option<&'a FieldConstraint> {
    offer.fields.iter().find(|f| f.field == field)
}

/// Collect candidate values a constraint might fixate to, in a natural order.
fn push_candidates(c: &Constraint, out: &mut Vec<Value>) {
    match c {
        Constraint::Any => {}
        Constraint::Eq(v) => out.push(*v),
        Constraint::Set(vs) => out.extend_from_slice(vs),
        Constraint::Range { min, max, .. } => {
            out.push(*min);
            out.push(*max);
        }
    }
}

/// Pick a single value both sides accept for one field.
///
/// Returns `Some(Some(v))` when fixated to `v`, `Some(None)` when both sides leave
/// the field unconstrained (so it isn't fixed), and `None` when the field is
/// constrained but has no common value (an empty intersection).
///
/// Fixation is candidate-based: it tries each side's `preferred` hint, then the
/// concrete values each constraint implies (Eq/Set values, Range endpoints), and
/// takes the first one both constraints accept. Preferences win by being tried
/// first. Limitation: two overlapping `Range`s whose overlap contains neither
/// endpoint and whose steps are misaligned can be missed — real formats use Eq/Set
/// or matching ranges, and this can grow an LCM walk later.
fn fixate(a: Option<&FieldConstraint>, b: Option<&FieldConstraint>) -> Option<Option<Value>> {
    let ca = a.map_or(&ANY, |f| &f.allowed);
    let cb = b.map_or(&ANY, |f| &f.allowed);

    if matches!(ca, Constraint::Any) && matches!(cb, Constraint::Any) {
        return Some(None); // unconstrained on both sides
    }

    let mut candidates: Vec<Value> = Vec::new();
    if let Some(v) = a.and_then(|f| f.preferred) {
        candidates.push(v);
    }
    if let Some(v) = b.and_then(|f| f.preferred) {
        candidates.push(v);
    }
    push_candidates(ca, &mut candidates);
    push_candidates(cb, &mut candidates);

    for cand in candidates {
        if ca.accepts(cand) && cb.accepts(cand) {
            return Some(Some(cand));
        }
    }
    None // constrained, but no value satisfies both
}

/// Intersect two offers on a link and fixate, or `None` when the intersection is
/// empty (a loud `Ready`-time failure, never a runtime flow error). Fields are
/// fixated with [`fixate`]; fields unconstrained on both sides are left unset.
pub fn intersect(a: &FormatOffer, b: &FormatOffer) -> Option<FixedFormat> {
    if a.family != b.family {
        return None;
    }
    let mut out = FixedFormat::new(a.family);

    for fc_a in a.fields {
        let value = fixate(Some(fc_a), find_field(b, fc_a.field))?;
        if let Some(v) = value {
            if !out.set(fc_a.field, v) {
                return None; // too many fixed fields
            }
        }
    }
    // Fields present only on b.
    for fc_b in b.fields {
        if find_field(a, fc_b.field).is_some() {
            continue;
        }
        if let Some(v) = fixate(None, Some(fc_b))? {
            if !out.set(fc_b.field, v) {
                return None;
            }
        }
    }

    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::id::{FieldId, FormatId, ValueId};

    const fn vi(n: i64) -> Value {
        Value::Int(n)
    }

    #[test]
    fn int_ordering() {
        use core::cmp::Ordering;
        assert_eq!(vi(1).partial_cmp(&vi(2)), Some(Ordering::Less));
        assert_eq!(vi(2).partial_cmp(&vi(2)), Some(Ordering::Equal));
        // Across kinds: incomparable.
        assert_eq!(vi(1).partial_cmp(&Value::Id(ValueId(1))), None);
    }

    #[test]
    fn rational_ordering() {
        use core::cmp::Ordering;
        assert_eq!(Value::Rat(1, 2).partial_cmp(&Value::Rat(2, 3)), Some(Ordering::Less));
        assert_eq!(Value::Rat(2, 4).partial_cmp(&Value::Rat(1, 2)), Some(Ordering::Equal));
        // Negative denominators are normalised: 1/-2 == -1/2.
        assert_eq!(Value::Rat(1, -2).partial_cmp(&Value::Rat(-1, 2)), Some(Ordering::Equal));
        // Zero denominator is incomparable, not a panic.
        assert_eq!(Value::Rat(1, 0).partial_cmp(&Value::Rat(1, 2)), None);
    }

    #[test]
    fn accepts_eq_and_set() {
        assert!(Constraint::Any.accepts(vi(5)));
        assert!(Constraint::Eq(vi(5)).accepts(vi(5)));
        assert!(!Constraint::Eq(vi(5)).accepts(vi(6)));

        let set = Constraint::Set(&[Value::Int(8), Value::Int(16), Value::Int(24)]);
        assert!(set.accepts(vi(16)));
        assert!(!set.accepts(vi(12)));
    }

    #[test]
    fn accepts_range_with_step() {
        let r = Constraint::Range { min: vi(16), max: vi(1920), step: vi(16) };
        assert!(r.accepts(vi(16)));
        assert!(r.accepts(vi(1920)));
        assert!(r.accepts(vi(1280)));
        assert!(!r.accepts(vi(1281)), "not step-aligned");
        assert!(!r.accepts(vi(8)), "below min");
        assert!(!r.accepts(vi(2048)), "above max");
    }

    #[test]
    fn fixed_format_get_set_overwrite() {
        let mut f = FixedFormat::new(FormatId(1));
        let (w, h) = (FieldId(10), FieldId(11));
        assert!(f.get(w).is_none());
        assert!(f.set(w, vi(1920)));
        assert!(f.set(h, vi(1080)));
        assert_eq!(f.get(w), Some(vi(1920)));
        assert_eq!(f.get(h), Some(vi(1080)));
        assert!(f.set(w, vi(1280)), "overwrite in place");
        assert_eq!(f.get(w), Some(vi(1280)));
        assert_eq!(f.len(), 2);
    }

    #[test]
    fn fixed_format_capacity() {
        let mut f = FixedFormat::new(FormatId(0));
        for i in 0..16 {
            assert!(f.set(FieldId(i), vi(i as i64)));
        }
        assert_eq!(f.len(), 16);
        // 17th distinct field does not fit.
        assert!(!f.set(FieldId(100), vi(0)));
        // But overwriting an existing field still works when full.
        assert!(f.set(FieldId(0), vi(42)));
        assert_eq!(f.get(FieldId(0)), Some(vi(42)));
    }

    // --- solver (intersect) ---

    static PIXFMTS: [Value; 3] = [
        Value::Id(ValueId(1)),
        Value::Id(ValueId(2)),
        Value::Id(ValueId(3)),
    ];

    fn fc(field: u32, allowed: Constraint, preferred: Option<Value>) -> FieldConstraint {
        FieldConstraint { field: FieldId(field), allowed, preferred }
    }

    fn offer(family: u32, fields: Vec<FieldConstraint>) -> FormatOffer {
        // Leak is fine in tests: FormatOffer.fields is 'static (element descriptors).
        FormatOffer { family: FormatId(family), fields: Box::leak(fields.into_boxed_slice()) }
    }

    #[test]
    fn intersect_family_mismatch() {
        assert!(intersect(&offer(1, vec![]), &offer(2, vec![])).is_none());
    }

    #[test]
    fn intersect_eq_eq() {
        let a = offer(1, vec![fc(10, Constraint::Eq(vi(48000)), None)]);
        let same = offer(1, vec![fc(10, Constraint::Eq(vi(48000)), None)]);
        assert_eq!(intersect(&a, &same).unwrap().get(FieldId(10)), Some(vi(48000)));
        let diff = offer(1, vec![fc(10, Constraint::Eq(vi(44100)), None)]);
        assert!(intersect(&a, &diff).is_none());
    }

    #[test]
    fn intersect_range_and_eq() {
        let range = offer(
            1,
            vec![fc(10, Constraint::Range { min: vi(16), max: vi(4096), step: vi(16) }, None)],
        );
        let good = offer(1, vec![fc(10, Constraint::Eq(vi(1920)), None)]);
        assert_eq!(intersect(&range, &good).unwrap().get(FieldId(10)), Some(vi(1920)));
        let misaligned = offer(1, vec![fc(10, Constraint::Eq(vi(1921)), None)]);
        assert!(intersect(&range, &misaligned).is_none());
    }

    #[test]
    fn intersect_set_set() {
        static A: [Value; 3] = [Value::Int(8), Value::Int(16), Value::Int(24)];
        static B: [Value; 2] = [Value::Int(16), Value::Int(32)];
        let a = offer(1, vec![fc(10, Constraint::Set(&A), None)]);
        let b = offer(1, vec![fc(10, Constraint::Set(&B), None)]);
        assert_eq!(intersect(&a, &b).unwrap().get(FieldId(10)), Some(vi(16)));
    }

    #[test]
    fn intersect_prefers_hint() {
        let wide = |pref| {
            offer(1, vec![fc(10, Constraint::Range { min: vi(0), max: vi(100), step: vi(1) }, pref)])
        };
        assert_eq!(
            intersect(&wide(Some(vi(60))), &wide(Some(vi(30)))).unwrap().get(FieldId(10)),
            Some(vi(60))
        );
    }

    #[test]
    fn intersect_field_only_on_one_side() {
        let a = offer(1, vec![fc(10, Constraint::Eq(vi(2)), None)]);
        let b = offer(1, vec![fc(11, Constraint::Eq(vi(3)), None)]);
        let f = intersect(&a, &b).unwrap();
        assert_eq!(f.get(FieldId(10)), Some(vi(2)));
        assert_eq!(f.get(FieldId(11)), Some(vi(3)));
    }

    #[test]
    fn intersect_any_is_unconstrained() {
        let any = || offer(1, vec![fc(10, Constraint::Any, None)]);
        assert_eq!(intersect(&any(), &any()).unwrap().get(FieldId(10)), None);
        let eq = offer(1, vec![fc(10, Constraint::Eq(vi(7)), None)]);
        assert_eq!(intersect(&any(), &eq).unwrap().get(FieldId(10)), Some(vi(7)));
    }

    #[test]
    fn intersect_multi_field_video() {
        let (w, h, fmt) = (1u32, 2u32, 3u32);
        let src = offer(
            100,
            vec![
                fc(w, Constraint::Range { min: vi(0), max: vi(7680), step: vi(8) }, Some(vi(1920))),
                fc(h, Constraint::Range { min: vi(0), max: vi(7680), step: vi(8) }, Some(vi(1080))),
                fc(fmt, Constraint::Set(&PIXFMTS), None),
            ],
        );
        let sink = offer(
            100,
            vec![
                fc(w, Constraint::Eq(vi(1920)), None),
                fc(h, Constraint::Eq(vi(1080)), None),
                fc(fmt, Constraint::Eq(Value::Id(ValueId(2))), None),
            ],
        );
        let f = intersect(&src, &sink).unwrap();
        assert_eq!(f.get(FieldId(w)), Some(vi(1920)));
        assert_eq!(f.get(FieldId(h)), Some(vi(1080)));
        assert_eq!(f.get(FieldId(fmt)), Some(Value::Id(ValueId(2))));
    }
}
