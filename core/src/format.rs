//! The closed constraint algebra + whole-graph solver (spec: Formats and
//! negotiation). Open vocabulary (interned ids), closed value/constraint set —
//! intersection is a page of integer code, no backtracking search.

use crate::id::{FieldId, FormatId, Interner, ValueId};

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
                        // `checked_sub`: the bounds come from an element's `OfferDesc`, so a range
                        // wider than `i64::MAX` — the plausible "any integer" idiom
                        // `Range { min: i64::MIN, max: i64::MAX, step: n }` — overflowed here.
                        // Debug panicked; release wrapped and silently accepted or rejected the
                        // wrong values. A range that wide cannot meaningfully constrain a step, so
                        // treat it as unstepped.
                        si <= 0 || vi.checked_sub(mi).is_none_or(|d| d.rem_euclid(si) == 0)
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
    /// A **wildcard** offer — a family-level marker meaning "adopt the peer's family"
    /// (spec: Formats — family-agnostic elements). It is not a new constraint kind: the
    /// closed value/constraint algebra is untouched; this flag only widens *family*
    /// matching in [`intersect`]. A wildcard offer carries no fields (a wildcard pad
    /// cannot constrain fields — it does not know the peer's field vocabulary); an
    /// `intersect` involving a wildcard-with-fields is rejected loudly. When both sides
    /// are wildcards the result adopts the interned "any" family ([`WILDCARD`]).
    pub wildcard: bool,
}

// --- Static, string-keyed descriptor layer -------------------------------------
//
// The id-based [`FormatOffer`] above is what the solver consumes, but interned ids
// are a *pipeline-scoped* fact (one interner per graph). Element descriptors are
// `'static` and shared, so they cannot bake ids in. Instead a pad declares its
// offers with `&'static str` families/fields/values — a `const`-constructible
// mirror of the offer types — and the pipeline lowers them to id-based offers at
// **link time** (spec: Formats — "interning happens once, in the pipeline").
//
// The mirror is 1:1 with `Value`/`Constraint`/`FieldConstraint`/`FormatOffer`, the
// only difference being that every interned handle (`FormatId`, `FieldId`,
// `ValueId`) is a string here. Lowering is a pure interning walk — no solving.

/// String form of [`Value`]: categorical ids are names, resolved at link time.
#[derive(Clone, Copy, Debug)]
pub enum ValueDesc {
    Int(i64),
    Rat(i32, i32),
    /// A categorical value (pixel format, codec profile) named for interning.
    Id(&'static str),
}

/// String form of [`Constraint`] — the same closed algebra over [`ValueDesc`].
#[derive(Clone, Copy, Debug)]
pub enum ConstraintDesc {
    Any,
    Eq(ValueDesc),
    Range { min: ValueDesc, max: ValueDesc, step: ValueDesc },
    Set(&'static [ValueDesc]),
}

/// String form of [`FieldConstraint`]: the field is named, not yet interned.
#[derive(Clone, Copy, Debug)]
pub struct FieldDesc {
    pub field: &'static str,
    pub allowed: ConstraintDesc,
    pub preferred: Option<ValueDesc>,
}

/// String form of [`FormatOffer`] — what a [`PadDesc`](crate::element::PadDesc)
/// declares. `const`-constructible: families/fields/values are `&'static str`, and
/// the field table is a `&'static [FieldDesc]`.
#[derive(Clone, Copy, Debug)]
pub struct OfferDesc {
    pub family: &'static str,
    pub fields: &'static [FieldDesc],
}

/// The reserved wildcard family name (spec: Formats — the wildcard is a family-level
/// marker). A pad whose offer names this family "adopts the peer's family" at
/// negotiation; it must carry no fields. Distinct from [`OfferDesc::any`], which is a
/// *concrete* family with no field constraints — `any("bytes")` still only intersects
/// `bytes`, whereas [`OfferDesc::wildcard`] intersects *any* family. The name is chosen
/// so an accidental collision with a real family is implausible, and so the graph dump
/// renders it legibly.
pub const WILDCARD: &str = "*";

impl OfferDesc {
    /// Convenience for a pure passthrough / byte pad: match `family` with no fields
    /// constrained, so it intersects with any other offer in the same family.
    pub const fn any(family: &'static str) -> Self {
        OfferDesc { family, fields: &[] }
    }

    /// A **wildcard** offer: adopts the peer's family at negotiation (spec: Formats —
    /// the blocker for family-agnostic elements like `queue`, `tee`, `funnel`). A
    /// wildcard pad advertises this single offer; intersecting it with any concrete
    /// offer yields that concrete side's family *and* fields, and wildcard-vs-wildcard
    /// picks the interned "any" family. A wildcard offer declares no fields, and it is
    /// an error to give it any — a wildcard pad cannot constrain fields it does not know
    /// (enforced in [`intersect`], which rejects a wildcard-with-fields loudly).
    pub const fn wildcard() -> Self {
        OfferDesc { family: WILDCARD, fields: &[] }
    }

    /// Whether this static offer is a wildcard (its family is [`WILDCARD`]).
    pub fn is_wildcard(&self) -> bool {
        self.family == WILDCARD
    }
}

impl ValueDesc {
    /// Intern any named categorical value into a concrete [`Value`].
    fn lower(&self, values: &mut Interner) -> Value {
        match *self {
            ValueDesc::Int(n) => Value::Int(n),
            ValueDesc::Rat(n, d) => Value::Rat(n, d),
            ValueDesc::Id(s) => Value::Id(ValueId(values.intern(s))),
        }
    }
}

impl ConstraintDesc {
    fn lower(&self, values: &mut Interner) -> Constraint {
        match *self {
            ConstraintDesc::Any => Constraint::Any,
            ConstraintDesc::Eq(v) => Constraint::Eq(v.lower(values)),
            ConstraintDesc::Range { min, max, step } => Constraint::Range {
                min: min.lower(values),
                max: max.lower(values),
                step: step.lower(values),
            },
            ConstraintDesc::Set(vs) => {
                // `Constraint::Set` borrows `&'static [Value]`; the interned values
                // are pipeline-lived, so we leak the lowered slice. Offers are lowered
                // once per link (spec: solve is a link-time cost, never per-buffer), so
                // this is bounded by the graph's edge count, not the stream.
                let lowered: Vec<Value> = vs.iter().map(|v| v.lower(values)).collect();
                Constraint::Set(Box::leak(lowered.into_boxed_slice()))
            }
        }
    }
}

impl FieldDesc {
    fn lower(&self, fields: &mut Interner, values: &mut Interner) -> FieldConstraint {
        FieldConstraint {
            field: FieldId(fields.intern(self.field)),
            allowed: self.allowed.lower(values),
            preferred: self.preferred.map(|v| v.lower(values)),
        }
    }
}

impl OfferDesc {
    /// Lower this static, string-keyed offer to the id-based [`FormatOffer`] the
    /// solver consumes, interning the family, every field name, and every categorical
    /// value through the pipeline's interners. Called once per pad at link time.
    ///
    /// The result borrows `&'static [FieldConstraint]`: the field table is leaked, as
    /// its interned ids are pipeline-lived and the number of offers is bounded by the
    /// graph, not the stream.
    ///
    /// A [`wildcard`](Self::wildcard) offer lowers to a wildcard [`FormatOffer`] whose
    /// interned family is [`WILDCARD`] ("any"): concrete peers adopt *their* family, and
    /// two wildcards meeting adopt this interned "any" family (spec).
    pub fn lower(
        &self,
        formats: &mut Interner,
        fields: &mut Interner,
        values: &mut Interner,
    ) -> FormatOffer {
        let lowered: Vec<FieldConstraint> =
            self.fields.iter().map(|f| f.lower(fields, values)).collect();
        FormatOffer {
            family: FormatId(formats.intern(self.family)),
            fields: Box::leak(lowered.into_boxed_slice()),
            wildcard: self.is_wildcard(),
        }
    }
}

impl FormatOffer {
    /// The offer that admits **exactly** `fixed` and nothing else: its family, every field
    /// it fixes pinned with [`Constraint::Eq`].
    ///
    /// Used to re-express a format a *pure wildcard transport* is already carrying as an
    /// offer, so the transport's remaining pads negotiate through the ordinary
    /// [`intersect`] path instead of adopting a second, contradictory format (spec:
    /// Formats — a wildcard adopts the peer's family, it does not launder a
    /// contradiction). Leaks its field table exactly as [`OfferDesc::lower`] does, and for
    /// the same reason: the count is bounded by the graph, not the stream.
    pub fn pinned_to(fixed: &FixedFormat) -> FormatOffer {
        let lowered: Vec<FieldConstraint> = fixed
            .fields()
            .iter()
            .map(|(field, v)| FieldConstraint {
                field: *field,
                allowed: Constraint::Eq(*v),
                preferred: Some(*v),
            })
            .collect();
        FormatOffer {
            family: fixed.family,
            fields: Box::leak(lowered.into_boxed_slice()),
            wildcard: false,
        }
    }
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

    /// The fixed `(field, value)` pairs, in insertion order. For the graph dump (spec:
    /// Debuggability — the dump shows, per edge, what was chosen) and for tests that
    /// assert on the negotiated result.
    pub fn fields(&self) -> &[(FieldId, Value)] {
        &self.fields[..self.len as usize]
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
// Link-time format fixation (runs once per link when solving the graph), not per buffer.
#[allow(clippy::disallowed_methods)]
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
///
/// **Wildcards** (spec: Formats — a family-level marker, not a new constraint kind):
/// a wildcard offer "adopts the peer's family". Intersecting a wildcard with a concrete
/// offer yields the concrete side's family *and* its fixated fields (the wildcard adds
/// no constraints); wildcard-vs-wildcard yields the interned "any" family with no
/// fields. A wildcard offer must carry no fields — a wildcard-with-fields is rejected
/// loudly (`None`), because a wildcard pad cannot constrain fields it doesn't know.
pub fn intersect(a: &FormatOffer, b: &FormatOffer) -> Option<FixedFormat> {
    // A wildcard may never carry fields (a wildcard pad cannot constrain fields it does
    // not know — e.g. a link-filtered wildcard). Reject loudly rather than silently
    // dropping the constraints.
    if (a.wildcard && !a.fields.is_empty()) || (b.wildcard && !b.fields.is_empty()) {
        return None;
    }

    match (a.wildcard, b.wildcard) {
        // Both wildcards: adopt the interned "any" family (they share it after lowering,
        // so `a.family == b.family`), no fields to fixate.
        (true, true) => Some(FixedFormat::new(a.family)),
        // One wildcard adopts the concrete peer's family and fields — the wildcard adds
        // no constraints, so this is just the concrete side fixated against `Any`.
        (true, false) => fixate_solo(b),
        (false, true) => fixate_solo(a),
        (false, false) => {
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
    }
}

/// Fixate a single concrete offer as if its peer were fully unconstrained (the wildcard
/// case): adopt its family and fixate each of its fields against `Any`. Never fails
/// except by overflowing the 16-field cap.
fn fixate_solo(concrete: &FormatOffer) -> Option<FixedFormat> {
    let mut out = FixedFormat::new(concrete.family);
    for fc in concrete.fields {
        if let Some(v) = fixate(Some(fc), None)? {
            if !out.set(fc.field, v) {
                return None;
            }
        }
    }
    Some(out)
}

/// Negotiate a link between two pads that each advertise a *list* of alternative
/// offers. Tries `src` offers against `sink` offers in declaration order and returns
/// the [`FixedFormat`] of the first pair that intersects, or `None` if no pair is
/// compatible (a loud `Ready`-time failure). Declaration order is the preference
/// order — a pad lists its favourite offer first.
///
/// A pad with no offers matches nothing; give byte/passthrough pads a single
/// [`OfferDesc::any`] offer so they negotiate against any peer in that family.
pub fn negotiate(src: &[FormatOffer], sink: &[FormatOffer]) -> Option<FixedFormat> {
    for a in src {
        for b in sink {
            if let Some(fixed) = intersect(a, b) {
                return Some(fixed);
            }
        }
    }
    None
}

/// A read-only snapshot of the pipeline's interning tables (spec: Formats). Built once at
/// `run()` after linking has frozen the tables, then shared with every group thread and
/// installed on each `Ctx`. It lets an element resolve the names in its negotiated
/// [`FixedFormat`] — `ctx.field_id("rate")`, `ctx.value_name(id)` — and lets the scheduler
/// turn a string-keyed announcement into a `FixedFormat`, all without touching the live
/// tables (spec: Formats — dynamic caps). Never mutated on a streaming thread.
pub struct Vocabulary {
    pub formats: Interner,
    pub fields: Interner,
    pub values: Interner,
}

impl Vocabulary {
    pub fn family_id(&self, name: &str) -> Option<FormatId> {
        self.formats.get(name).map(FormatId)
    }
    pub fn field_id(&self, name: &str) -> Option<FieldId> {
        self.fields.get(name).map(FieldId)
    }
    pub fn value_id(&self, name: &str) -> Option<ValueId> {
        self.values.get(name).map(ValueId)
    }
    pub fn family_name(&self, id: FormatId) -> Option<&str> {
        self.formats.resolve(id.0)
    }
    pub fn field_name(&self, id: FieldId) -> Option<&str> {
        self.fields.resolve(id.0)
    }
    pub fn value_name(&self, id: ValueId) -> Option<&str> {
        self.values.resolve(id.0)
    }

    /// Resolve a string-keyed runtime announcement into a [`FixedFormat`] (spec: dynamic
    /// caps). `None` if the family or any field/categorical name was never interned (the
    /// element announced something it never offered), so a bogus format is never fixed.
    pub fn build_fixed(
        &self,
        family: &str,
        fields: &[(&'static str, ValueDesc)],
    ) -> Option<FixedFormat> {
        let mut fixed = FixedFormat::new(self.family_id(family)?);
        for (name, vd) in fields {
            let field = self.field_id(name)?;
            let value = match vd {
                ValueDesc::Int(n) => Value::Int(*n),
                ValueDesc::Rat(n, d) => Value::Rat(*n, *d),
                ValueDesc::Id(s) => Value::Id(self.value_id(s)?),
            };
            fixed.set(field, value);
        }
        Some(fixed)
    }

    /// Does this static offer *admit* the concrete `format`? Runtime re-validation of a
    /// dynamic-caps announcement against a peer's declared offer (spec: Formats — an
    /// announced format must still satisfy the downstream, or it is a loud negotiation
    /// failure, never a silent install). The family must match, and for every field the
    /// offer constrains *and* the format fixes, the constraint must accept the value. A
    /// field the offer constrains but the announcement leaves unset is **not** a
    /// conflict — the peer would fixate it, exactly as at link time. Read-only: resolves
    /// names against the frozen tables, interning nothing.
    ///
    /// **A wildcard pad admits any announced format** (spec: Formats — the wildcard
    /// adopts the peer's family). A wildcard offer has no fields and no fixed family, so
    /// there is nothing to conflict with: a `queue`/`tee`-style pad passes every runtime
    /// `FormatChange` straight through re-validation, exactly as it accepts every family
    /// at link time. (A wildcard-with-fields is a contradiction — it admits nothing, the
    /// same loud rejection [`intersect`] gives it.)
    pub fn offer_admits(&self, offer: &OfferDesc, format: &FixedFormat) -> bool {
        if offer.is_wildcard() {
            // A malformed wildcard-with-fields admits nothing (see `intersect`).
            return offer.fields.is_empty();
        }
        match self.family_id(offer.family) {
            Some(fam) if fam == format.family => {}
            _ => return false,
        }
        for fd in offer.fields {
            if let Some(field) = self.field_id(fd.field) {
                if let Some(v) = format.get(field) {
                    if !self.constraint_admits(&fd.allowed, v) {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// Whether *any* of a pad's alternative offers admits `format` (declaration order is
    /// preference, but for validation any match suffices). An empty offer list admits
    /// nothing, mirroring [`negotiate`]. A wildcard offer in the list admits everything
    /// (see [`offer_admits`](Self::offer_admits)).
    pub fn offers_admit(&self, offers: &[OfferDesc], format: &FixedFormat) -> bool {
        offers.iter().any(|o| self.offer_admits(o, format))
    }

    /// Resolve a string-keyed [`ValueDesc`] read-only (`None` if a categorical name was
    /// never interned in this graph).
    fn desc_value(&self, vd: &ValueDesc) -> Option<Value> {
        Some(match *vd {
            ValueDesc::Int(n) => Value::Int(n),
            ValueDesc::Rat(n, d) => Value::Rat(n, d),
            ValueDesc::Id(s) => Value::Id(self.value_id(s)?),
        })
    }

    /// Read-only counterpart of [`Constraint::accepts`] over the string-keyed
    /// [`ConstraintDesc`], resolving categorical names against the frozen tables. An
    /// unresolvable name means the constraint can match no interned value.
    fn constraint_admits(&self, c: &ConstraintDesc, v: Value) -> bool {
        match c {
            ConstraintDesc::Any => true,
            ConstraintDesc::Eq(x) => self.desc_value(x) == Some(v),
            ConstraintDesc::Set(xs) => xs.iter().any(|x| self.desc_value(x) == Some(v)),
            ConstraintDesc::Range { min, max, step } => {
                match (self.desc_value(min), self.desc_value(max), self.desc_value(step)) {
                    (Some(min), Some(max), Some(step)) => {
                        Constraint::Range { min, max, step }.accepts(v)
                    }
                    _ => false,
                }
            }
        }
    }
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
        FormatOffer {
            family: FormatId(family),
            fields: Box::leak(fields.into_boxed_slice()),
            wildcard: false,
        }
    }

    /// A lowered wildcard offer over an interned "any" family id, no fields — what
    /// `OfferDesc::wildcard().lower(..)` produces (both wildcards share this family id).
    fn wildcard(any_family: u32) -> FormatOffer {
        FormatOffer { family: FormatId(any_family), fields: &[], wildcard: true }
    }

    /// A malformed wildcard that (illegally) carries fields — used to prove the loud
    /// rejection. Real code cannot build one: `OfferDesc::wildcard()` has empty fields.
    fn wildcard_with_fields(any_family: u32, fields: Vec<FieldConstraint>) -> FormatOffer {
        FormatOffer {
            family: FormatId(any_family),
            fields: Box::leak(fields.into_boxed_slice()),
            wildcard: true,
        }
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

    // --- wildcard family: adopts the peer's family (spec: family-agnostic elements) ---

    #[test]
    fn offerdesc_wildcard_is_marked_and_lowers_to_wildcard() {
        let wc = OfferDesc::wildcard();
        assert!(wc.is_wildcard());
        assert!(wc.fields.is_empty(), "a wildcard offer carries no fields");
        assert!(!OfferDesc::any("bytes").is_wildcard(), "any() is a concrete family");

        let (mut fmts, mut flds, mut vals) =
            (Interner::new(), Interner::new(), Interner::new());
        let lowered = wc.lower(&mut fmts, &mut flds, &mut vals);
        assert!(lowered.wildcard);
        assert_eq!(lowered.family, FormatId(fmts.get(WILDCARD).unwrap()));
        assert!(lowered.fields.is_empty());
    }

    #[test]
    fn intersect_wildcard_adopts_concrete_family_and_fields_both_orders() {
        // A concrete audio offer (family 5) with two fields, one preferred.
        static RATES: [Value; 2] = [Value::Int(44100), Value::Int(48000)];
        let concrete = || {
            offer(
                5,
                vec![
                    fc(10, Constraint::Set(&RATES), Some(vi(48000))),
                    fc(11, Constraint::Eq(vi(2)), None),
                ],
            )
        };
        // wildcard × concrete and concrete × wildcard both adopt family 5 + fixated fields.
        for (a, b) in [(wildcard(0), concrete()), (concrete(), wildcard(0))] {
            let f = intersect(&a, &b).expect("wildcard adopts the peer");
            assert_eq!(f.family, FormatId(5), "adopted the concrete family");
            assert_eq!(f.get(FieldId(10)), Some(vi(48000)), "preferred fixation");
            assert_eq!(f.get(FieldId(11)), Some(vi(2)));
        }
    }

    #[test]
    fn intersect_wildcard_vs_wildcard_picks_any_family() {
        // Both lowered wildcards share the interned "any" family id (here 7).
        let f = intersect(&wildcard(7), &wildcard(7)).expect("two wildcards intersect");
        assert_eq!(f.family, FormatId(7), "the interned any family");
        assert!(f.is_empty(), "no fields to fixate");
    }

    #[test]
    fn intersect_wildcard_with_fields_is_rejected_loudly() {
        // A wildcard pad cannot constrain fields it does not know — a wildcard-with-fields
        // is a contradiction and must be rejected, in either position.
        let bad = || wildcard_with_fields(0, vec![fc(10, Constraint::Eq(vi(1)), None)]);
        assert!(intersect(&bad(), &offer(5, vec![])).is_none(), "wildcard-with-fields (a)");
        assert!(intersect(&offer(5, vec![]), &bad()).is_none(), "wildcard-with-fields (b)");
        assert!(intersect(&bad(), &wildcard(0)).is_none(), "even vs a clean wildcard");
    }

    #[test]
    fn negotiate_wildcard_pad_links_against_any_family() {
        // A wildcard src pad negotiates against a concrete sink of any family — the
        // queue/tee use case, end-to-end through the descriptor layer.
        let (mut fmts, mut flds, mut vals) =
            (Interner::new(), Interner::new(), Interner::new());
        static SINK_F: [FieldDesc; 1] = [FieldDesc {
            field: "rate",
            allowed: ConstraintDesc::Eq(ValueDesc::Int(48000)),
            preferred: None,
        }];
        let src = [OfferDesc::wildcard().lower(&mut fmts, &mut flds, &mut vals)];
        let sink = [OfferDesc { family: "audio/raw", fields: &SINK_F }
            .lower(&mut fmts, &mut flds, &mut vals)];
        let f = negotiate(&src, &sink).expect("wildcard links against audio/raw");
        assert_eq!(f.family, FormatId(fmts.get("audio/raw").unwrap()));
        assert_eq!(f.get(FieldId(flds.get("rate").unwrap())), Some(vi(48000)));
    }

    // --- negotiate over lists of alternatives ---

    #[test]
    fn negotiate_picks_first_compatible_pair() {
        // src prefers family 1, sink prefers family 2; the only common family is 2, so
        // that's what negotiation must land on regardless of declaration order.
        let src = vec![offer(1, vec![]), offer(2, vec![fc(10, Constraint::Eq(vi(48000)), None)])];
        let sink = vec![offer(2, vec![fc(10, Constraint::Eq(vi(48000)), None)]), offer(3, vec![])];
        let f = negotiate(&src, &sink).unwrap();
        assert_eq!(f.family, FormatId(2));
        assert_eq!(f.get(FieldId(10)), Some(vi(48000)));
    }

    #[test]
    fn negotiate_prefers_earlier_offer() {
        // Both families intersect; the first src offer that finds any sink match wins.
        let src = vec![offer(1, vec![]), offer(2, vec![])];
        let sink = vec![offer(2, vec![]), offer(1, vec![])];
        assert_eq!(negotiate(&src, &sink).unwrap().family, FormatId(1));
    }

    #[test]
    fn negotiate_none_when_disjoint() {
        let src = vec![offer(1, vec![]), offer(2, vec![])];
        let sink = vec![offer(3, vec![]), offer(4, vec![])];
        assert!(negotiate(&src, &sink).is_none());
    }

    #[test]
    fn negotiate_empty_offer_list_matches_nothing() {
        assert!(negotiate(&[], &[offer(1, vec![])]).is_none());
        assert!(negotiate(&[offer(1, vec![])], &[]).is_none());
    }

    // --- the string-keyed descriptor layer + lowering ---

    #[test]
    fn lower_interns_family_field_and_value() {
        let mut fmts = Interner::new();
        let mut flds = Interner::new();
        let mut vals = Interner::new();

        static FIELDS: [FieldDesc; 2] = [
            FieldDesc { field: "rate", allowed: ConstraintDesc::Eq(ValueDesc::Int(48000)), preferred: None },
            FieldDesc {
                field: "layout",
                allowed: ConstraintDesc::Eq(ValueDesc::Id("interleaved")),
                preferred: None,
            },
        ];
        let desc = OfferDesc { family: "audio/raw", fields: &FIELDS };
        let lowered = desc.lower(&mut fmts, &mut flds, &mut vals);

        // Family and fields interned to dense ids; the categorical value too.
        assert_eq!(lowered.family, FormatId(fmts.get("audio/raw").unwrap()));
        assert_eq!(lowered.fields.len(), 2);
        assert_eq!(lowered.fields[0].field, FieldId(flds.get("rate").unwrap()));
        assert!(matches!(lowered.fields[0].allowed, Constraint::Eq(Value::Int(48000))));
        let layout_id = vals.get("interleaved").unwrap();
        assert!(matches!(
            lowered.fields[1].allowed,
            Constraint::Eq(Value::Id(ValueId(id))) if id == layout_id
        ));
    }

    #[test]
    fn lower_is_consistent_across_offers() {
        // The same names lowered twice (as two pads would) intern to the same ids, so
        // the solver sees them as equal — the whole point of link-time interning.
        let (mut fmts, mut flds, mut vals) = (Interner::new(), Interner::new(), Interner::new());
        static F: [FieldDesc; 1] =
            [FieldDesc { field: "rate", allowed: ConstraintDesc::Any, preferred: None }];
        let a = OfferDesc { family: "audio/raw", fields: &F }.lower(&mut fmts, &mut flds, &mut vals);
        let b = OfferDesc { family: "audio/raw", fields: &F }.lower(&mut fmts, &mut flds, &mut vals);
        assert_eq!(a.family, b.family);
        assert_eq!(a.fields[0].field, b.fields[0].field);
    }

    #[test]
    fn lower_set_interns_every_value() {
        let (mut fmts, mut flds, mut vals) = (Interner::new(), Interner::new(), Interner::new());
        static FMTS: [ValueDesc; 3] =
            [ValueDesc::Id("s16"), ValueDesc::Id("s24"), ValueDesc::Id("f32")];
        static F: [FieldDesc; 1] =
            [FieldDesc { field: "sample", allowed: ConstraintDesc::Set(&FMTS), preferred: None }];
        let lowered =
            OfferDesc { family: "audio/raw", fields: &F }.lower(&mut fmts, &mut flds, &mut vals);
        let Constraint::Set(vs) = lowered.fields[0].allowed else {
            panic!("expected a Set");
        };
        assert_eq!(vs.len(), 3);
        // All three names are now interned and the lowered set carries their ids.
        for name in ["s16", "s24", "f32"] {
            let id = vals.get(name).expect("value interned");
            assert!(vs.contains(&Value::Id(ValueId(id))));
        }
    }

    /// A tiny audio negotiation solved end-to-end through the descriptor layer: two
    /// pads' string-keyed offers, lowered through shared interners, then negotiated.
    #[test]
    fn descriptor_to_negotiated_audio() {
        let (mut fmts, mut flds, mut vals) = (Interner::new(), Interner::new(), Interner::new());
        // src: 44.1k or 48k, prefers 48k; sink: fixed 48k. Expect 48000.
        static SRC_RATES: [ValueDesc; 2] = [ValueDesc::Int(44100), ValueDesc::Int(48000)];
        static SRC_F: [FieldDesc; 1] = [FieldDesc {
            field: "rate",
            allowed: ConstraintDesc::Set(&SRC_RATES),
            preferred: Some(ValueDesc::Int(48000)),
        }];
        static SINK_F: [FieldDesc; 1] = [FieldDesc {
            field: "rate",
            allowed: ConstraintDesc::Eq(ValueDesc::Int(48000)),
            preferred: None,
        }];
        let src = [OfferDesc { family: "audio/raw", fields: &SRC_F }
            .lower(&mut fmts, &mut flds, &mut vals)];
        let sink = [OfferDesc { family: "audio/raw", fields: &SINK_F }
            .lower(&mut fmts, &mut flds, &mut vals)];

        let f = negotiate(&src, &sink).expect("compatible");
        assert_eq!(f.family, FormatId(fmts.get("audio/raw").unwrap()));
        assert_eq!(f.get(FieldId(flds.get("rate").unwrap())), Some(vi(48000)));
    }

    /// Table-driven fuzz-style regression: (src family, sink family) pairs → expected
    /// negotiability, so a future change to `negotiate`/`intersect` that silently
    /// widens or narrows matching trips a case here.
    #[test]
    fn negotiate_table() {
        // Each row: src offer families, sink offer families, expected Some family.
        let cases: &[(&[u32], &[u32], Option<u32>)] = &[
            (&[1], &[1], Some(1)),
            (&[1], &[2], None),
            (&[1, 2], &[2, 1], Some(1)),  // first src that matches anything wins
            (&[3, 1], &[1, 3], Some(3)),  // ...even if the shared family is listed later on sink
            (&[1, 2, 3], &[3], Some(3)),
            (&[], &[1], None),
            (&[1], &[], None),
            (&[5, 6], &[7, 8], None),
        ];
        for (i, (sa, sb, want)) in cases.iter().enumerate() {
            let src: Vec<_> = sa.iter().map(|&f| offer(f, vec![])).collect();
            let sink: Vec<_> = sb.iter().map(|&f| offer(f, vec![])).collect();
            let got = negotiate(&src, &sink).map(|f| f.family.0);
            assert_eq!(got, *want, "case {i}: src={sa:?} sink={sb:?}");
        }
    }

    // --- runtime re-validation (dynamic caps): Vocabulary::offer_admits ---

    fn vocab_with(families: &[&str], fields: &[&str], values: &[&str]) -> Vocabulary {
        let (mut formats, mut flds, mut vals) =
            (Interner::new(), Interner::new(), Interner::new());
        for f in families {
            formats.intern(f);
        }
        for f in fields {
            flds.intern(f);
        }
        for v in values {
            vals.intern(v);
        }
        Vocabulary { formats, fields: flds, values: vals }
    }

    #[test]
    fn offer_admits_family_and_value_conflict() {
        let vocab = vocab_with(&["audio/raw", "video/raw"], &["rate"], &[]);
        let fam = vocab.family_id("audio/raw").unwrap();
        let rate = vocab.field_id("rate").unwrap();
        let mut f = FixedFormat::new(fam);
        f.set(rate, Value::Int(48000));

        // rate Eq 44100 rejects the announced 48000 — the loud-error case.
        static STRICT: [FieldDesc; 1] = [FieldDesc {
            field: "rate",
            allowed: ConstraintDesc::Eq(ValueDesc::Int(44100)),
            preferred: None,
        }];
        assert!(!vocab.offer_admits(&OfferDesc { family: "audio/raw", fields: &STRICT }, &f));

        // rate Set{44100,48000} admits it.
        static SET: [ValueDesc; 2] = [ValueDesc::Int(44100), ValueDesc::Int(48000)];
        static OK: [FieldDesc; 1] = [FieldDesc {
            field: "rate",
            allowed: ConstraintDesc::Set(&SET),
            preferred: None,
        }];
        assert!(vocab.offer_admits(&OfferDesc { family: "audio/raw", fields: &OK }, &f));

        // Family mismatch is a conflict; an unconstrained same-family offer admits.
        assert!(!vocab.offer_admits(&OfferDesc::any("video/raw"), &f));
        assert!(vocab.offer_admits(&OfferDesc::any("audio/raw"), &f));
    }

    #[test]
    fn offer_admits_unset_field_is_not_a_conflict() {
        // The offer constrains a field the announcement never fixed — the peer fixates
        // it (as at link time), so this must NOT be rejected.
        let vocab = vocab_with(&["audio/raw"], &["rate", "channels"], &[]);
        let fam = vocab.family_id("audio/raw").unwrap();
        let rate = vocab.field_id("rate").unwrap();
        let mut f = FixedFormat::new(fam);
        f.set(rate, Value::Int(44100)); // channels deliberately unset

        static CH: [FieldDesc; 1] = [FieldDesc {
            field: "channels",
            allowed: ConstraintDesc::Eq(ValueDesc::Int(2)),
            preferred: None,
        }];
        assert!(vocab.offer_admits(&OfferDesc { family: "audio/raw", fields: &CH }, &f));
    }

    #[test]
    fn offer_admits_categorical_value() {
        let vocab = vocab_with(&["audio/raw"], &["sample"], &["s16", "s24"]);
        let fam = vocab.family_id("audio/raw").unwrap();
        let sample = vocab.field_id("sample").unwrap();
        let s16 = vocab.value_id("s16").unwrap();
        let mut f = FixedFormat::new(fam);
        f.set(sample, Value::Id(s16));

        static OKV: [ValueDesc; 1] = [ValueDesc::Id("s16")];
        static OKF: [FieldDesc; 1] = [FieldDesc {
            field: "sample",
            allowed: ConstraintDesc::Set(&OKV),
            preferred: None,
        }];
        assert!(vocab.offer_admits(&OfferDesc { family: "audio/raw", fields: &OKF }, &f));

        static BADF: [FieldDesc; 1] = [FieldDesc {
            field: "sample",
            allowed: ConstraintDesc::Eq(ValueDesc::Id("s24")),
            preferred: None,
        }];
        assert!(!vocab.offer_admits(&OfferDesc { family: "audio/raw", fields: &BADF }, &f));
    }

    #[test]
    fn offers_admit_needs_one_match_and_empty_admits_nothing() {
        let vocab = vocab_with(&["audio/raw"], &["rate"], &[]);
        let fam = vocab.family_id("audio/raw").unwrap();
        let rate = vocab.field_id("rate").unwrap();
        let mut f = FixedFormat::new(fam);
        f.set(rate, Value::Int(48000));

        static A: [FieldDesc; 1] = [FieldDesc {
            field: "rate",
            allowed: ConstraintDesc::Eq(ValueDesc::Int(44100)),
            preferred: None,
        }];
        static B: [FieldDesc; 1] = [FieldDesc {
            field: "rate",
            allowed: ConstraintDesc::Eq(ValueDesc::Int(48000)),
            preferred: None,
        }];
        let alts = [
            OfferDesc { family: "audio/raw", fields: &A },
            OfferDesc { family: "audio/raw", fields: &B },
        ];
        assert!(vocab.offers_admit(&alts, &f), "second alternative admits it");
        assert!(!vocab.offers_admit(&[], &f), "no offers admit nothing");
    }

    #[test]
    fn wildcard_pad_admits_any_announced_format() {
        // A wildcard pad passes every runtime FormatChange through re-validation — it
        // adopts the peer's family, so there is nothing to conflict with. This is what
        // lets a dynamic-caps announcement cross a `queue`/`tee`.
        let vocab = vocab_with(&["audio/raw", "video/raw"], &["rate", "width"], &["s16"]);
        let mut audio = FixedFormat::new(vocab.family_id("audio/raw").unwrap());
        audio.set(vocab.field_id("rate").unwrap(), Value::Int(48000));
        let mut video = FixedFormat::new(vocab.family_id("video/raw").unwrap());
        video.set(vocab.field_id("width").unwrap(), Value::Int(1920));

        let wc = OfferDesc::wildcard();
        assert!(vocab.offer_admits(&wc, &audio), "wildcard admits audio");
        assert!(vocab.offer_admits(&wc, &video), "wildcard admits video (a different family)");
        // Even a family the vocabulary never interned is admitted (nothing resolved).
        let alien = FixedFormat::new(FormatId(9999));
        assert!(vocab.offer_admits(&wc, &alien), "wildcard admits an unknown family");
    }
}
