//! An owned, interner-free image of the pipeline topology (spec: server architecture —
//! "ALL interned ids resolved to owned strings at build time — server never touches
//! pipeline interners"). Built on the app thread by
//! [`Pipeline::introspect_snapshot`](crate::pipeline::Pipeline::introspect_snapshot)
//! after topology is built (and rebuilt after preroll/links, so dynamic pads are
//! visible), then published behind a `Mutex<Arc<..>>` the server reads.
//!
//! Everything here is a plain owned value: strings resolved once, formats flattened.
//! The server serializes these rows without ever locking or resolving anything on the
//! pipeline — the snapshot is the boundary.

/// A `(field_name, value)` pair on an edge's fixed format, names already resolved.
#[derive(Clone, Debug)]
pub struct SnapField {
    pub field_name: String,
    /// Wire value tag (1 Int, 2 Rat, 3 Id) — see [`WireValue`](super::wire::WireValue).
    pub tag: u8,
    pub bits: u64,
    /// For an `Id` value, the resolved categorical name (so the server interns it into
    /// the connection string table). Empty for Int/Rat.
    pub id_name: String,
}

/// One element, everything the server needs to build an `ElementRow` and its pads.
#[derive(Clone, Debug)]
pub struct SnapElement {
    pub id: u32,
    pub name: String,
    pub group: u32,
    /// 0 passive, 1 active.
    pub sched: u8,
    pub is_source: bool,
    pub is_sink: bool,
    pub is_live: bool,
    pub latency_min_ns: u64,
    pub latency_max_ns: u64,
    pub jitter_ns: u64,
    pub pads: Vec<SnapPad>,
}

/// One pad on an element.
#[derive(Clone, Debug)]
pub struct SnapPad {
    pub pad: u32,
    pub name: String,
    /// 0 sink, 1 src (Direction ordinal).
    pub direction: u8,
    pub dynamic: bool,
    pub linked: bool,
}

/// A negotiated edge with its fixed format flattened to owned names.
#[derive(Clone, Debug)]
pub struct SnapEdge {
    pub src: u32,
    pub src_pad: u32,
    pub sink: u32,
    pub sink_pad: u32,
    pub family_name: String,
    pub fields: Vec<SnapField>,
}

/// One property's descriptor + current value, names resolved.
#[derive(Clone, Debug)]
pub struct SnapProp {
    pub element: u32,
    pub prop_index: u16,
    pub name: String,
    pub live: bool,
    /// 0 Any, 1 Eq, 2 Range, 3 Set — see [`PropRow`](super::wire::PropRow).
    pub ckind: u8,
    /// The constraint's allowed values (Eq⇒1, Range⇒3 min/max/step, Set⇒len, Any⇒0).
    pub allowed: Vec<SnapValue>,
    /// The current value, or `None` if never set.
    pub current: Option<SnapValue>,
}

/// A resolved `Value` (from `format::Value` or a `Constraint` bound). Categorical ids
/// carry the resolved name so the server can put it in the connection string table.
#[derive(Clone, Debug)]
pub struct SnapValue {
    /// Wire value tag (1 Int, 2 Rat, 3 Id).
    pub tag: u8,
    pub bits: u64,
    /// For `Id`, the resolved categorical name; empty otherwise. When the server sends
    /// an `Id` value it emits a StrDef for this name and puts the connection string id
    /// in `bits` (spec: WireValue tag 3 = connection string id). The `bits` here is the
    /// pipeline `ValueId`, used to map back a `SetProp` with tag=Id.
    pub id_name: String,
}

/// The full topology image: elements, edges, props, and the `dump_dot` string. Owned,
/// interner-free, cheap to `Arc`-share and republish.
#[derive(Clone, Debug)]
pub struct TopologySnapshot {
    pub pid: u32,
    pub elements: Vec<SnapElement>,
    pub edges: Vec<SnapEdge>,
    pub props: Vec<SnapProp>,
    pub dot: String,
    /// Latency report (spec: LatencyReport), flattened.
    pub latency_paths: Vec<SnapLatencyPath>,
}

/// One flattened latency path.
#[derive(Clone, Debug)]
pub struct SnapLatencyPath {
    pub sink: u32,
    pub is_live: bool,
    pub total_ns: u64,
    pub elems: Vec<(u32, u64)>, // (element, min_ns)
}
