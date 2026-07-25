//! Layered (Sugiyama-style) DAG layout for the scraft-scope graph view.
//!
//! Pure, deterministic, dependency-free (std only, no `unsafe`). Topology in
//! ([`Node`] + [`Edge`]), pixel positions out ([`Layout`]). Called once per
//! topology change; graphs are small (< a few hundred nodes), so a fixed number
//! of barycenter sweeps is plenty — we do not chase optimality.
//!
//! **Flow direction is left → right** (media convention: sources on the left,
//! sinks on the right). Layers map to `x`; order within a layer maps to `y`.
//!
//! The pipeline is the classic three-phase layered ("hierarchical") drawing of
//!
//!   K. Sugiyama, S. Tagawa & M. Toda, "Methods for Visual Understanding of
//!   Hierarchical System Structures", IEEE Trans. Systems, Man, and Cybernetics,
//!   SMC-11(2):109–125, 1981.
//!
//! Each phase cites the specific heuristic it uses at its point of use below.
//! This is a clean-room implementation from the papers named here — nothing is
//! ported from graphviz/dagre or any other layout library.

use std::collections::BTreeMap;

/// A node to place. `id` is caller-defined and only needs to be unique; `w`/`h`
/// are the box size in pixels; `group` clusters nodes by thread group (used for
/// contiguity + cluster hulls). Ports are addressed by index in the edges.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Node {
    pub id: u32,
    pub w: f32,
    pub h: f32,
    pub group: u32,
}

/// A directed edge from `src`'s output port `src_port` to `dst`'s input port
/// `dst_port`. Ports order the attach points vertically on multi-pad nodes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Edge {
    pub src: u32,
    pub src_port: u16,
    pub dst: u32,
    pub dst_port: u16,
}

/// Layout tuning. All distances in pixels.
#[derive(Clone, Copy, Debug)]
pub struct Opts {
    /// Horizontal gap between successive layers (added to node widths).
    pub layer_gap: f32,
    /// Minimum vertical gap between two nodes stacked in the same layer.
    pub node_gap: f32,
    /// Vertical width reserved for a routing dummy node (a long edge's waypoint).
    pub dummy_h: f32,
    /// Number of barycenter sweeps for crossing reduction. Even ⇒ ends on a
    /// down (forward) sweep. 8 is ample for these small graphs.
    pub sweeps: u32,
    /// Extra vertical gap (on top of `node_gap`) between vertically adjacent
    /// nodes of *different* groups, so cluster hulls drawn around the groups
    /// (padding + a label strip) never overlap each other.
    pub group_gap: f32,
    /// Top-left origin of the whole drawing.
    pub origin_x: f32,
    pub origin_y: f32,
}

impl Default for Opts {
    fn default() -> Self {
        Opts {
            layer_gap: 60.0,
            node_gap: 20.0,
            dummy_h: 12.0,
            sweeps: 8,
            group_gap: 48.0,
            origin_x: 0.0,
            origin_y: 0.0,
        }
    }
}

/// A placed node: top-left pixel position + the size it was placed with.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Placed {
    pub id: u32,
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
    /// 0-based layer index (left→right). Useful to the UI for column framing.
    pub layer: u32,
}

/// The routed polyline for one edge, from the source port's attach point on the
/// right side of `src` to the destination port's attach point on the left side
/// of `dst`, passing through any dummy waypoints for spans > 1 layer.
#[derive(Clone, Debug, PartialEq)]
pub struct EdgeRoute {
    pub src: u32,
    pub src_port: u16,
    pub dst: u32,
    pub dst_port: u16,
    /// ≥ 2 points: first is on `src`'s right edge, last on `dst`'s left edge.
    pub points: Vec<(f32, f32)>,
}

/// A group's axis-aligned bounding box over its placed nodes, for cluster hulls.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GroupBox {
    pub group: u32,
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

/// The full result: placed nodes, edge routes, per-group boxes.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Layout {
    pub nodes: Vec<Placed>,
    pub edges: Vec<EdgeRoute>,
    pub groups: Vec<GroupBox>,
}

// ---------------------------------------------------------------------------
// Internal representation
// ---------------------------------------------------------------------------

/// A vertex in the working graph — either a real node or a routing dummy.
struct Vert {
    /// `Some(id)` for a real node, `None` for a routing dummy.
    id: Option<u32>,
    w: f32,
    h: f32,
    group: u32,
    layer: u32,
    /// Position within its layer (0-based rank), set by ordering.
    order: usize,
    /// Assigned centre-y (top-left y is `y - h/2` when emitted). Set in phase 3.
    cy: f32,
    /// Original edge this dummy belongs to (`None` for real nodes). Lets us
    /// stitch a long edge's waypoints back together in order.
    of_edge: Option<usize>,
    /// For dummies: waypoint index along its edge (to sort waypoints).
    seq: u32,
}

/// One edge in the working graph, after cycles broken and endpoints validated,
/// carrying the original `src_port`/`dst_port` for attach ordering.
struct WEdge {
    from: usize,
    to: usize,
    /// Index of the originating [`Edge`] in the caller's slice.
    orig: usize,
    src_port: u16,
    dst_port: u16,
}

/// Build the layout. Deterministic: identical inputs always yield identical
/// output, and there is no reliance on hash iteration order (we key everything
/// on sorted `Vec`/`BTreeMap`). Robust to ill-formed input:
///
/// - **Empty input** ⇒ empty output.
/// - **Unknown edge endpoints** (an `src`/`dst` not in `nodes`) ⇒ that edge is
///   skipped.
/// - **Duplicate node ids** ⇒ the first occurrence wins; later ones are dropped.
/// - **Cycles** ⇒ broken by ignoring back-edges discovered in a DFS (see
///   [`break_cycles`]); the layout is of the remaining DAG, so it always
///   terminates and never loops forever.
pub fn layout(nodes: &[Node], edges: &[Edge], opts: &Opts) -> Layout {
    // --- Index nodes by id (first wins), deterministically. -----------------
    // `index[id] = position in verts`. BTreeMap keeps this independent of input
    // order for the "first wins" tiebreak and gives us a sorted id space.
    let mut id_to_vert: BTreeMap<u32, usize> = BTreeMap::new();
    let mut verts: Vec<Vert> = Vec::with_capacity(nodes.len());
    for n in nodes {
        if id_to_vert.contains_key(&n.id) {
            continue; // duplicate id: keep the first
        }
        id_to_vert.insert(n.id, verts.len());
        verts.push(Vert {
            id: Some(n.id),
            w: n.w,
            h: n.h,
            group: n.group,
            layer: 0,
            order: 0,
            cy: 0.0,
            of_edge: None,
            seq: 0,
        });
    }
    if verts.is_empty() {
        return Layout::default();
    }

    // --- Validate + dedup edges (skip unknown endpoints, drop self-loops). ---
    // A self-loop can never satisfy dst_layer > src_layer, so we drop it here
    // rather than let cycle-breaking silently swallow it.
    let mut wedges: Vec<WEdge> = Vec::with_capacity(edges.len());
    for (i, e) in edges.iter().enumerate() {
        let (from, to) = match (id_to_vert.get(&e.src), id_to_vert.get(&e.dst)) {
            (Some(&f), Some(&t)) if f != t => (f, t),
            _ => continue,
        };
        wedges.push(WEdge {
            from,
            to,
            orig: i,
            src_port: e.src_port,
            dst_port: e.dst_port,
        });
    }

    // --- Phase 0: break cycles so the rest sees a DAG. ----------------------
    break_cycles(&mut wedges, verts.len());

    // --- Phase 1: layer assignment by longest path from sources. ------------
    assign_layers(&mut verts, &wedges);

    // --- Insert routing dummies for edges spanning > 1 layer. ---------------
    // Each long edge (dst_layer - src_layer > 1) is subdivided into unit-length
    // segments through one dummy vertex per intermediate layer, so that crossing
    // reduction and coordinate assignment "see" the long edge as a chain. This
    // is Sugiyama et al. (1981), §III — the standard dummy-vertex device.
    let mut chain: Vec<WEdge> = Vec::new(); // unit-length edges of the working graph
    for we in &wedges {
        let (lo, hi) = (verts[we.from].layer, verts[we.to].layer);
        if hi <= lo + 1 {
            chain.push(WEdge {
                from: we.from,
                to: we.to,
                orig: we.orig,
                src_port: we.src_port,
                dst_port: we.dst_port,
            });
            continue;
        }
        // Create dummies on layers lo+1 .. hi-1, chaining them. `seq` (the
        // waypoint index along the edge) is just the enumeration counter.
        let mut prev = we.from;
        let group = verts[we.from].group; // dummies inherit source's group
        for (seq, l) in ((lo + 1)..hi).enumerate() {
            let d = verts.len();
            verts.push(Vert {
                id: None,
                w: 0.0,
                h: opts.dummy_h,
                group,
                layer: l,
                order: 0,
                cy: 0.0,
                of_edge: Some(we.orig),
                seq: seq as u32,
            });
            chain.push(WEdge {
                from: prev,
                to: d,
                orig: we.orig,
                src_port: we.src_port,
                dst_port: we.dst_port,
            });
            prev = d;
        }
        chain.push(WEdge {
            from: prev,
            to: we.to,
            orig: we.orig,
            src_port: we.src_port,
            dst_port: we.dst_port,
        });
    }

    // --- Build per-layer vertex lists + adjacency (indices only). -----------
    let num_layers = verts.iter().map(|v| v.layer).max().unwrap_or(0) + 1;
    let mut layers: Vec<Vec<usize>> = vec![Vec::new(); num_layers as usize];
    for (vi, v) in verts.iter().enumerate() {
        layers[v.layer as usize].push(vi);
    }
    // Initial in-layer order: stable by (group, id/dummy-seq) for determinism
    // and to seed group contiguity. We sort by group first so same-group nodes
    // start adjacent; ties break on a stable key.
    for layer in &mut layers {
        layer.sort_by(|&a, &b| order_key(&verts[a]).cmp(&order_key(&verts[b])));
    }
    // Adjacency: `up[v]` = neighbours on the layer to the left (sources),
    // `down[v]` = neighbours on the layer to the right (targets).
    let mut up: Vec<Vec<usize>> = vec![Vec::new(); verts.len()];
    let mut down: Vec<Vec<usize>> = vec![Vec::new(); verts.len()];
    for e in &chain {
        down[e.from].push(e.to);
        up[e.to].push(e.from);
    }

    // --- Phase 2: crossing reduction by iterated barycenter sweeps. ---------
    reduce_crossings(&mut verts, &mut layers, &up, &down, opts.sweeps);

    // --- Phase 3: coordinate assignment (y within layer, x per layer). ------
    assign_coordinates(&mut verts, &layers, &up, &down, opts);

    // --- Emit. --------------------------------------------------------------
    emit(&verts, &wedges, &chain, &layers, opts)
}

/// Ordering key for the initial in-layer sort and group tiebreak: `(group,
/// real-before-dummy, id-or-seq)`. Deterministic and total.
fn order_key(v: &Vert) -> (u32, u8, u32) {
    match v.id {
        Some(id) => (v.group, 0, id),
        None => (v.group, 1, v.seq),
    }
}

/// Break cycles by ignoring back-edges found in an iterative DFS.
///
/// We compute a DFS forest over the vertices (visiting them and their
/// out-neighbours in deterministic index order). An edge whose target is
/// currently on the DFS stack ("grey") is a back-edge closing a cycle; we drop
/// it. All other edges are kept. The retained graph is acyclic, so every later
/// phase terminates. This is the standard DFS back-edge removal; we accept that
/// it is heuristic (not minimum feedback arc set — that is NP-hard) which is
/// fine for a display graph.
fn break_cycles(wedges: &mut Vec<WEdge>, n: usize) {
    // Deterministic adjacency: out-edges per vertex, in insertion order.
    let mut out: Vec<Vec<usize>> = vec![Vec::new(); n]; // stores edge indices
    for (i, e) in wedges.iter().enumerate() {
        out[e.from].push(i);
    }
    // Colours: 0 = white (unseen), 1 = grey (on stack), 2 = black (done).
    let mut colour = vec![0u8; n];
    let mut drop_edge = vec![false; wedges.len()];
    // Iterative DFS to avoid stack overflow on deep chains.
    // Stack frame: (vertex, next out-edge cursor).
    for start in 0..n {
        if colour[start] != 0 {
            continue;
        }
        let mut stack: Vec<(usize, usize)> = vec![(start, 0)];
        colour[start] = 1;
        while let Some(&mut (v, ref mut cur)) = stack.last_mut() {
            if *cur < out[v].len() {
                let ei = out[v][*cur];
                *cur += 1;
                let w = wedges[ei].to;
                match colour[w] {
                    0 => {
                        colour[w] = 1;
                        stack.push((w, 0));
                    }
                    1 => {
                        // Back-edge → part of a cycle; drop it.
                        drop_edge[ei] = true;
                    }
                    _ => {} // black: forward/cross edge, keep.
                }
            } else {
                colour[v] = 2;
                stack.pop();
            }
        }
    }
    let mut kept = Vec::with_capacity(wedges.len());
    for (i, e) in wedges.drain(..).enumerate() {
        if !drop_edge[i] {
            kept.push(e);
        }
    }
    *wedges = kept;
}

/// Longest-path layering: a source (no in-edges) is at layer 0; every other
/// vertex sits one past the deepest of its predecessors. On a DAG this is the
/// classic longest-path layer assignment (Sugiyama et al. 1981, §II) and puts
/// every edge strictly forward (dst_layer > src_layer). We compute it by
/// processing vertices in topological order.
fn assign_layers(verts: &mut [Vert], wedges: &[WEdge]) {
    let n = verts.len();
    let mut indeg = vec![0u32; n];
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for e in wedges {
        adj[e.from].push(e.to);
        indeg[e.to] += 1;
    }
    // Kahn's algorithm, seeding roots in ascending index for determinism.
    let mut queue: Vec<usize> = (0..n).filter(|&v| indeg[v] == 0).collect();
    let mut head = 0;
    // layer starts at 0 for everyone; relax as we go.
    for v in verts.iter_mut() {
        v.layer = 0;
    }
    while head < queue.len() {
        let v = queue[head];
        head += 1;
        let lv = verts[v].layer;
        for &w in &adj[v] {
            if verts[w].layer < lv + 1 {
                verts[w].layer = lv + 1;
            }
            indeg[w] -= 1;
            if indeg[w] == 0 {
                queue.push(w);
            }
        }
    }
    // (If break_cycles left the graph acyclic, every vertex is processed. Any
    //  vertex not reached — impossible on a true DAG — keeps layer 0.)
}

/// Phase 2: crossing reduction via iterated one-sided barycenter ordering.
///
/// Barycenter heuristic (Sugiyama, Tagawa & Toda 1981, §IV): repeatedly reorder
/// each layer by the average position ("barycenter") of each vertex's neighbours
/// in the adjacent fixed layer, alternating sweep direction (down = order by the
/// left neighbours, up = order by the right neighbours). Vertices with no
/// neighbour on the reference side keep their current position (barycenter = own
/// order). Ties — and the group tiebreak — are resolved by a **stable** sort, so
/// same-group nodes stay contiguous where crossings don't force them apart.
fn reduce_crossings(
    verts: &mut [Vert],
    layers: &mut [Vec<usize>],
    up: &[Vec<usize>],
    down: &[Vec<usize>],
    sweeps: u32,
) {
    // Seed each vertex's `order` from the initial per-layer arrangement.
    for layer in layers.iter() {
        for (rank, &vi) in layer.iter().enumerate() {
            verts[vi].order = rank;
        }
    }
    let nl = layers.len();
    if nl <= 1 {
        return;
    }
    for s in 0..sweeps {
        let down_sweep = s % 2 == 0;
        if down_sweep {
            // Left→right: order layer l by barycenter of its up-neighbours.
            for l in 1..nl {
                reorder_layer(verts, layers, up, l);
            }
        } else {
            // Right→left: order layer l by barycenter of its down-neighbours.
            for l in (0..nl - 1).rev() {
                reorder_layer(verts, layers, down, l);
            }
        }
    }
}

/// Reorder one layer by the barycenter of each vertex's neighbours on `refside`
/// (whose `order` fields are already fixed for this sweep). Stable sort keeps
/// the group tiebreak and leaves neighbourless vertices where they were.
fn reorder_layer(verts: &mut [Vert], layers: &mut [Vec<usize>], refside: &[Vec<usize>], l: usize) {
    let layer = &mut layers[l];
    // Compute a sort key per vertex: (barycenter, group, current-rank).
    // Neighbourless ⇒ barycenter = own current rank (keeps it put).
    let mut keyed: Vec<(f32, u32, usize, usize)> = layer
        .iter()
        .enumerate()
        .map(|(rank, &vi)| {
            let nbrs = &refside[vi];
            let bc = if nbrs.is_empty() {
                verts[vi].order as f32
            } else {
                let sum: f32 = nbrs.iter().map(|&u| verts[u].order as f32).sum();
                sum / nbrs.len() as f32
            };
            (bc, verts[vi].group, rank, vi)
        })
        .collect();
    // Sort by barycenter, then group (contiguity tiebreak), then original rank
    // (stability). Total order ⇒ deterministic.
    keyed.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.cmp(&b.1))
            .then(a.2.cmp(&b.2))
    });
    for (rank, &(_, _, _, vi)) in keyed.iter().enumerate() {
        layer[rank] = vi;
        verts[vi].order = rank;
    }
}

/// Phase 3: coordinate assignment.
///
/// - **x** is per-layer: layer `l`'s left edge is the running sum of previous
///   layers' max widths plus `layer_gap` each. Every node in a layer shares the
///   layer's left x (so a column lines up); its own width still varies.
/// - **y** aligns each vertex to the barycenter of its neighbours, then a single
///   downward pass removes overlaps by pushing nodes down to respect `node_gap`
///   and node heights. This is the simple "priority/barycenter alignment then
///   separation" coordinate step in the spirit of Sugiyama et al. (1981) §V — we
///   deliberately keep it O(V) and non-iterative rather than solving the full QP.
fn assign_coordinates(
    verts: &mut [Vert],
    layers: &[Vec<usize>],
    up: &[Vec<usize>],
    down: &[Vec<usize>],
    opts: &Opts,
) {
    // First, seed cy from the stacked order so the barycenter pass has a
    // starting position: stack each layer top-down with node_gap (+ the group
    // gap across group boundaries).
    for layer in layers {
        let mut y = opts.origin_y;
        let mut prev_group: Option<u32> = None;
        for &vi in layer {
            let h = verts[vi].h;
            if prev_group.is_some_and(|g| g != verts[vi].group) {
                y += opts.group_gap;
            }
            verts[vi].cy = y + h / 2.0;
            y += h + opts.node_gap;
            prev_group = Some(verts[vi].group);
        }
    }
    // A few alignment passes: pull each node toward the barycenter of its
    // neighbours on one side, then re-separate within the layer. We alternate
    // the reference side each pass AND sweep in the matching direction so that,
    // within a pass, a layer is aligned only *after* its reference-side
    // neighbours have already been placed this pass:
    //
    //   - down pass: align to up-neighbours (left side), sweep left→right;
    //   - up pass:   align to down-neighbours (right side), sweep right→left.
    //
    // This is the "priority/barycenter alignment" coordinate step in the spirit
    // of Sugiyama et al. (1981) §V; bounded to a fixed pass count (no
    // convergence hunt) since these graphs are small.
    let nl = layers.len();
    for pass in 0..6u32 {
        let align_up = pass % 2 == 0; // reference = up-neighbours (left)
        // Iterate layers in the settling direction. `enumerate` over a chosen
        // index sequence so each layer sees finalized reference neighbours.
        let seq: Vec<usize> = if align_up {
            (0..nl).collect()
        } else {
            (0..nl).rev().collect()
        };
        for &li in &seq {
            for &vi in &layers[li] {
                let nbrs = if align_up { &up[vi] } else { &down[vi] };
                if !nbrs.is_empty() {
                    let sum: f32 = nbrs.iter().map(|&u| verts[u].cy).sum();
                    verts[vi].cy = sum / nbrs.len() as f32;
                }
            }
            separate_layer(verts, &layers[li], opts);
        }
    }
    // Final separation pass to guarantee no overlaps after the last alignment.
    for layer in layers {
        separate_layer(verts, layer, opts);
    }
}

/// Resolve vertical overlaps within one layer, preserving in-layer order and
/// (crucially) the layer's *barycenter*, so symmetric fan-in/fan-out stays
/// visually centred.
///
/// Nodes are in their layer's order. We enforce that consecutive nodes are
/// separated by `(h_i + h_{i+1})/2 + node_gap`. A naïve top-down push would
/// drift the whole stack downward; instead we (1) push down to fix gaps, then
/// (2) shift the whole rigid block back up toward each node's *desired* cy by
/// the mean displacement, clamped so no node rises above `origin_y`. Because the
/// shift is the mean of desired−actual, a symmetric layer keeps its centre
/// exactly. O(k), monotone in the sense that gap constraints are always met.
fn separate_layer(verts: &mut [Vert], layer: &[usize], opts: &Opts) {
    if layer.is_empty() {
        return;
    }
    // Record desired positions before we perturb them.
    let desired: Vec<f32> = layer.iter().map(|&vi| verts[vi].cy).collect();

    // (1) Downward sweep: ensure each node is at least min-gap below the prev.
    // Crossing a group boundary widens the gap by `group_gap` so the cluster
    // hulls (node padding + label strip) drawn around groups cannot overlap.
    let mut prev_bottom_center = f32::NEG_INFINITY;
    let mut prev_h = 0.0f32;
    let mut prev_group = 0u32;
    let mut first = true;
    for &vi in layer {
        let h = verts[vi].h;
        if !first {
            let extra = if prev_group != verts[vi].group { opts.group_gap } else { 0.0 };
            let min_cy = prev_bottom_center + prev_h / 2.0 + opts.node_gap + extra + h / 2.0;
            if verts[vi].cy < min_cy {
                verts[vi].cy = min_cy;
            }
        }
        first = false;
        prev_bottom_center = verts[vi].cy;
        prev_h = h;
        prev_group = verts[vi].group;
    }

    // (2) Recentre the (now rigid, gap-satisfying) block toward desired. The
    // ideal uniform shift is mean(desired - actual); apply it but clamp so the
    // topmost node does not go above origin_y.
    let mut shift = 0.0f32;
    for (k, &vi) in layer.iter().enumerate() {
        shift += desired[k] - verts[vi].cy;
    }
    shift /= layer.len() as f32;
    // Clamp: topmost node's top edge must stay ≥ origin_y.
    let top0 = &layer[0];
    let top_edge_after = verts[*top0].cy + shift - verts[*top0].h / 2.0;
    if top_edge_after < opts.origin_y {
        shift += opts.origin_y - top_edge_after;
    }
    if shift != 0.0 {
        for &vi in layer {
            verts[vi].cy += shift;
        }
    }
}

/// Turn the working graph into the public [`Layout`]: real-node positions, edge
/// polylines (attach points ordered by edge direction, port as tiebreak), and
/// per-group boxes.
fn emit(
    verts: &[Vert],
    wedges: &[WEdge],
    chain: &[WEdge],
    layers: &[Vec<usize>],
    opts: &Opts,
) -> Layout {
    // Per-layer left x = running sum of prior layers' max width + layer_gap.
    let nl = layers.len();
    let mut layer_x = vec![opts.origin_x; nl];
    let mut layer_w = vec![0.0f32; nl];
    for (l, layer) in layers.iter().enumerate() {
        layer_w[l] = layer.iter().map(|&vi| verts[vi].w).fold(0.0, f32::max);
    }
    for l in 1..nl {
        layer_x[l] = layer_x[l - 1] + layer_w[l - 1] + opts.layer_gap;
    }

    // Placed real nodes, in ascending id order for deterministic output.
    let mut placed: Vec<Placed> = verts
        .iter()
        .filter_map(|v| {
            v.id.map(|id| Placed {
                id,
                x: layer_x[v.layer as usize],
                y: v.cy - v.h / 2.0,
                w: v.w,
                h: v.h,
                layer: v.layer,
            })
        })
        .collect();
    placed.sort_by_key(|p| p.id);

    // --- Group boxes over placed real nodes, ascending group id. ------------
    let mut gmap: BTreeMap<u32, (f32, f32, f32, f32)> = BTreeMap::new(); // min_x,min_y,max_x,max_y
    for p in &placed {
        // find the group for this id
        let g = verts
            .iter()
            .find(|v| v.id == Some(p.id))
            .map(|v| v.group)
            .unwrap_or(0);
        let entry = gmap
            .entry(g)
            .or_insert((f32::INFINITY, f32::INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY));
        entry.0 = entry.0.min(p.x);
        entry.1 = entry.1.min(p.y);
        entry.2 = entry.2.max(p.x + p.w);
        entry.3 = entry.3.max(p.y + p.h);
    }
    let groups: Vec<GroupBox> = gmap
        .into_iter()
        .map(|(group, (x0, y0, x1, y1))| GroupBox {
            group,
            x: x0,
            y: y0,
            w: x1 - x0,
            h: y1 - y0,
        })
        .collect();

    // --- Edge routes. -------------------------------------------------------
    // For each original edge, gather its chain segments (in layer order), read
    // the waypoint centres from the dummies, and cap the ends with attach
    // points on the source's right side and destination's left side.
    //
    // Attach slots are ordered by the edge's *direction* — the y of its first
    // waypoint past the node (a dummy centre, else the far node's centre) —
    // with the port index only as tiebreak. Ordering by port index alone would
    // cross/overlap a fan-out right at the node boundary whenever crossing
    // reduction stacked the targets in a different order than the pads.
    let vert_x = |vi: usize| layer_x[verts[vi].layer as usize];

    // Dummy waypoints per original edge, in layer order.
    let mut dummies_by_edge: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for seg in chain {
        if verts[seg.to].id.is_none() {
            dummies_by_edge.entry(seg.orig).or_default().push(seg.to);
        }
    }
    for d in dummies_by_edge.values_mut() {
        d.sort_by_key(|&v| (verts[v].layer, verts[v].seq));
    }

    // The y an edge heads toward right after leaving its source / before
    // reaching its destination.
    let out_anchor = |ei: usize| -> f32 {
        let we = &wedges[ei];
        dummies_by_edge
            .get(&we.orig)
            .and_then(|d| d.first())
            .map(|&d| verts[d].cy)
            .unwrap_or(verts[we.to].cy)
    };
    let in_anchor = |ei: usize| -> f32 {
        let we = &wedges[ei];
        dummies_by_edge
            .get(&we.orig)
            .and_then(|d| d.last())
            .map(|&d| verts[d].cy)
            .unwrap_or(verts[we.from].cy)
    };

    // Group edges per node side, order by (anchor, port), assign evenly spaced
    // slots down the side: slot k of n at top + h*(k+1)/(n+1).
    let mut start_y: Vec<f32> = vec![0.0; wedges.len()];
    let mut end_y: Vec<f32> = vec![0.0; wedges.len()];
    let mut out_by_node: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    let mut in_by_node: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (ei, we) in wedges.iter().enumerate() {
        out_by_node.entry(we.from).or_default().push(ei);
        in_by_node.entry(we.to).or_default().push(ei);
    }
    let slot_y = |vi: usize, k: usize, n: usize| -> f32 {
        if n <= 1 {
            verts[vi].cy
        } else {
            let top = verts[vi].cy - verts[vi].h / 2.0;
            top + verts[vi].h * (k as f32 + 1.0) / (n as f32 + 1.0)
        }
    };
    for (&vi, eis) in &out_by_node {
        let mut order = eis.clone();
        order.sort_by(|&a, &b| {
            out_anchor(a)
                .partial_cmp(&out_anchor(b))
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| wedges[a].src_port.cmp(&wedges[b].src_port))
                .then_with(|| a.cmp(&b))
        });
        let n = order.len();
        for (k, &ei) in order.iter().enumerate() {
            start_y[ei] = slot_y(vi, k, n);
        }
    }
    for (&vi, eis) in &in_by_node {
        let mut order = eis.clone();
        order.sort_by(|&a, &b| {
            in_anchor(a)
                .partial_cmp(&in_anchor(b))
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| wedges[a].dst_port.cmp(&wedges[b].dst_port))
                .then_with(|| a.cmp(&b))
        });
        let n = order.len();
        for (k, &ei) in order.iter().enumerate() {
            end_y[ei] = slot_y(vi, k, n);
        }
    }

    let mut routes: Vec<EdgeRoute> = Vec::with_capacity(wedges.len());
    for (ei, we) in wedges.iter().enumerate() {
        let mut points: Vec<(f32, f32)> = Vec::new();
        // Start: attach on the right side of `we.from`.
        points.push((vert_x(we.from) + verts[we.from].w, start_y[ei]));
        // Interior: dummy waypoints (centres), in layer order.
        if let Some(dummies) = dummies_by_edge.get(&we.orig) {
            for &d in dummies {
                points.push((vert_x(d), verts[d].cy));
            }
        }
        // End: attach on the left side of `we.to`.
        points.push((vert_x(we.to), end_y[ei]));
        routes.push(EdgeRoute {
            src: verts[we.from].id.unwrap_or(0),
            src_port: we.src_port,
            dst: verts[we.to].id.unwrap_or(0),
            dst_port: we.dst_port,
            points,
        });
    }
    // Deterministic route order: by (src, src_port, dst, dst_port).
    routes.sort_by(|a, b| {
        (a.src, a.src_port, a.dst, a.dst_port).cmp(&(b.src, b.src_port, b.dst, b.dst_port))
    });

    Layout {
        nodes: placed,
        edges: routes,
        groups,
    }
}

// ===========================================================================
// Tests
// ===========================================================================
//
// Unit tests live here (inside a `#[cfg(test)]` module) rather than in
// `scope/tests/` to avoid colliding with the parallel UI agent's test file.
#[cfg(test)]
mod tests {
    use super::*;

    fn n(id: u32, group: u32) -> Node {
        Node {
            id,
            w: 100.0,
            h: 40.0,
            group,
        }
    }
    fn e(src: u32, dst: u32) -> Edge {
        Edge {
            src,
            src_port: 0,
            dst,
            dst_port: 0,
        }
    }
    fn ep(src: u32, sp: u16, dst: u32, dp: u16) -> Edge {
        Edge {
            src,
            src_port: sp,
            dst,
            dst_port: dp,
        }
    }

    /// Fetch a placed node by id.
    fn get(l: &Layout, id: u32) -> &Placed {
        l.nodes.iter().find(|p| p.id == id).expect("node placed")
    }
    /// The layer (column) each node landed in, keyed by id.
    fn layer_of(l: &Layout, id: u32) -> u32 {
        get(l, id).layer
    }

    // --- (1) Golden coordinates for pinned small graphs. --------------------

    #[test]
    fn golden_chain() {
        // 0 → 1 → 2 : three layers, single column each.
        let nodes = [n(0, 0), n(1, 0), n(2, 0)];
        let edges = [e(0, 1), e(1, 2)];
        let opts = Opts::default();
        let l = layout(&nodes, &edges, &opts);
        assert_eq!(layer_of(&l, 0), 0);
        assert_eq!(layer_of(&l, 1), 1);
        assert_eq!(layer_of(&l, 2), 2);
        // x columns: 0, 100+60=160, 320.
        assert_eq!(get(&l, 0).x, 0.0);
        assert_eq!(get(&l, 1).x, 160.0);
        assert_eq!(get(&l, 2).x, 320.0);
        // Single node per layer ⇒ all aligned to same y (origin, centred).
        assert_eq!(get(&l, 0).y, get(&l, 1).y);
        assert_eq!(get(&l, 1).y, get(&l, 2).y);
        assert_eq!(get(&l, 0).y, 0.0);
        // One straight edge each, 2 points.
        assert_eq!(l.edges.len(), 2);
        for r in &l.edges {
            assert_eq!(r.points.len(), 2);
        }
        // First edge starts on node 0's right side (x=100) and ends on node 1's
        // left side (x=160).
        let r01 = l.edges.iter().find(|r| r.src == 0 && r.dst == 1).unwrap();
        assert_eq!(r01.points[0].0, 100.0);
        assert_eq!(r01.points[1].0, 160.0);
    }

    #[test]
    fn golden_diamond() {
        // 0 → 1, 0 → 2, 1 → 3, 2 → 3. Layers: 0|{1,2}|3.
        let nodes = [n(0, 0), n(1, 0), n(2, 0), n(3, 0)];
        let edges = [e(0, 1), e(0, 2), e(1, 3), e(2, 3)];
        let opts = Opts::default();
        let l = layout(&nodes, &edges, &opts);
        assert_eq!(layer_of(&l, 0), 0);
        assert_eq!(layer_of(&l, 1), 1);
        assert_eq!(layer_of(&l, 2), 1);
        assert_eq!(layer_of(&l, 3), 2);
        // The middle layer has two nodes; they must not overlap and be node_gap
        // apart (h=40, gap=20 ⇒ centres 60 apart).
        let y1 = get(&l, 1).y;
        let y2 = get(&l, 2).y;
        assert!((y2 - y1).abs() >= 40.0 + 20.0 - 0.01);
        // Source and sink centred between the two middles (symmetry).
        let c1 = y1 + 20.0;
        let c2 = y2 + 20.0;
        let mid = (c1 + c2) / 2.0;
        let c0 = get(&l, 0).y + 20.0;
        let c3 = get(&l, 3).y + 20.0;
        assert!((c0 - mid).abs() < 0.5, "source centred: {c0} vs {mid}");
        assert!((c3 - mid).abs() < 0.5, "sink centred: {c3} vs {mid}");
    }

    #[test]
    fn golden_tee_fanout() {
        // One source → three sinks (a tee). All sinks in layer 1, stacked.
        let nodes = [n(0, 0), n(1, 0), n(2, 0), n(3, 0)];
        let edges = [e(0, 1), e(0, 2), e(0, 3)];
        let l = layout(&nodes, &edges, &Opts::default());
        assert_eq!(layer_of(&l, 0), 0);
        for s in [1, 2, 3] {
            assert_eq!(layer_of(&l, s), 1);
        }
        // Three stacked sinks, strictly increasing, non-overlapping.
        let mut ys: Vec<f32> = [1, 2, 3].iter().map(|&s| get(&l, s).y).collect();
        ys.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for w in ys.windows(2) {
            assert!(w[1] - w[0] >= 40.0 + 20.0 - 0.01);
        }
        // Source centred on the middle sink.
        let c0 = get(&l, 0).y + 20.0;
        let mid = ys[1] + 20.0;
        assert!((c0 - mid).abs() < 0.5);
        assert_eq!(l.edges.len(), 3);
    }

    #[test]
    fn golden_muxer_fanin() {
        // Three sources → one muxer. Sources stacked in layer 0.
        let nodes = [n(0, 0), n(1, 0), n(2, 0), n(3, 0)];
        let edges = [e(0, 3), e(1, 3), e(2, 3)];
        let l = layout(&nodes, &edges, &Opts::default());
        for s in [0, 1, 2] {
            assert_eq!(layer_of(&l, s), 0);
        }
        assert_eq!(layer_of(&l, 3), 1);
        // Muxer centred on the middle source.
        let mut ys: Vec<f32> = [0, 1, 2].iter().map(|&s| get(&l, s).y).collect();
        ys.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let c3 = get(&l, 3).y + 20.0;
        let mid = ys[1] + 20.0;
        assert!((c3 - mid).abs() < 0.5);
    }

    #[test]
    fn golden_multilayer_span_dummy() {
        // 0 → 1 → 2 and a long edge 0 → 2 spanning 2 layers ⇒ one dummy in
        // layer 1. The long edge's route must have 3 points (src, dummy, dst).
        let nodes = [n(0, 0), n(1, 0), n(2, 0)];
        let edges = [e(0, 1), e(1, 2), e(0, 2)];
        let l = layout(&nodes, &edges, &Opts::default());
        assert_eq!(layer_of(&l, 0), 0);
        assert_eq!(layer_of(&l, 1), 1);
        assert_eq!(layer_of(&l, 2), 2);
        let long = l.edges.iter().find(|r| r.src == 0 && r.dst == 2).unwrap();
        assert_eq!(long.points.len(), 3, "long edge routed through a dummy");
        // The dummy waypoint sits in layer 1's x column (=160).
        assert_eq!(long.points[1].0, 160.0);
        // Waypoint x strictly between source-right and dst-left.
        assert!(long.points[0].0 < long.points[1].0);
        assert!(long.points[1].0 < long.points[2].0);
    }

    // --- (2) Invariants over generated graphs. ------------------------------

    /// A small deterministic PRNG (xorshift) so generated graphs are stable.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
        fn below(&mut self, n: u32) -> u32 {
            (self.next() % n as u64) as u32
        }
    }

    /// Generate a random DAG-ish graph (edges may form cycles; that's tested).
    fn gen_graph(seed: u64, nnodes: u32, nedges: u32, ngroups: u32) -> (Vec<Node>, Vec<Edge>) {
        let mut rng = Rng(seed | 1);
        let nodes: Vec<Node> = (0..nnodes)
            .map(|i| Node {
                id: i,
                w: 60.0 + (i % 4) as f32 * 20.0,
                h: 30.0 + (i % 3) as f32 * 15.0,
                group: rng.below(ngroups),
            })
            .collect();
        let mut edges = Vec::new();
        for _ in 0..nedges {
            let a = rng.below(nnodes);
            let b = rng.below(nnodes);
            if a != b {
                edges.push(ep(a, rng.below(3) as u16, b, rng.below(3) as u16));
            }
        }
        (nodes, edges)
    }

    #[test]
    fn invariant_edges_go_left_to_right() {
        let opts = Opts::default();
        for seed in 1..40u64 {
            let (nodes, edges) = gen_graph(seed, 12, 20, 3);
            let l = layout(&nodes, &edges, &opts);
            // After cycle-breaking, every *drawn* edge's route x is monotone
            // non-decreasing and the endpoints strictly advance (src layer <
            // dst layer OR equal-layer only when the edge was a back-edge that
            // got dropped — but dropped edges do not appear in routes at all).
            for r in &l.edges {
                let sl = layer_of(&l, r.src);
                let dl = layer_of(&l, r.dst);
                assert!(dl > sl, "edge {}->{} not left→right (layers {sl}->{dl})", r.src, r.dst);
                // Polyline x strictly increases across the span.
                for w in r.points.windows(2) {
                    assert!(w[1].0 >= w[0].0 - 0.001, "route x went backwards");
                }
                assert!(r.points.first().unwrap().0 <= r.points.last().unwrap().0);
            }
        }
    }

    #[test]
    fn invariant_no_overlap_within_layer() {
        let opts = Opts::default();
        for seed in 1..40u64 {
            let (nodes, edges) = gen_graph(seed, 14, 24, 4);
            let l = layout(&nodes, &edges, &opts);
            // Group placed nodes by layer, check vertical separation.
            let mut by_layer: BTreeMap<u32, Vec<&Placed>> = BTreeMap::new();
            for p in &l.nodes {
                by_layer.entry(p.layer).or_default().push(p);
            }
            for (_, mut ps) in by_layer {
                ps.sort_by(|a, b| a.y.partial_cmp(&b.y).unwrap());
                for w in ps.windows(2) {
                    let a = w[0];
                    let b = w[1];
                    // b.top must be ≥ a.bottom + node_gap (minus float slack).
                    assert!(
                        b.y >= a.y + a.h + opts.node_gap - 0.01,
                        "overlap in layer {}: {} (y {}..{}) vs {} (y {})",
                        a.layer,
                        a.id,
                        a.y,
                        a.y + a.h,
                        b.id,
                        b.y
                    );
                }
            }
        }
    }

    #[test]
    fn invariant_determinism() {
        let opts = Opts::default();
        for seed in 1..30u64 {
            let (nodes, edges) = gen_graph(seed, 15, 26, 3);
            let a = layout(&nodes, &edges, &opts);
            let b = layout(&nodes, &edges, &opts);
            assert_eq!(a, b, "layout not deterministic for seed {seed}");
        }
    }

    #[test]
    fn invariant_cycles_terminate() {
        // Pure cycle 0→1→2→0 plus extras. Must not loop/panic and must place
        // all nodes with a valid (acyclic) layering of the retained edges.
        let nodes = [n(0, 0), n(1, 0), n(2, 0)];
        let edges = [e(0, 1), e(1, 2), e(2, 0)];
        let l = layout(&nodes, &edges, &Opts::default());
        assert_eq!(l.nodes.len(), 3);
        // Every retained route still goes strictly left→right.
        for r in &l.edges {
            assert!(layer_of(&l, r.dst) > layer_of(&l, r.src));
        }
        // A larger random-with-cycles batch: just assert it returns.
        for seed in 1..25u64 {
            let (nodes, edges) = gen_graph(seed, 20, 60, 4); // dense ⇒ cycles likely
            let l = layout(&nodes, &edges, &Opts::default());
            assert_eq!(l.nodes.len(), 20);
            for r in &l.edges {
                assert!(layer_of(&l, r.dst) > layer_of(&l, r.src));
            }
        }
    }

    // --- (3) Group contiguity on a mixed-group example. ---------------------

    #[test]
    fn group_contiguity() {
        // A wide fan-out where sinks belong to two groups; within the sink
        // layer, same-group nodes should be contiguous (no interleaving) once
        // ordering settles, since no crossing pressure separates them.
        // source (grp 0) → six sinks: ids 1,3,5 in grp 1; ids 2,4,6 in grp 2.
        let nodes = [
            n(0, 0),
            n(1, 1),
            n(2, 2),
            n(3, 1),
            n(4, 2),
            n(5, 1),
            n(6, 2),
        ];
        let edges = [e(0, 1), e(0, 2), e(0, 3), e(0, 4), e(0, 5), e(0, 6)];
        let l = layout(&nodes, &edges, &Opts::default());
        // Order sink nodes by y; the group sequence must be a block of 1s then
        // a block of 2s (or vice-versa) — never interleaved.
        let mut sinks: Vec<&Placed> = l.nodes.iter().filter(|p| p.id != 0).collect();
        sinks.sort_by(|a, b| a.y.partial_cmp(&b.y).unwrap());
        let group_of = |id: u32| nodes.iter().find(|n| n.id == id).unwrap().group;
        let seq: Vec<u32> = sinks.iter().map(|p| group_of(p.id)).collect();
        // Count group changes along the ordered sequence: contiguous ⇒ exactly 1.
        let changes = seq.windows(2).filter(|w| w[0] != w[1]).count();
        assert_eq!(changes, 1, "groups not contiguous, sequence = {seq:?}");
    }

    // --- (4) Port-ordered attach points. ------------------------------------

    #[test]
    fn port_ordered_attach_no_cross() {
        // Two sources into a node with two input ports. The edge into dst_port 0
        // must attach ABOVE (smaller y) the edge into dst_port 1, regardless of
        // source order, so the two edges don't cross at the node boundary.
        let nodes = [n(0, 0), n(1, 0), n(2, 0)];
        // node 2 is the 2-pad sink; port 0 fed by node 0, port 1 fed by node 1.
        let edges = [ep(0, 0, 2, 0), ep(1, 0, 2, 1)];
        let l = layout(&nodes, &edges, &Opts::default());
        let r_to0 = l.edges.iter().find(|r| r.dst_port == 0).unwrap();
        let r_to1 = l.edges.iter().find(|r| r.dst_port == 1).unwrap();
        // Attach y at the destination (last point).
        let y_port0 = r_to0.points.last().unwrap().1;
        let y_port1 = r_to1.points.last().unwrap().1;
        assert!(
            y_port0 < y_port1,
            "port 0 attach ({y_port0}) should be above port 1 ({y_port1})"
        );
        // Both attach on the left side of node 2 (same x).
        assert_eq!(r_to0.points.last().unwrap().0, r_to1.points.last().unwrap().0);
    }

    #[test]
    fn port_ordered_out_attach() {
        // Symmetric: one source with two output ports feeding two sinks. Port 0
        // out attaches above port 1 out on the source's right side.
        let nodes = [n(0, 0), n(1, 0), n(2, 0)];
        let edges = [ep(0, 0, 1, 0), ep(0, 1, 2, 0)];
        let l = layout(&nodes, &edges, &Opts::default());
        let r_p0 = l.edges.iter().find(|r| r.src_port == 0).unwrap();
        let r_p1 = l.edges.iter().find(|r| r.src_port == 1).unwrap();
        let y0 = r_p0.points.first().unwrap().1;
        let y1 = r_p1.points.first().unwrap().1;
        assert!(y0 < y1, "src port 0 ({y0}) should be above src port 1 ({y1})");
    }

    // --- Robustness edge cases. ---------------------------------------------

    #[test]
    fn empty_input() {
        let l = layout(&[], &[], &Opts::default());
        assert!(l.nodes.is_empty() && l.edges.is_empty() && l.groups.is_empty());
    }

    #[test]
    fn unknown_endpoints_skipped() {
        // Edge references node 99 which does not exist ⇒ skipped, no panic.
        let nodes = [n(0, 0), n(1, 0)];
        let edges = [e(0, 1), e(0, 99), e(99, 1)];
        let l = layout(&nodes, &edges, &Opts::default());
        assert_eq!(l.nodes.len(), 2);
        assert_eq!(l.edges.len(), 1); // only 0→1 survives
    }

    #[test]
    fn duplicate_ids_first_wins() {
        let a = Node {
            id: 5,
            w: 10.0,
            h: 10.0,
            group: 0,
        };
        let b = Node {
            id: 5,
            w: 999.0,
            h: 999.0,
            group: 9,
        };
        let l = layout(&[a, b], &[], &Opts::default());
        assert_eq!(l.nodes.len(), 1);
        assert_eq!(l.nodes[0].w, 10.0); // first definition wins
    }

    #[test]
    fn self_loop_dropped() {
        let nodes = [n(0, 0)];
        let edges = [e(0, 0)];
        let l = layout(&nodes, &edges, &Opts::default());
        assert_eq!(l.nodes.len(), 1);
        assert_eq!(l.edges.len(), 0);
    }

    #[test]
    fn group_boxes_present() {
        let nodes = [n(0, 0), n(1, 1), n(2, 1)];
        let edges = [e(0, 1), e(0, 2)];
        let l = layout(&nodes, &edges, &Opts::default());
        // Two groups ⇒ two boxes, ascending group id, each covering its nodes.
        assert_eq!(l.groups.len(), 2);
        assert_eq!(l.groups[0].group, 0);
        assert_eq!(l.groups[1].group, 1);
        // Group 1's box must contain both node 1 and node 2.
        let g1 = &l.groups[1];
        for id in [1u32, 2] {
            let p = get(&l, id);
            assert!(p.x >= g1.x - 0.01 && p.x + p.w <= g1.x + g1.w + 0.01);
            assert!(p.y >= g1.y - 0.01 && p.y + p.h <= g1.y + g1.h + 0.01);
        }
    }

    #[test]
    fn fan_out_attach_follows_target_order() {
        // One source with three out-edges whose PORT order (0,1,2) deliberately
        // disagrees with where the targets end up vertically. The attach points
        // on the source's right side must follow the edges' *direction* (target
        // y order), not the port order — otherwise the fan-out overlaps/crosses
        // right at the node boundary.
        let nodes = vec![
            Node { id: 0, w: 80.0, h: 40.0, group: 0 },
            Node { id: 1, w: 80.0, h: 40.0, group: 0 },
            Node { id: 2, w: 80.0, h: 40.0, group: 0 },
            Node { id: 3, w: 80.0, h: 40.0, group: 0 },
        ];
        let edges = vec![
            Edge { src: 0, src_port: 0, dst: 1, dst_port: 0 },
            Edge { src: 0, src_port: 1, dst: 2, dst_port: 0 },
            Edge { src: 0, src_port: 2, dst: 3, dst_port: 0 },
        ];
        let l = layout(&nodes, &edges, &Opts::default());
        // Each route starts at a distinct y (no overlapping attach points).
        let mut starts: Vec<(u32, f32)> =
            l.edges.iter().map(|e| (e.dst, e.points[0].1)).collect();
        for i in 0..starts.len() {
            for j in (i + 1)..starts.len() {
                assert!(
                    (starts[i].1 - starts[j].1).abs() > 1.0,
                    "attach points must not overlap: {starts:?}"
                );
            }
        }
        // Attach order matches the targets' vertical order: sort both ways and
        // compare pairings.
        let y_of = |id: u32| l.nodes.iter().find(|n| n.id == id).unwrap().y;
        starts.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let mut by_target: Vec<u32> = starts.iter().map(|(d, _)| *d).collect();
        let mut targets_by_y: Vec<u32> = vec![1, 2, 3];
        targets_by_y.sort_by(|&a, &b| y_of(a).partial_cmp(&y_of(b)).unwrap());
        by_target.dedup();
        assert_eq!(by_target, targets_by_y, "attach order must follow target order");
    }

    #[test]
    fn different_groups_keep_hull_clearance() {
        // Two two-node chains in different groups share layers; vertically
        // adjacent nodes from different groups must be separated by node_gap +
        // group_gap so the drawn hulls (padding + label strip) cannot overlap.
        let nodes = vec![
            Node { id: 0, w: 80.0, h: 40.0, group: 0 },
            Node { id: 1, w: 80.0, h: 40.0, group: 0 },
            Node { id: 2, w: 80.0, h: 40.0, group: 1 },
            Node { id: 3, w: 80.0, h: 40.0, group: 1 },
        ];
        let edges = vec![
            Edge { src: 0, src_port: 0, dst: 1, dst_port: 0 },
            Edge { src: 2, src_port: 0, dst: 3, dst_port: 0 },
        ];
        let opts = Opts::default();
        let l = layout(&nodes, &edges, &opts);
        let g0 = l.groups.iter().find(|g| g.group == 0).unwrap();
        let g1 = l.groups.iter().find(|g| g.group == 1).unwrap();
        let (top, bottom) = if g0.y < g1.y { (g0, g1) } else { (g1, g0) };
        let clearance = bottom.y - (top.y + top.h);
        assert!(
            clearance >= opts.node_gap + opts.group_gap - 0.5,
            "group boxes must be separated by node_gap+group_gap, got {clearance}"
        );
    }

}
