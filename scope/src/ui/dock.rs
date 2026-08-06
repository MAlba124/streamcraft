//! A dockable pane layout: a binary tree of split regions whose leaves are tab
//! groups of panels — the model simprof's dockspace uses (VS Code-style), adapted
//! to this immediate-mode UI. The tree lays out into flat geometry (content
//! rects, tab bars, dividers, drop regions); an interaction driver
//! ([`dock_begin`] / [`dock_finish`]) turns mouse input into tab activation,
//! divider resizing, and drag-a-tab-to-dock moves with drop-zone overlays.
//!
//! The tree itself is pure data (tested without a window): node indices are
//! stable across mutations (parents are re-pointed, slots reused), so geometry
//! captured before a mutation stays meaningful through it.

use crate::ui::draw::Rect;
use crate::ui::widgets::Ui;
use crate::ui::MouseButton;

/// Tab bar height (px).
pub const TAB_H: f32 = 24.0;
/// Divider thickness (px).
pub const DIVIDER: f32 = 6.0;
/// Drag distance (px, manhattan) before a pressed tab becomes a drag.
const DRAG_THRESHOLD: f32 = 6.0;

/// Where a dragged panel lands relative to a target leaf region: join its tab
/// group, or split it on one side.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DropZone {
    Center,
    Left,
    Right,
    Top,
    Bottom,
}

enum Node<P> {
    Split {
        /// true = children side by side (a vertical divider bar).
        horizontal: bool,
        ratio: f32,
        a: usize,
        b: usize,
    },
    Leaf {
        panels: Vec<P>,
        active: usize,
    },
}

/// The dock tree. `P` is the caller's panel id (small `Copy` enum).
pub struct DockTree<P> {
    nodes: Vec<Node<P>>,
    root: usize,
}

/// One visible panel's content rect (below its tab bar).
#[derive(Clone, Copy, Debug)]
pub struct PanelGeom<P> {
    pub id: P,
    pub rect: Rect,
}

/// One tab in a leaf's tab bar.
#[derive(Clone, Debug)]
pub struct TabSlot<P> {
    pub panel: P,
    pub label: String,
    pub active: bool,
    pub leaf: usize,
    pub rect: Rect,
}

/// A divider bar between two split children, plus the split's full region so a
/// drag recomputes the ratio in place.
#[derive(Clone, Copy, Debug)]
pub struct DividerSlot {
    pub split: usize,
    pub vertical: bool,
    pub rect: Rect,
    pub region: Rect,
}

/// A leaf's full rectangle (tab bar + content) for drop-zone hit testing.
#[derive(Clone, Copy, Debug)]
pub struct RegionSlot {
    pub leaf: usize,
    pub rect: Rect,
}

/// The flattened layout of the whole tree for one frame.
pub struct DockLayout<P> {
    pub panels: Vec<PanelGeom<P>>,
    pub tabs: Vec<TabSlot<P>>,
    pub dividers: Vec<DividerSlot>,
    pub regions: Vec<RegionSlot>,
}

fn push<P>(nodes: &mut Vec<Node<P>>, n: Node<P>) -> usize {
    nodes.push(n);
    nodes.len() - 1
}

impl<P: Copy + PartialEq> DockTree<P> {
    /// A single leaf holding `panels` (first is active).
    // Cold: the dock tree is built once at startup, not per frame (clippy.toml).
    #[allow(clippy::disallowed_methods)]
    pub fn single(panels: Vec<P>) -> DockTree<P> {
        let mut nodes = Vec::new();
        let root = push(&mut nodes, Node::Leaf { panels, active: 0 });
        DockTree { nodes, root }
    }

    /// A leaf per panel: `a | b` side by side above `bottom` (the scope default —
    /// graph | elements over the log).
    // Cold: the dock tree is built once at startup, not per frame (clippy.toml).
    #[allow(clippy::disallowed_methods)]
    pub fn two_over_one(a: P, b: P, bottom: P, x_ratio: f32, y_ratio: f32) -> DockTree<P> {
        let mut nodes = Vec::new();
        let la = push(&mut nodes, Node::Leaf { panels: vec![a], active: 0 });
        let lb = push(&mut nodes, Node::Leaf { panels: vec![b], active: 0 });
        let lc = push(&mut nodes, Node::Leaf { panels: vec![bottom], active: 0 });
        let top = push(&mut nodes, Node::Split { horizontal: true, ratio: x_ratio, a: la, b: lb });
        let root = push(&mut nodes, Node::Split { horizontal: false, ratio: y_ratio, a: top, b: lc });
        DockTree { nodes, root }
    }

    /// Every panel docked in the *reachable* tree. Walked from the root, not a
    /// flat node scan: collapses leave orphaned slots in `nodes` by design
    /// (index stability), and orphans must not count.
    // Cold: a query helper (not on the per-frame draw path; only tests call it here).
    #[allow(clippy::disallowed_methods)]
    pub fn all_panels(&self) -> Vec<P> {
        let mut ids = Vec::new();
        let mut stack = vec![self.root];
        while let Some(i) = stack.pop() {
            match &self.nodes[i] {
                Node::Split { a, b, .. } => {
                    stack.push(*a);
                    stack.push(*b);
                }
                Node::Leaf { panels, .. } => ids.extend_from_slice(panels),
            }
        }
        ids
    }

    /// Lay the tree out inside `area`. `label` supplies tab text; `tab_w` its
    /// width (the caller measures with its font).
    pub fn layout(
        &self,
        area: Rect,
        label: &dyn Fn(P) -> String,
        tab_w: &dyn Fn(&str) -> f32,
    ) -> DockLayout<P> {
        let mut out =
            DockLayout { panels: Vec::new(), tabs: Vec::new(), dividers: Vec::new(), regions: Vec::new() };
        self.layout_node(self.root, area, label, tab_w, &mut out);
        out
    }

    fn layout_node(
        &self,
        idx: usize,
        r: Rect,
        label: &dyn Fn(P) -> String,
        tab_w: &dyn Fn(&str) -> f32,
        out: &mut DockLayout<P>,
    ) {
        match &self.nodes[idx] {
            Node::Split { horizontal, ratio, a, b } => {
                if *horizontal {
                    let aw = ((r.w - DIVIDER) * ratio).max(0.0);
                    let bw = (r.w - DIVIDER - aw).max(0.0);
                    self.layout_node(*a, Rect::new(r.x, r.y, aw, r.h), label, tab_w, out);
                    out.dividers.push(DividerSlot {
                        split: idx,
                        vertical: true,
                        rect: Rect::new(r.x + aw, r.y, DIVIDER, r.h),
                        region: r,
                    });
                    self.layout_node(*b, Rect::new(r.x + aw + DIVIDER, r.y, bw, r.h), label, tab_w, out);
                } else {
                    let ah = ((r.h - DIVIDER) * ratio).max(0.0);
                    let bh = (r.h - DIVIDER - ah).max(0.0);
                    self.layout_node(*a, Rect::new(r.x, r.y, r.w, ah), label, tab_w, out);
                    out.dividers.push(DividerSlot {
                        split: idx,
                        vertical: false,
                        rect: Rect::new(r.x, r.y + ah, r.w, DIVIDER),
                        region: r,
                    });
                    self.layout_node(*b, Rect::new(r.x, r.y + ah + DIVIDER, r.w, bh), label, tab_w, out);
                }
            }
            Node::Leaf { panels, active } => {
                let mut tx = r.x;
                for (i, p) in panels.iter().enumerate() {
                    let text = label(*p);
                    let w = tab_w(&text);
                    out.tabs.push(TabSlot {
                        panel: *p,
                        label: text,
                        active: i == *active,
                        leaf: idx,
                        rect: Rect::new(tx, r.y, w, TAB_H),
                    });
                    tx += w;
                }
                if let Some(p) = panels.get(*active) {
                    out.panels.push(PanelGeom {
                        id: *p,
                        rect: Rect::new(r.x, r.y + TAB_H, r.w, (r.h - TAB_H).max(0.0)),
                    });
                }
                out.regions.push(RegionSlot { leaf: idx, rect: r });
            }
        }
    }

    /// Make `panel` the active tab of whichever leaf holds it.
    pub fn activate(&mut self, panel: P) {
        for n in &mut self.nodes {
            if let Node::Leaf { panels, active } = n {
                if let Some(i) = panels.iter().position(|p| *p == panel) {
                    *active = i;
                }
            }
        }
    }

    /// Recompute a split's ratio from an absolute pointer position over `div`.
    pub fn set_ratio_from_pointer(&mut self, div: &DividerSlot, x: f32, y: f32) {
        let r = if div.vertical {
            (x - div.region.x) / div.region.w.max(1.0)
        } else {
            (y - div.region.y) / div.region.h.max(1.0)
        };
        let r = r.clamp(0.1, 0.9);
        if let Node::Split { ratio, .. } = &mut self.nodes[div.split] {
            *ratio = r;
        }
    }

    /// Move `panel` out of its current leaf and dock it at `target_leaf`:
    /// `Center` joins the tab group; a side splits the leaf with the panel on
    /// that side. Node indices stay stable through the move.
    pub fn move_panel(&mut self, panel: P, target_leaf: usize, zone: DropZone) {
        // Dropping a leaf's only panel back onto itself is a no-op.
        if let Node::Leaf { panels, .. } = &self.nodes[target_leaf] {
            if panels.len() == 1 && panels[0] == panel {
                return;
            }
        }
        self.remove_panel(panel);
        match zone {
            DropZone::Center => {
                if let Node::Leaf { panels, active } = &mut self.nodes[target_leaf] {
                    panels.push(panel);
                    *active = panels.len() - 1;
                }
            }
            _ => {
                let horizontal = matches!(zone, DropZone::Left | DropZone::Right);
                let new_leaf = push(&mut self.nodes, Node::Leaf { panels: vec![panel], active: 0 });
                let moved = self.move_node_to_new_slot(target_leaf);
                let (a, b) = match zone {
                    DropZone::Left | DropZone::Top => (new_leaf, moved),
                    _ => (moved, new_leaf),
                };
                self.nodes[target_leaf] = Node::Split { horizontal, ratio: 0.5, a, b };
            }
        }
    }

    /// Remove `panel` from its leaf; an emptied leaf collapses its parent split.
    pub fn remove_panel(&mut self, panel: P) {
        let mut emptied = None;
        for (i, n) in self.nodes.iter_mut().enumerate() {
            if let Node::Leaf { panels, active } = n {
                if let Some(pos) = panels.iter().position(|p| *p == panel) {
                    panels.remove(pos);
                    if *active >= panels.len() {
                        *active = panels.len().saturating_sub(1);
                    }
                    if panels.is_empty() {
                        emptied = Some(i);
                    }
                    break;
                }
            }
        }
        if let Some(empty) = emptied {
            self.collapse_leaf(empty);
        }
    }

    /// Replace the split that has `empty_leaf` as a child with its sibling child,
    /// by **repointing the grandparent (or the root) at the sibling** — never by
    /// moving nodes between slots. This keeps every reachable node's index valid
    /// through the collapse (a drop target captured before a `move_panel` may be
    /// the sibling itself); the parent split and the empty leaf become orphans in
    /// `nodes`, which is why parents are found by walking the *reachable* tree —
    /// a flat scan could match a dead orphan still recording `empty_leaf` as a
    /// child. (The same invariant simprof's dockspace documents.)
    // Cold: runs only when a drag-drop empties a leaf (a rare structural mutation),
    // not per frame (clippy.toml allocation ban).
    #[allow(clippy::disallowed_methods)]
    fn collapse_leaf(&mut self, empty_leaf: usize) {
        // Reachable-parent map: node index → its parent split index.
        let mut parent_of: Vec<(usize, usize)> = Vec::new();
        let mut stack = vec![self.root];
        while let Some(i) = stack.pop() {
            if let Node::Split { a, b, .. } = self.nodes[i] {
                parent_of.push((a, i));
                parent_of.push((b, i));
                stack.push(a);
                stack.push(b);
            }
        }
        let find = |idx: usize| parent_of.iter().find(|(c, _)| *c == idx).map(|(_, p)| *p);
        let Some(parent) = find(empty_leaf) else {
            return; // the empty leaf is the root (or already detached): leave as-is
        };
        let sibling = match &self.nodes[parent] {
            Node::Split { a, b, .. } => {
                if *a == empty_leaf {
                    *b
                } else {
                    *a
                }
            }
            _ => return,
        };
        match find(parent) {
            None => self.root = sibling,
            Some(grand) => {
                if let Node::Split { a, b, .. } = &mut self.nodes[grand] {
                    if *a == parent {
                        *a = sibling;
                    }
                    if *b == parent {
                        *b = sibling;
                    }
                }
            }
        }
    }

    /// Move the node at `idx` into a fresh slot, returning the new index; the old
    /// slot is left for the caller to overwrite.
    // Cold: runs only on a drag-drop panel move, not per frame; `vec![]` is an empty
    // placeholder that allocates nothing (clippy.toml allocation ban).
    #[allow(clippy::disallowed_methods)]
    fn move_node_to_new_slot(&mut self, idx: usize) -> usize {
        let taken = std::mem::replace(&mut self.nodes[idx], Node::Leaf { panels: vec![], active: 0 });
        push(&mut self.nodes, taken)
    }
}

// ---------------------------------------------------------------------------
// Immediate-mode interaction driver
// ---------------------------------------------------------------------------

/// Retained interaction state for the dock (one per window).
#[derive(Default)]
pub struct DockState {
    /// A pressed tab: `(leaf-at-press, press position)`, promoted to a drag past
    /// the threshold.
    press: Option<(f32, f32)>,
    pressed_panel_index: Option<usize>, // index into this frame's `tabs`
    /// The panel being dragged (by tab), once past the threshold.
    dragging_label: Option<String>,
    dragging_index: Option<usize>,
    /// Divider currently being dragged.
    div_drag: Option<DividerSlot>,
}

impl DockState {
    /// True while a tab drag or divider drag is in flight (panels should ignore
    /// the mouse).
    pub fn interacting(&self) -> bool {
        self.dragging_index.is_some() || self.div_drag.is_some()
    }
}

/// Per-frame result of [`dock_begin`]: where to draw each visible panel, plus
/// the hit-test data [`dock_finish`] needs.
pub struct DockFrame<P> {
    pub layout: DockLayout<P>,
    /// The drop target under the cursor while dragging: `(leaf, zone, preview)`.
    pub drop: Option<(usize, DropZone, Rect)>,
}

/// Handle divider/tab interaction and draw the dock chrome (tab bars, dividers).
/// Returns the visible panels' content rects; draw their contents, then call
/// [`dock_finish`] so the drag ghost and drop overlay paint on top.
pub fn dock_begin<P: Copy + PartialEq>(
    ui: &mut Ui<'_>,
    tree: &mut DockTree<P>,
    state: &mut DockState,
    area: Rect,
    label: &dyn Fn(P) -> String,
) -> DockFrame<P> {
    let t = *ui.theme();
    let font_w = {
        let f = ui.font();
        let per = f.cell_w();
        move |s: &str| s.chars().count() as f32 * per + 20.0
    };
    let layout = tree.layout(area, label, &font_w);
    let (mx, my) = (ui.input().mouse_x, ui.input().mouse_y);
    let pressed = ui.input().pressed(MouseButton::Left);
    let down = ui.input().down(MouseButton::Left);

    // --- dividers: hover + drag resize ---
    if state.div_drag.is_none() && state.dragging_index.is_none() && pressed {
        if let Some(d) = layout.dividers.iter().find(|d| d.rect.contains(mx, my)) {
            state.div_drag = Some(*d);
        }
    }
    if !down {
        state.div_drag = None;
    }
    if let Some(d) = state.div_drag {
        tree.set_ratio_from_pointer(&d, mx, my);
    }

    // --- tabs: click to activate, drag past threshold to move ---
    if pressed && state.div_drag.is_none() {
        if let Some((i, _)) = layout.tabs.iter().enumerate().find(|(_, s)| s.rect.contains(mx, my)) {
            state.press = Some((mx, my));
            state.pressed_panel_index = Some(i);
        }
    }
    if down {
        if let (Some((px, py)), Some(i), None) =
            (state.press, state.pressed_panel_index, state.dragging_index.as_ref())
        {
            if (mx - px).abs() + (my - py).abs() > DRAG_THRESHOLD {
                state.dragging_index = Some(i);
                state.dragging_label = layout.tabs.get(i).map(|s| s.label.clone());
            }
        }
    }

    // Drop target while dragging: which region is hovered, and which zone of it.
    let drop = state.dragging_index.and_then(|_| {
        layout.regions.iter().find(|r| r.rect.contains(mx, my)).map(|r| {
            let (zone, preview) = drop_zone_at(r.rect, mx, my);
            (r.leaf, zone, preview)
        })
    });

    // --- draw chrome: per-leaf tab bar strip, tabs, dividers ---
    // The strip is darker than the tabs and each tab is drawn 1px narrower than
    // its slot, so the strip shows through as a divider between adjacent tabs.
    for r in &layout.regions {
        ui.draw_list_mut()
            .fill_rect(Rect::new(r.rect.x, r.rect.y, r.rect.w, TAB_H), t.bg);
    }
    for (i, tab) in layout.tabs.iter().enumerate() {
        let hot = tab.rect.contains(mx, my);
        let bg = if tab.active {
            t.panel_bg
        } else if hot {
            t.button_hot
        } else {
            t.title_bg
        };
        let face = Rect::new(tab.rect.x, tab.rect.y, tab.rect.w - 1.0, tab.rect.h);
        ui.draw_list_mut().fill_rect(face, bg);
        if tab.active {
            ui.draw_list_mut()
                .fill_rect(Rect::new(face.x, face.y, face.w, 2.0), t.accent);
        }
        let color = if tab.active { t.text } else { t.text_dim };
        let ty = tab.rect.y + (TAB_H - ui.font().line_h()) * 0.5;
        ui.text(tab.rect.x + 10.0, ty.floor(), &tab.label, color);
        // Click (release without drag) activates.
        if ui.input().released(MouseButton::Left)
            && state.pressed_panel_index == Some(i)
            && state.dragging_index.is_none()
            && hot
        {
            tree.activate(tab.panel);
        }
    }
    for d in &layout.dividers {
        let hot = d.rect.contains(mx, my) || state.div_drag.map(|x| x.split) == Some(d.split);
        if hot {
            ui.draw_list_mut().fill_rect(d.rect, t.accent.with_alpha(90));
        }
    }

    DockFrame { layout, drop }
}

/// Finish the dock frame: draw the drag ghost + drop overlay above panel content
/// and apply the drop on release.
pub fn dock_finish<P: Copy + PartialEq>(
    ui: &mut Ui<'_>,
    tree: &mut DockTree<P>,
    state: &mut DockState,
    frame: &DockFrame<P>,
) {
    let t = *ui.theme();
    let released = ui.input().released(MouseButton::Left);
    let (mx, my) = (ui.input().mouse_x, ui.input().mouse_y);

    if let Some(i) = state.dragging_index {
        // Drop-zone preview.
        if let Some((_, _, preview)) = frame.drop {
            ui.draw_list_mut().fill_rect(preview, t.accent.with_alpha(50));
            ui.draw_list_mut().rect_outline(preview, 1.5, t.accent);
        }
        // Ghost tab at the cursor.
        if let Some(label) = &state.dragging_label {
            let w = ui.font().measure_line(label) + 20.0;
            let g = Rect::new(mx + 10.0, my + 10.0, w, TAB_H);
            ui.draw_list_mut().fill_rect(g, t.button_active.with_alpha(230));
            ui.draw_list_mut().rect_outline(g, 1.0, t.accent);
            let label = label.clone();
            ui.text(g.x + 10.0, g.y + 4.0, &label, t.text);
        }
        if released {
            if let (Some(tab), Some((leaf, zone, _))) = (frame.layout.tabs.get(i), frame.drop) {
                tree.move_panel(tab.panel, leaf, zone);
            }
            state.dragging_index = None;
            state.dragging_label = None;
            state.press = None;
            state.pressed_panel_index = None;
        }
    } else if released {
        state.press = None;
        state.pressed_panel_index = None;
    }
}

/// Which [`DropZone`] of `region` the point is over, plus the preview rect for
/// the overlay: the central area joins the tab group; the outer quarters split.
pub fn drop_zone_at(region: Rect, x: f32, y: f32) -> (DropZone, Rect) {
    let fx = ((x - region.x) / region.w.max(1.0)).clamp(0.0, 1.0);
    let fy = ((y - region.y) / region.h.max(1.0)).clamp(0.0, 1.0);
    let (hw, hh) = (region.w * 0.5, region.h * 0.5);
    // Distance from centre in each axis, past which the edge zones start.
    if fx < 0.25 {
        (DropZone::Left, Rect::new(region.x, region.y, hw, region.h))
    } else if fx > 0.75 {
        (DropZone::Right, Rect::new(region.x + hw, region.y, hw, region.h))
    } else if fy < 0.25 {
        (DropZone::Top, Rect::new(region.x, region.y, region.w, hh))
    } else if fy > 0.75 {
        (DropZone::Bottom, Rect::new(region.x, region.y + hh, region.w, hh))
    } else {
        (DropZone::Center, region)
    }
}
