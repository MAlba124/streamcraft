//! The inspector application: the live panels over the protocol client (spec:
//! Introspection protocol and pf-scope — "what it shows, all live").
//!
//! Panels: the **graph view** (topology via the layered layout; pan with left-drag,
//! zoom with the wheel, click an edge to pin its negotiated format, hover for a
//! tooltip; groups get hue-tinted hulls), an **elements** panel (throughput derived
//! as Δbytes/Δnow between counter samples — the observer's arithmetic, spec: taps;
//! wheel-scrolls when overflowing), the merged **events and logs** feed, transport
//! (pause/resume + running time / duration when a format announced one), and
//! draggable splitters between the panes. Latency histograms, property editing,
//! and the MCP server are later phases.

use std::path::Path;

use crate::client::{Client, Model};
use crate::layout::{self, Layout};
use crate::ui::backend::Backend;
use crate::ui::dock::{self, DockState, DockTree};
use crate::ui::widgets::{LogView, UiState};
use crate::ui::{Color, Font, Input, Rect, Ui};

/// Options for [`run`]. `max_frames` renders N frames then exits (headless CI).
#[derive(Default)]
pub struct AppOpts {
    pub max_frames: Option<u64>,
}

/// Distinct muted hues for thread-group hulls, cycled by group id.
const GROUP_HUES: [Color; 8] = [
    Color::rgb(0x4a, 0xa8, 0xff),
    Color::rgb(0x5c, 0xc8, 0x78),
    Color::rgb(0xe0, 0xa8, 0x4c),
    Color::rgb(0xb0, 0x7c, 0xe8),
    Color::rgb(0x4c, 0xc8, 0xc0),
    Color::rgb(0xe0, 0x7c, 0xa8),
    Color::rgb(0xd8, 0xd0, 0x60),
    Color::rgb(0xe0, 0x6c, 0x5c),
];

/// Per-node display data derived from the last topology.
struct NodeMeta {
    id: u32,
    name: String,
}

/// Display strings for one edge (the caps view).
struct EdgeInfo {
    /// `src.pad -> dst.pad` with element names resolved.
    title: String,
    family: String,
    fields: Vec<String>,
}

/// Layout cache + display strings, rebuilt when `Model::topo_gen` changes.
#[derive(Default)]
struct GraphCache {
    laid: Option<Layout>,
    gen: u64,
    nodes: Vec<NodeMeta>,
    edges: Vec<EdgeInfo>,
}

/// Retained camera + interaction state for the graph view.
struct GraphView {
    pan: (f32, f32),
    zoom: f32,
    /// Topology generation the camera was last auto-fitted for.
    fitted_gen: u64,
    /// Mouse position at left-press inside the body (screen space).
    press: Option<(f32, f32)>,
    last_mouse: (f32, f32),
    panning: bool,
    /// Dragging the minimap's viewport rectangle.
    mm_drag: bool,
    hover_edge: Option<usize>,
    selected_edge: Option<usize>,
}

impl Default for GraphView {
    fn default() -> Self {
        Self {
            pan: (0.0, 0.0),
            zoom: 1.0,
            fitted_gen: u64::MAX,
            press: None,
            last_mouse: (0.0, 0.0),
            panning: false,
            mm_drag: false,
            hover_edge: None,
            selected_edge: None,
        }
    }
}

/// Per-element derived rates (spec: taps — the window is the observer's policy).
struct ElemRate {
    id: u32,
    bytes_per_s: f64,
    buffers_out: u64,
    queue_hw: u32,
    drops: u64,
}

/// The dockable panels (spec: what pf-scope shows; the dock model follows
/// simprof's tree-of-splits-with-tabbed-leaves).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum PanelId {
    Graph,
    Elements,
    Log,
}

fn panel_label(p: PanelId) -> String {
    match p {
        PanelId::Graph => "Graph",
        PanelId::Elements => "Elements",
        PanelId::Log => "Events & logs",
    }
    .into()
}

/// Connect to `path` and run the inspector window until closed (or `max_frames`).
pub fn run(path: &Path, opts: AppOpts) -> Result<(), String> {
    let mut client = Client::connect(path).map_err(|e| format!("connect {path:?}: {e}"))?;

    let font = Font::new();
    let mut backend = Backend::open("pf-scope", 1100, 700, &font)
        .map_err(|e| format!("open window: {e}"))?;

    let mut ui_state = UiState::default();
    let mut log = LogView::new(2048);
    let mut cache = GraphCache::default();
    let mut gview = GraphView::default();
    let mut paused = false;
    // The dock: graph | elements side by side, the log across the bottom.
    let mut dock = DockTree::two_over_one(PanelId::Graph, PanelId::Elements, PanelId::Log, 0.66, 0.7);
    let mut dock_state = DockState::default();
    let mut elements_scroll = 0.0f32;
    // In-flight progress-bar scrub: the fraction under the cursor; seek on release.
    let mut seek_drag: Option<f32> = None;

    // ~120 fps budget: a real compositor vsync-blocks in present, so this only
    // paces headless drivers (SDL_VIDEODRIVER=dummy present returns immediately).
    let frame_budget = std::time::Duration::from_millis(8);

    let mut frame: u64 = 0;
    loop {
        let frame_start = std::time::Instant::now();
        client.poll();
        let input = backend.begin_frame();
        if backend.should_close() {
            break;
        }
        let size = backend.window_size();

        // --- pull the frame's data out of the model (short lock), before any UI ---
        let (status, rates, fatal, position_ns, duration_ns) = {
            let mut m = client.model.lock().unwrap_or_else(|e| e.into_inner());
            drain_feed(&mut m, &mut log);
            relayout_if_needed(&mut cache, &m, &font);
            // The flagged `String::new()` is the empty else-branch of the status
            // `format!` (an empty String allocates nothing); the status line is a
            // short once-per-frame diagnostic, not per-buffer data (clippy.toml).
            #[allow(clippy::disallowed_methods)]
            let status = format!(
                "{} pid={} {}{}",
                client.socket_path,
                m.pid,
                if m.connected { "connected" } else { "DISCONNECTED" },
                if m.bus_dropped + m.logs_dropped > 0 {
                    format!("  dropped={}", m.bus_dropped + m.logs_dropped)
                } else {
                    String::new()
                },
            );
            let position = m.cur.as_ref().map(|c| c.now_ns).filter(|&t| t != u64::MAX);
            let duration = m.duration_ns;
            (status, element_rates(&m), m.error.clone(), position, duration)
        };

        let mut ui = Ui::new(&input, &font, &mut ui_state, size);
        ui.clear_background();

        // --- top bar: title, status, transport (time / duration), pause ---
        let pad = 8.0;
        let top_h = 28.0;
        let btn = Rect::new(size.0 - 90.0 - pad, pad, 90.0, top_h - 6.0);
        {
            let dim = ui.theme().text_dim;
            let text = ui.theme().text;
            let accent = ui.theme().accent;
            let bar_bg = ui.theme().bar_bg;
            // One shared mid-line for everything in the bar (title, status,
            // progress, time), matching the pause button's vertical centre.
            let mid = pad + (top_h - 6.0) * 0.5;
            let text_y = (mid - font.line_h() * 0.5).floor();
            let tw = ui.text(pad, text_y, "pf-scope", text);

            // Transport block right of centre: [progress bar] mm:ss / mm:ss.
            let mut left_limit = btn.x - 12.0;
            if let Some(pos) = position_ns {
                let time_text = match duration_ns {
                    Some(dur) => format!("{} / {}", fmt_time(pos), fmt_time(dur)),
                    None => fmt_time(pos),
                };
                let text_w = font.measure_line(&time_text);
                let bar_w = if duration_ns.is_some() { 160.0 } else { 0.0 };
                let x0 = btn.x - 12.0 - text_w - if bar_w > 0.0 { bar_w + 8.0 } else { 0.0 };
                if let Some(dur) = duration_ns {
                    let frac = (pos as f64 / dur.max(1) as f64).clamp(0.0, 1.0) as f32;
                    let bar = Rect::new(x0, mid - 4.0, bar_w, 8.0);

                    // Click / drag to seek: scrub shows the target; release seeks.
                    let hit = Rect::new(bar.x, bar.y - 5.0, bar.w, bar.h + 10.0);
                    let mouse_frac =
                        (((input.mouse_x - bar.x) / bar.w.max(1.0)).clamp(0.0, 1.0)).max(0.0);
                    if hit.contains(input.mouse_x, input.mouse_y)
                        && input.pressed(crate::ui::MouseButton::Left)
                    {
                        seek_drag = Some(mouse_frac);
                    }
                    if seek_drag.is_some() && input.down(crate::ui::MouseButton::Left) {
                        seek_drag = Some(mouse_frac);
                    }
                    let shown_frac = seek_drag.unwrap_or(frac);
                    ui.draw_list_mut().fill_rect(bar, bar_bg);
                    ui.draw_list_mut()
                        .fill_rect(Rect::new(bar.x, bar.y, bar.w * shown_frac, bar.h), accent);
                    if let Some(f) = seek_drag {
                        // Scrub marker + target time above the cursor.
                        let x = bar.x + bar.w * f;
                        ui.draw_list_mut().fill_rect(Rect::new(x - 1.0, bar.y - 3.0, 2.0, bar.h + 6.0), text);
                        let t = (f as f64 * dur as f64) as u64;
                        ui.text(x + 6.0, bar.y - 18.0, &fmt_time(t), text);
                        if input.released(crate::ui::MouseButton::Left) {
                            client.seek(t);
                            seek_drag = None;
                        }
                    }
                }
                ui.text(x0 + if bar_w > 0.0 { bar_w + 8.0 } else { 0.0 }, text_y, &time_text, text);
                left_limit = x0 - 16.0;
            }

            // Status, truncated to the space between the title and the transport.
            let avail = (left_limit - (pad + tw + 14.0)).max(0.0);
            let max_chars = (avail / font.cell_w()) as usize;
            let shown: String = if status.chars().count() > max_chars {
                status.chars().take(max_chars.saturating_sub(1)).chain("…".chars()).collect()
            } else {
                status.clone()
            };
            ui.text(pad + tw + 14.0, text_y, &shown, dim);

            let label = if paused { "Resume" } else { "Pause" };
            if ui.button_in(label, btn) {
                paused = !paused;
                client.set_paused(paused);
            }
        }
        if let Some(err) = &fatal {
            ui.text(pad, top_h + pad, err, Color::rgb(0xe0, 0x6c, 0x4c));
        }

        // --- the dockspace: tab bars, dividers, drag-to-dock (model: simprof) ---
        let area = Rect::new(
            pad,
            pad + top_h,
            (size.0 - 2.0 * pad).max(100.0),
            (size.1 - top_h - 2.0 * pad).max(120.0),
        );
        let frame_dock = dock::dock_begin(&mut ui, &mut dock, &mut dock_state, area, &panel_label);
        let busy = dock_state.interacting();
        for geom in &frame_dock.layout.panels {
            match geom.id {
                PanelId::Graph => {
                    draw_graph_panel(&mut ui, geom.rect, &cache, &mut gview, &rates, busy)
                }
                PanelId::Elements => {
                    draw_elements_panel(&mut ui, geom.rect, &cache, &rates, &mut elements_scroll)
                }
                PanelId::Log => {
                    ui.begin_region(geom.rect);
                    ui.log_view(&mut log);
                    ui.end_region();
                }
            }
        }
        dock::dock_finish(&mut ui, &mut dock, &mut dock_state, &frame_dock);

        let dl = ui.finish();
        backend.render(&dl, Color::rgb(0x14, 0x16, 0x1a));

        frame += 1;
        if let Some(max) = opts.max_frames {
            if frame >= max {
                break;
            }
        }
        if let Some(rest) = frame_budget.checked_sub(frame_start.elapsed()) {
            std::thread::sleep(rest);
        }
    }

    // A one-line summary so headless CI runs leave evidence of what arrived.
    {
        let m = client.model.lock().unwrap_or_else(|e| e.into_inner());
        let (nel, ned) = m.topo.as_ref().map(|t| (t.elements.len(), t.edges.len())).unwrap_or((0, 0));
        eprintln!(
            "pf-scope: {frame} frames, {nel} elements, {ned} edges, counters={}, log lines={}, duration={}",
            m.cur.is_some(),
            log.len(),
            m.duration_ns.map(fmt_time).unwrap_or_else(|| "unknown".into()),
        );
    }
    Ok(())
}

/// Move pending feed lines into the LogView, colored by severity.
fn drain_feed(m: &mut Model, log: &mut LogView) {
    while let Some(line) = m.feed.pop_front() {
        let color = match line.severity {
            1 => Color::rgb(0xe0, 0x5c, 0x4c),                  // error
            2 => Color::rgb(0xe0, 0xa8, 0x4c),                  // warn
            3 if line.is_event => Color::rgb(0x4a, 0xa8, 0xff), // bus events: accent
            3 => Color::rgb(0xd8, 0xdd, 0xe4),                  // info
            4 => Color::rgb(0x8a, 0x93, 0xa0),                  // debug
            _ => Color::rgb(0x5c, 0x64, 0x70),                  // trace
        };
        log.push(line.text, color);
    }
}

/// Node box sizing + the layered layout, recomputed only when the topology changes
/// (`topo_gen`), never per frame.
// Cold: early-returns unless `topo_gen` changed, so these Vecs are built once per
// topology change, not per frame (clippy.toml allocation ban).
#[allow(clippy::disallowed_methods)]
fn relayout_if_needed(cache: &mut GraphCache, m: &Model, font: &Font) {
    if cache.laid.is_some() && cache.gen == m.topo_gen {
        return;
    }
    let Some(topo) = &m.topo else { return };

    let mut nodes = Vec::with_capacity(topo.elements.len());
    cache.nodes.clear();
    for e in &topo.elements {
        // The box holds two lines: the name (14 px) and the live stats line
        // (11 px, e.g. "27.1 Mb/s q4"). Size for the wider of the two — short
        // names ("tee") otherwise overflow their box with the stats text.
        let (name_w, _) = font.measure_px(&e.name, 14.0);
        let (stats_w, _) = font.measure_px("999.9 Mb/s q99", 11.0);
        nodes.push(layout::Node {
            id: e.id,
            w: name_w.max(stats_w) + 16.0,
            h: 46.0,
            group: e.group,
        });
        cache.nodes.push(NodeMeta { id: e.id, name: e.name.clone() });
    }
    let name_of = |id: u32| {
        topo.elements
            .iter()
            .find(|e| e.id == id)
            .map(|e| e.name.as_str())
            .unwrap_or("?")
    };
    let edges: Vec<layout::Edge> = topo
        .edges
        .iter()
        .map(|e| layout::Edge {
            src: e.src,
            src_port: e.src_pad as u16,
            dst: e.sink,
            dst_port: e.sink_pad as u16,
        })
        .collect();
    cache.edges = topo
        .edges
        .iter()
        .map(|e| EdgeInfo {
            title: format!(
                "{}.{} -> {}.{}",
                name_of(e.src),
                e.src_pad,
                name_of(e.sink),
                e.sink_pad
            ),
            family: e.family.clone(),
            fields: e.fields.clone(),
        })
        .collect();

    // Gaps sized against the hull chrome: horizontally each hull adds
    // GROUP_PAD on both sides (24 px between neighbours), vertically
    // pad + pad + label strip (42 px) — the defaults left 6 px of daylight.
    let opts = layout::Opts {
        layer_gap: 84.0,
        node_gap: 26.0,
        group_gap: 64.0,
        ..layout::Opts::default()
    };
    cache.laid = Some(layout::layout(&nodes, &edges, &opts));
    cache.gen = m.topo_gen;
}

/// Rates are Δ between the last two counter samples.
// The flagged `Vec::new()` is the empty no-data early-return (an empty Vec allocates
// nothing); the populated path returns via `.collect()`. The `Vec<ElemRate>` result
// is a pub-facing per-frame collection whose size (element count) is small and fixed,
// not per-buffer geometry (clippy.toml allocation ban).
#[allow(clippy::disallowed_methods)]
fn element_rates(m: &Model) -> Vec<ElemRate> {
    let Some(cur) = &m.cur else { return Vec::new() };
    let dt_ns = match &m.prev {
        Some(prev)
            if cur.now_ns != u64::MAX && prev.now_ns != u64::MAX && cur.now_ns > prev.now_ns =>
        {
            (cur.now_ns - prev.now_ns) as f64
        }
        _ => 0.0,
    };
    cur.rows
        .iter()
        .map(|row| {
            let prev_bytes = m
                .prev
                .as_ref()
                .and_then(|p| p.rows.iter().find(|r| r.element == row.element))
                .map(|r| r.bytes_out)
                .unwrap_or(row.bytes_out);
            let bytes_per_s = if dt_ns > 0.0 {
                (row.bytes_out.saturating_sub(prev_bytes)) as f64 * 1e9 / dt_ns
            } else {
                0.0
            };
            ElemRate {
                id: row.element,
                bytes_per_s,
                buffers_out: row.buffers_out,
                queue_hw: row.queue_high_water,
                drops: row.drops,
            }
        })
        .collect()
}

fn rate_of(rates: &[ElemRate], id: u32) -> Option<&ElemRate> {
    rates.iter().find(|r| r.id == id)
}

fn fmt_bitrate(bytes_per_s: f64) -> String {
    let bits = bytes_per_s * 8.0;
    if bits >= 1e6 {
        format!("{:.1} Mb/s", bits / 1e6)
    } else if bits >= 1e3 {
        format!("{:.0} kb/s", bits / 1e3)
    } else {
        format!("{bits:.0} b/s")
    }
}

/// `m:ss` (or `h:mm:ss` past an hour) from nanoseconds.
pub fn fmt_time(ns: u64) -> String {
    let s = ns / 1_000_000_000;
    let (h, m, sec) = (s / 3600, (s % 3600) / 60, s % 60);
    if h > 0 {
        format!("{h}:{m:02}:{sec:02}")
    } else {
        format!("{m}:{sec:02}")
    }
}

/// Distance from point `p` to segment `a`-`b` (pure; the edge hit-test).
pub fn dist_point_segment(p: (f32, f32), a: (f32, f32), b: (f32, f32)) -> f32 {
    let (abx, aby) = (b.0 - a.0, b.1 - a.1);
    let (apx, apy) = (p.0 - a.0, p.1 - a.1);
    let len2 = abx * abx + aby * aby;
    let t = if len2 <= f32::EPSILON { 0.0 } else { ((apx * abx + apy * aby) / len2).clamp(0.0, 1.0) };
    let (cx, cy) = (a.0 + t * abx - p.0, a.1 + t * aby - p.1);
    (cx * cx + cy * cy).sqrt()
}

/// Zoom `(pan, zoom)` by `factor`, keeping the content point under `anchor`
/// (body-relative) stationary. Pure; clamped to [0.2, 3.0].
pub fn zoom_at(pan: (f32, f32), zoom: f32, anchor: (f32, f32), factor: f32) -> ((f32, f32), f32) {
    let new_zoom = (zoom * factor).clamp(0.2, 3.0);
    let k = new_zoom / zoom;
    let pan = (anchor.0 - (anchor.0 - pan.0) * k, anchor.1 - (anchor.1 - pan.1) * k);
    (pan, new_zoom)
}

/// Fit the camera to `laid`'s bounds inside a `w`×`h` body (pure).
pub fn fit_camera(bounds: (f32, f32, f32, f32), w: f32, h: f32) -> ((f32, f32), f32) {
    let (x0, y0, x1, y1) = bounds;
    let (bw, bh) = ((x1 - x0).max(1.0), (y1 - y0).max(1.0));
    let margin = 24.0;
    let zoom = (((w - margin) / bw).min((h - margin) / bh)).clamp(0.2, 1.4);
    let pan = (
        (w - bw * zoom) * 0.5 - x0 * zoom,
        (h - bh * zoom) * 0.5 - y0 * zoom,
    );
    (pan, zoom)
}

fn content_bounds(laid: &Layout) -> (f32, f32, f32, f32) {
    let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
    for n in &laid.nodes {
        x0 = x0.min(n.x);
        y0 = y0.min(n.y);
        x1 = x1.max(n.x + n.w);
        y1 = y1.max(n.y + n.h);
    }
    for g in &laid.groups {
        x0 = x0.min(g.x);
        y0 = y0.min(g.y);
        x1 = x1.max(g.x + g.w);
        y1 = y1.max(g.y + g.h);
    }
    if x0 > x1 {
        (0.0, 0.0, 1.0, 1.0)
    } else {
        (x0, y0, x1, y1)
    }
}

/// The graph view: camera interactions + group hulls, routed edges, node boxes.
fn draw_graph_panel(
    ui: &mut Ui<'_>,
    rect: Rect,
    cache: &GraphCache,
    gv: &mut GraphView,
    rates: &[ElemRate],
    splitting: bool,
) {
    let dim = ui.theme().text_dim;
    let text = ui.theme().text;
    let accent = ui.theme().accent;
    let node_bg = ui.theme().panel_bg;
    let title_bg = ui.theme().title_bg;
    let border = ui.theme().panel_border;

    // Region chrome (the dock's tab bar replaces the old title).
    ui.draw_list_mut().fill_rect(rect, node_bg);
    ui.draw_list_mut().rect_outline(rect, 1.0, border);
    let body = rect.inset(1.0);

    // "Fit" button + zoom readout overlay, top-right of the content.
    let fit_btn = Rect::new(body.right() - 48.0, body.y + 4.0, 42.0, 20.0);
    let refit = ui.button_in("Fit", fit_btn);

    let Some(laid) = &cache.laid else {
        ui.text(body.x + 8.0, body.y + 8.0, "waiting for topology...", dim);
        return;
    };

    // --- camera: auto-fit per topology, wheel zoom at cursor, drag pan ---
    if refit || gv.fitted_gen != cache.gen {
        let (pan, zoom) = fit_camera(content_bounds(laid), body.w, body.h);
        gv.pan = pan;
        gv.zoom = zoom;
        gv.fitted_gen = cache.gen;
        gv.selected_edge = None;
    }
    let input: Input = ui.input().clone();
    let mouse = (input.mouse_x, input.mouse_y);
    let over = body.contains(mouse.0, mouse.1) && !splitting;
    let rel = (mouse.0 - body.x, mouse.1 - body.y);

    // --- minimap geometry + interaction (steals the press from pan/select) ---
    let bounds = content_bounds(laid);
    let (bx0, by0, bx1, by1) = bounds;
    let (bw, bh) = ((bx1 - bx0).max(1.0), (by1 - by0).max(1.0));
    let mm_scale = ((body.w * 0.28).clamp(80.0, 220.0) / bw)
        .min((body.h * 0.28).clamp(60.0, 170.0) / bh);
    let mm = Rect::new(
        body.right() - bw * mm_scale - 10.0,
        body.bottom() - bh * mm_scale - 10.0,
        bw * mm_scale,
        bh * mm_scale,
    );
    let over_mm = over && mm.contains(mouse.0, mouse.1);
    if over_mm && input.pressed(crate::ui::MouseButton::Left) {
        gv.mm_drag = true;
    }
    if !input.down(crate::ui::MouseButton::Left) {
        gv.mm_drag = false;
    }
    if gv.mm_drag {
        // Centre the viewport on the content point under the cursor.
        let c = ((mouse.0 - mm.x) / mm_scale + bx0, (mouse.1 - mm.y) / mm_scale + by0);
        gv.pan = (body.w * 0.5 - c.0 * gv.zoom, body.h * 0.5 - c.1 * gv.zoom);
    }

    if over && !over_mm && input.wheel != 0.0 {
        let factor = 1.1f32.powf(input.wheel);
        let (pan, zoom) = zoom_at(gv.pan, gv.zoom, rel, factor);
        gv.pan = pan;
        gv.zoom = zoom;
    }
    if over && !over_mm && !gv.mm_drag && input.pressed(crate::ui::MouseButton::Left) {
        gv.press = Some(mouse);
        gv.panning = false;
    }
    if input.down(crate::ui::MouseButton::Left) && !gv.mm_drag {
        if let Some(origin) = gv.press {
            let moved = (mouse.0 - origin.0).abs() + (mouse.1 - origin.1).abs();
            if gv.panning || moved > 4.0 {
                gv.panning = true;
                gv.pan.0 += mouse.0 - gv.last_mouse.0;
                gv.pan.1 += mouse.1 - gv.last_mouse.1;
            }
        }
    }
    let zoom = gv.zoom;
    let to_screen =
        |p: (f32, f32)| (body.x + gv.pan.0 + p.0 * zoom, body.y + gv.pan.1 + p.1 * zoom);

    // Edge hover: nearest polyline within 7 screen px.
    gv.hover_edge = None;
    if over && !over_mm && !gv.panning && !gv.mm_drag {
        let mut best = 7.0f32;
        for (i, e) in laid.edges.iter().enumerate() {
            for w in e.points.windows(2) {
                let d = dist_point_segment(mouse, to_screen(w[0]), to_screen(w[1]));
                if d < best {
                    best = d;
                    gv.hover_edge = Some(i);
                }
            }
        }
    }
    if input.released(crate::ui::MouseButton::Left) {
        if gv.press.is_some() && !gv.panning && over {
            gv.selected_edge = gv.hover_edge;
        }
        gv.press = None;
        gv.panning = false;
    }
    gv.last_mouse = mouse;

    // --- draw, clipped to the body ---
    ui.draw_list_mut().push_clip(body);

    // Group hulls: hue-tinted fill + border + label. The hull reserves a strip
    // above the content for the label so nodes never paint over it.
    const GROUP_PAD: f32 = 12.0; // graph units around the content
    const GROUP_LABEL_H: f32 = 18.0; // extra headroom for the label strip
    for g in &laid.groups {
        let hue = GROUP_HUES[g.group as usize % GROUP_HUES.len()];
        let (x, y) = to_screen((g.x - GROUP_PAD, g.y - GROUP_PAD - GROUP_LABEL_H));
        let r = Rect::new(
            x,
            y,
            (g.w + 2.0 * GROUP_PAD) * zoom,
            (g.h + 2.0 * GROUP_PAD + GROUP_LABEL_H) * zoom,
        );
        ui.draw_list_mut().fill_rect(r, hue.with_alpha(18));
        ui.draw_list_mut().rect_outline(r, 1.0, hue.with_alpha(150));
        let px = (12.0 * zoom).clamp(8.0, 16.0);
        if 12.0 * zoom >= 7.0 {
            ui.text_px(r.x + 5.0, r.y + 3.0, &format!("group {}", g.group), px, hue.with_alpha(220));
        }
    }

    // Edges: routed polylines; hovered/selected get the accent and more weight.
    for (i, e) in laid.edges.iter().enumerate() {
        let active = gv.selected_edge == Some(i) || gv.hover_edge == Some(i);
        let (col, th) = if active { (accent, 2.0) } else { (dim, 1.0) };
        for w in e.points.windows(2) {
            let (ax, ay) = to_screen(w[0]);
            let (bx, by) = to_screen(w[1]);
            ui.draw_list_mut().line(ax, ay, bx, by, th, col);
        }
    }

    // Nodes: box, name, live stats.
    for n in &laid.nodes {
        let (x, y) = to_screen((n.x, n.y));
        let r = Rect::new(x, y, n.w * zoom, n.h * zoom);
        ui.draw_list_mut().fill_rect(r, node_bg);
        ui.draw_list_mut().rect_outline(r, 1.0, accent.with_alpha(180));
        let name_px = 14.0 * zoom;
        if name_px >= 6.5 {
            let name = cache
                .nodes
                .iter()
                .find(|m| m.id == n.id)
                .map(|m| m.name.as_str())
                .unwrap_or("?");
            ui.text_px(r.x + 8.0 * zoom, r.y + 6.0 * zoom, name, name_px.min(30.0), text);
        }
        if 11.0 * zoom >= 7.0 {
            if let Some(rate) = rate_of(rates, n.id) {
                let stats = format!("{} q{}", fmt_bitrate(rate.bytes_per_s), rate.queue_hw);
                ui.text_px(r.x + 8.0 * zoom, r.y + 26.0 * zoom, &stats, 11.0 * zoom, dim);
            }
        }
    }

    // Edge family labels — drawn after the nodes so a label near a box lands on
    // top of it, never underneath (full caps live on hover/click).
    if 11.0 * zoom >= 7.5 {
        for (i, e) in laid.edges.iter().enumerate() {
            // Label the LONGEST polyline segment's midpoint: the first segment
            // hugs the source port, so labels there pile onto node boxes and
            // hull borders (the cramped-graph render bug).
            let longest = e.points.windows(2).max_by(|a, b| {
                let la = (a[1].0 - a[0].0).hypot(a[1].1 - a[0].1);
                let lb = (b[1].0 - b[0].0).hypot(b[1].1 - b[0].1);
                la.total_cmp(&lb)
            });
            if let (Some(info), Some(w)) = (cache.edges.get(i), longest) {
                let (ax, ay) = to_screen(w[0]);
                let (bx2, by2) = to_screen(w[1]);
                let (mx, my) = ((ax + bx2) * 0.5, (ay + by2) * 0.5);
                ui.text_px(mx + 4.0, my - 12.0 * zoom, &info.family, 11.0 * zoom, dim);
            }
        }
    }

    // Minimap: whole-content overview bottom-right, with the visible viewport as a
    // movable rectangle (drag inside the minimap to recentre the camera).
    {
        ui.draw_list_mut().fill_rect(mm, Color::rgba(0x0f, 0x11, 0x15, 225));
        ui.draw_list_mut().rect_outline(mm, 1.0, border);
        for n in &laid.nodes {
            let r = Rect::new(
                mm.x + (n.x - bx0) * mm_scale,
                mm.y + (n.y - by0) * mm_scale,
                (n.w * mm_scale).max(1.5),
                (n.h * mm_scale).max(1.5),
            );
            ui.draw_list_mut().fill_rect(r, dim.with_alpha(170));
        }
        // Viewport rectangle: the content region currently visible in the body.
        let v = Rect::new(
            mm.x + (-gv.pan.0 / zoom - bx0) * mm_scale,
            mm.y + (-gv.pan.1 / zoom - by0) * mm_scale,
            body.w / zoom * mm_scale,
            body.h / zoom * mm_scale,
        );
        let v = v.intersect(&mm);
        if v.w > 2.0 && v.h > 2.0 {
            ui.draw_list_mut().rect_outline(v, 1.5, accent);
            ui.draw_list_mut().fill_rect(v, accent.with_alpha(20));
        }
    }

    // Caps: hover tooltip near the cursor; click pins the details bottom-left.
    let caps_box = |ui: &mut Ui<'_>, info: &EdgeInfo, x: f32, y: f32| {
        let mut lines = vec![info.title.clone(), info.family.clone()];
        lines.extend(info.fields.iter().cloned());
        let lh = ui.font().line_h() + 2.0;
        let w = lines
            .iter()
            .map(|l| ui.font().measure_line(l))
            .fold(0.0f32, f32::max)
            + 16.0;
        let h = lines.len() as f32 * lh + 10.0;
        let bx = x.min(body.right() - w - 4.0).max(body.x + 2.0);
        let by = y.min(body.bottom() - h - 4.0).max(body.y + 2.0);
        let r = Rect::new(bx, by, w, h);
        ui.draw_list_mut().fill_rect(r, Color::rgb(0x0f, 0x11, 0x15).with_alpha(240));
        ui.draw_list_mut().rect_outline(r, 1.0, accent.with_alpha(160));
        let mut ty = by + 5.0;
        for (i, l) in lines.iter().enumerate() {
            let c = if i == 0 { text } else if i == 1 { accent } else { dim };
            ui.text(bx + 8.0, ty, l, c);
            ty += lh;
        }
    };
    if let Some(i) = gv.selected_edge {
        if let Some(info) = cache.edges.get(i) {
            let h = (info.fields.len() + 2) as f32 * (ui.font().line_h() + 2.0) + 10.0;
            caps_box(ui, info, body.x + 6.0, body.bottom() - h - 6.0);
        }
    }
    if let Some(i) = gv.hover_edge {
        if gv.selected_edge != Some(i) {
            if let Some(info) = cache.edges.get(i) {
                caps_box(ui, info, mouse.0 + 14.0, mouse.1 + 14.0);
            }
        }
    }

    ui.draw_list_mut().pop_clip();

    // Zoom readout, left of the Fit button.
    let ztext = format!("{:.0}%", zoom * 100.0);
    let zw = ui.font().measure_line(&ztext);
    ui.draw_list_mut().fill_rect(
        Rect::new(fit_btn.x - zw - 14.0, fit_btn.y, zw + 10.0, fit_btn.h),
        title_bg,
    );
    ui.text(fit_btn.x - zw - 9.0, fit_btn.y + 3.0, &ztext, dim);
}

/// The elements panel: per-element throughput (relative bar) + counters.
/// Wheel-scrolls when the list overflows the panel body.
fn draw_elements_panel(
    ui: &mut Ui<'_>,
    rect: Rect,
    cache: &GraphCache,
    rates: &[ElemRate],
    scroll: &mut f32,
) {
    let bar_fill = ui.theme().bar_fill;
    let row_h = ui.theme().row_h;
    let bar_bg = ui.theme().bar_bg;
    let text_dim = ui.theme().text_dim;
    let body = ui.begin_region(rect);

    // Row gap is the panel cursor's 4.0; each element renders 3 rows.
    let block_h = 3.0 * (row_h + 4.0);
    let content_h = cache.nodes.len() as f32 * block_h;
    let max_scroll = (content_h - body.h).max(0.0);
    if body.contains(ui.input().mouse_x, ui.input().mouse_y) && ui.input().wheel != 0.0 {
        *scroll -= ui.input().wheel * block_h * 0.75;
    }
    *scroll = scroll.clamp(0.0, max_scroll);
    ui.scroll_body(*scroll);

    let max_rate = rates.iter().map(|r| r.bytes_per_s).fold(1.0f64, f64::max);
    for meta in &cache.nodes {
        let Some(r) = rate_of(rates, meta.id) else { continue };
        ui.label(&meta.name);
        let frac = (r.bytes_per_s / max_rate) as f32;
        let color = if r.drops > 0 { Color::rgb(0xe0, 0x6c, 0x4c) } else { bar_fill };
        ui.fill_bar(frac, color, &fmt_bitrate(r.bytes_per_s));
        ui.kv("buffers", &format!("{}  q{}  drops {}", r.buffers_out, r.queue_hw, r.drops));
    }
    if cache.nodes.is_empty() {
        ui.label_dim("no elements yet");
    }

    // A thin scrollbar when the list overflows (same idiom as the log view).
    if max_scroll > 0.0 {
        let track = Rect::new(rect.right() - 5.0, body.y, 3.0, body.h);
        ui.draw_list_mut().fill_rect(track, bar_bg);
        let thumb_h = (body.h * (body.h / content_h)).max(8.0);
        let thumb_y = body.y + (body.h - thumb_h) * (*scroll / max_scroll);
        ui.draw_list_mut()
            .fill_rect(Rect::new(track.x, thumb_y, track.w, thumb_h), text_dim);
    }
    ui.end_region();
}
