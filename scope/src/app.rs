//! The inspector application: the live panels over the protocol client (spec:
//! Introspection protocol and scraft-scope — "what it shows, all live").
//!
//! v1 panels: the **graph view** (topology via the layered layout, per-edge
//! negotiated formats, per-node live counters), an **elements** panel (throughput
//! derived as Δbytes/Δnow between counter samples — the observer's arithmetic,
//! spec: taps), the merged **events and logs** feed, and pause/resume. Latency
//! histograms, property editing, and the MCP server are later phases.

use std::path::Path;

use crate::client::{Client, Model};
use crate::layout::{self, Layout};
use crate::ui::backend::Backend;
use crate::ui::widgets::{LogView, UiState};
use crate::ui::{Color, Font, Rect, Ui};

/// Options for [`run`]. `max_frames` renders N frames then exits (headless CI).
#[derive(Default)]
pub struct AppOpts {
    pub max_frames: Option<u64>,
}

/// Per-node display data derived from the last topology.
struct NodeMeta {
    id: u32,
    name: String,
}

/// Layout cache + display strings, rebuilt when `Model::topo_gen` changes.
#[derive(Default)]
struct GraphCache {
    laid: Option<Layout>,
    gen: u64,
    nodes: Vec<NodeMeta>,
    edge_labels: Vec<String>,
}

/// Per-element derived rates (spec: taps — the window is the observer's policy).
struct ElemRate {
    id: u32,
    bytes_per_s: f64,
    buffers_out: u64,
    queue_hw: u32,
    drops: u64,
}

/// Connect to `path` and run the inspector window until closed (or `max_frames`).
pub fn run(path: &Path, opts: AppOpts) -> Result<(), String> {
    let mut client = Client::connect(path).map_err(|e| format!("connect {path:?}: {e}"))?;

    let font = Font::new();
    let mut backend = Backend::open("scraft-scope", 1100, 700, &font)
        .map_err(|e| format!("open window: {e}"))?;

    let mut ui_state = UiState::default();
    let mut log = LogView::new(2048);
    let mut graph = GraphCache::default();
    let mut paused = false;

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
        let (status, rates, fatal) = {
            let mut m = client.model.lock().unwrap_or_else(|e| e.into_inner());
            drain_feed(&mut m, &mut log);
            relayout_if_needed(&mut graph, &m, &font);
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
            (status, element_rates(&m), m.error.clone())
        };

        let mut ui = Ui::new(&input, &font, &mut ui_state, size);
        ui.clear_background();

        // --- top bar: status + pause/resume ---
        let pad = 8.0;
        let top_h = 26.0;
        {
            let dim = ui.theme().text_dim;
            let text = ui.theme().text;
            ui.text(pad, pad + 5.0, "scraft-scope", text);
            ui.text(pad + 110.0, pad + 5.0, &status, dim);
            let btn = Rect::new(size.0 - 90.0 - pad, pad, 90.0, top_h - 6.0);
            let label = if paused { "Resume" } else { "Pause" };
            if ui.button_in(label, btn) {
                paused = !paused;
                client.set_paused(paused);
            }
        }
        if let Some(err) = &fatal {
            ui.text(pad + 420.0, pad + 5.0, err, Color::rgb(0xe0, 0x6c, 0x4c));
        }

        // --- panel geometry ---
        let body_y = pad + top_h;
        let log_h = 170.0f32.min((size.1 - body_y) * 0.4);
        let cols_h = (size.1 - body_y - log_h - 2.0 * pad).max(80.0);
        let right_w = 300.0f32.min(size.0 * 0.35);
        let graph_w = size.0 - right_w - 3.0 * pad;

        draw_graph_panel(&mut ui, Rect::new(pad, body_y, graph_w, cols_h), &graph, &rates);
        draw_elements_panel(
            &mut ui,
            Rect::new(pad * 2.0 + graph_w, body_y, right_w, cols_h),
            &graph,
            &rates,
        );

        let log_rect = Rect::new(pad, body_y + cols_h + pad, size.0 - 2.0 * pad, log_h);
        ui.begin_panel("Events & logs", log_rect);
        ui.log_view(&mut log);
        ui.end_panel();

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
            "scraft-scope: {frame} frames, {nel} elements, {ned} edges, counters={}, log lines={}",
            m.cur.is_some(),
            log.len(),
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
fn relayout_if_needed(graph: &mut GraphCache, m: &Model, font: &Font) {
    if graph.laid.is_some() && graph.gen == m.topo_gen {
        return;
    }
    let Some(topo) = &m.topo else { return };

    let mut nodes = Vec::with_capacity(topo.elements.len());
    graph.nodes.clear();
    for e in &topo.elements {
        let (tw, _) = font.measure(&e.name);
        nodes.push(layout::Node { id: e.id, w: tw + 20.0, h: 44.0, group: e.group });
        graph.nodes.push(NodeMeta { id: e.id, name: e.name.clone() });
    }
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
    graph.edge_labels = topo
        .edges
        .iter()
        .map(|e| {
            if e.fields.is_empty() {
                e.family.clone()
            } else {
                format!("{} {}", e.family, e.fields.join(" "))
            }
        })
        .collect();

    let opts = layout::Opts { layer_gap: 70.0, node_gap: 26.0, ..layout::Opts::default() };
    graph.laid = Some(layout::layout(&nodes, &edges, &opts));
    graph.gen = m.topo_gen;
}

/// Rates are Δ between the last two counter samples.
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

/// The graph view: group hulls, routed edges with format labels, node boxes with
/// live per-node counters.
fn draw_graph_panel(ui: &mut Ui<'_>, rect: Rect, graph: &GraphCache, rates: &[ElemRate]) {
    let body = ui.begin_panel("Graph", rect);
    ui.end_panel();

    let dim = ui.theme().text_dim;
    let text = ui.theme().text;
    let border = ui.theme().panel_border;
    let accent = ui.theme().accent;
    let node_bg = ui.theme().panel_bg;

    let Some(laid) = &graph.laid else {
        ui.text(body.x + 8.0, body.y + 8.0, "waiting for topology...", dim);
        return;
    };

    // Content offset inside the panel body, clipped to it.
    let ox = body.x + 12.0;
    let oy = body.y + 12.0;

    ui.draw_list_mut().push_clip(body);

    // Group hulls first (dim outlines behind everything).
    for g in &laid.groups {
        let r = Rect::new(ox + g.x - 6.0, oy + g.y - 6.0, g.w + 12.0, g.h + 12.0);
        ui.draw_list_mut().rect_outline(r, 1.0, border);
    }

    // Edges: routed polylines + a format label near the first segment.
    for (i, e) in laid.edges.iter().enumerate() {
        for w in e.points.windows(2) {
            let (ax, ay) = w[0];
            let (bx, by) = w[1];
            ui.draw_list_mut().line(ox + ax, oy + ay, ox + bx, oy + by, 1.0, dim);
        }
        if let (Some(label), Some(&(ax, ay)), Some(&(bx, by))) =
            (graph.edge_labels.get(i), e.points.first(), e.points.get(1))
        {
            let (mx, my) = ((ax + bx) * 0.5, (ay + by) * 0.5);
            ui.text(ox + mx + 4.0, oy + my - 10.0, label, dim);
        }
    }

    // Nodes: box, name, live stats line.
    for n in &laid.nodes {
        let r = Rect::new(ox + n.x, oy + n.y, n.w, n.h);
        ui.draw_list_mut().fill_rect(r, node_bg);
        ui.draw_list_mut().rect_outline(r, 1.0, accent);
        let name = graph
            .nodes
            .iter()
            .find(|m| m.id == n.id)
            .map(|m| m.name.as_str())
            .unwrap_or("?");
        ui.text(r.x + 8.0, r.y + 6.0, name, text);
        if let Some(rate) = rate_of(rates, n.id) {
            let stats = format!("{} q{}", fmt_bitrate(rate.bytes_per_s), rate.queue_hw);
            ui.text(r.x + 8.0, r.y + 24.0, &stats, dim);
        }
    }

    ui.draw_list_mut().pop_clip();
}

/// The elements panel: per-element throughput (relative bar) + counters.
fn draw_elements_panel(ui: &mut Ui<'_>, rect: Rect, graph: &GraphCache, rates: &[ElemRate]) {
    let bar_fill = ui.theme().bar_fill;
    ui.begin_panel("Elements", rect);
    let max_rate = rates.iter().map(|r| r.bytes_per_s).fold(1.0f64, f64::max);
    for meta in &graph.nodes {
        let Some(r) = rate_of(rates, meta.id) else { continue };
        ui.label(&meta.name);
        let frac = (r.bytes_per_s / max_rate) as f32;
        let color = if r.drops > 0 { Color::rgb(0xe0, 0x6c, 0x4c) } else { bar_fill };
        ui.fill_bar(frac, color, &fmt_bitrate(r.bytes_per_s));
        ui.kv("buffers", &format!("{}  q{}  drops {}", r.buffers_out, r.queue_hw, r.drops));
    }
    if graph.nodes.is_empty() {
        ui.label_dim("no elements yet");
    }
    ui.end_panel();
}
