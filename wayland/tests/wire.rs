//! Wire-format + event-parsing integration tests for the hand-written Wayland client
//! (spec: task deliverable 5). These are all CPU-only and need no compositor.
//!
//! 1. **Serialisation round-trips**, table-driven: headers, every arg kind, and the 32-bit
//!    padding for strings/arrays.
//! 2. **Event parsing from canned server byte streams**: a `wl_registry.global`, a
//!    `wl_display.error`, and an `xdg_toplevel.configure` decoded exactly as a real
//!    compositor would send them — proving the parser never panics on well-formed *or*
//!    malformed input.

use sc_wayland::wire::{pad4, ArgReader, Header, MessageBuilder};

// --- 1. serialisation round-trips (table-driven) -----------------------------------------

/// An argument spec for the table: the builder call + the expected decoded value.
enum Arg {
    U32(u32),
    I32(i32),
    Str(&'static str),
}

#[test]
fn message_headers_and_args_round_trip_across_the_table() {
    // (object, opcode, args) — a spread of the requests the client actually sends.
    let cases: &[(u32, u16, &[Arg])] = &[
        // wl_display.get_registry(new_id) — one u32.
        (1, 1, &[Arg::U32(2)]),
        // wl_surface.commit() — no args, minimal 8-byte message.
        (5, 6, &[]),
        // wl_surface.attach(object, x, y) — object + two ints (one negative).
        (9, 1, &[Arg::U32(12), Arg::I32(0), Arg::I32(-3)]),
        // wl_shm_pool.create_buffer(new_id, offset, w, h, stride, format).
        (
            10,
            0,
            &[
                Arg::U32(13),
                Arg::I32(0),
                Arg::I32(640),
                Arg::I32(360),
                Arg::I32(640 * 4),
                Arg::U32(1),
            ],
        ),
        // xdg_toplevel.set_title(string) — exercises string length+NUL+padding.
        (14, 2, &[Arg::Str("streamcraft — play")]),
        // wl_registry.bind(name, interface, version, new_id) — the multi-kind bind encoding.
        (2, 0, &[Arg::U32(7), Arg::Str("wl_shm"), Arg::U32(1), Arg::U32(15)]),
    ];

    for (object, opcode, args) in cases {
        let mut b = MessageBuilder::new(*object, *opcode);
        for a in *args {
            match a {
                Arg::U32(v) => {
                    b.u32(*v);
                }
                Arg::I32(v) => {
                    b.i32(*v);
                }
                Arg::Str(s) => {
                    b.string(s);
                }
            }
        }
        let bytes = b.finish();

        // Size is always a multiple of 4 and matches the header.
        assert_eq!(bytes.len() % 4, 0, "message {object}/{opcode} is word-aligned");
        let h = Header::parse(&bytes).expect("header parses");
        assert_eq!(h.object, *object);
        assert_eq!(h.opcode, *opcode);
        assert_eq!(h.size as usize, bytes.len(), "size field matches actual length");

        // Decode the body back and compare each arg.
        let mut r = ArgReader::new(&bytes[Header::LEN..]);
        for a in *args {
            match a {
                Arg::U32(v) => assert_eq!(r.u32(), Some(*v)),
                Arg::I32(v) => assert_eq!(r.i32(), Some(*v)),
                Arg::Str(s) => assert_eq!(r.string().as_deref(), Some(*s)),
            }
        }
        assert_eq!(r.remaining(), 0, "no trailing bytes for {object}/{opcode}");
    }
}

#[test]
fn string_padding_matches_pad4_for_various_lengths() {
    // For each string, the body length is 4 (len word) + pad4(bytes+1 NUL).
    for s in ["", "a", "ab", "abc", "abcd", "hello world"] {
        let mut b = MessageBuilder::new(1, 0);
        b.string(s);
        let bytes = b.finish();
        let body = bytes.len() - Header::LEN;
        assert_eq!(body, 4 + pad4(s.len() + 1), "padding for {s:?}");
        let mut r = ArgReader::new(&bytes[Header::LEN..]);
        assert_eq!(r.string().as_deref(), Some(s));
    }
}

// --- 2. event parsing from canned server byte streams ------------------------------------

/// Build a canned event exactly as a server would frame it: header + LE args.
fn canned_event(object: u32, opcode: u16, body: &[u8]) -> Vec<u8> {
    let size = (Header::LEN + body.len()) as u32;
    let mut out = Vec::new();
    out.extend_from_slice(&object.to_le_bytes());
    out.extend_from_slice(&(((size) << 16) | opcode as u32).to_le_bytes());
    out.extend_from_slice(body);
    out
}

#[test]
fn parses_a_canned_wl_registry_global_event() {
    // wl_registry.global(name=7, interface="wl_compositor", version=6) on registry object 2.
    let mut body = Vec::new();
    body.extend_from_slice(&7u32.to_le_bytes());
    let iface = "wl_compositor";
    body.extend_from_slice(&((iface.len() + 1) as u32).to_le_bytes());
    body.extend_from_slice(iface.as_bytes());
    body.push(0);
    while body.len() % 4 != 0 {
        body.push(0);
    }
    body.extend_from_slice(&6u32.to_le_bytes());
    let ev = canned_event(2, 0, &body);

    let h = Header::parse(&ev).unwrap();
    assert_eq!((h.object, h.opcode, h.size as usize), (2, 0, ev.len()));
    let mut r = ArgReader::new(&ev[Header::LEN..]);
    assert_eq!(r.u32(), Some(7));
    assert_eq!(r.string().as_deref(), Some("wl_compositor"));
    assert_eq!(r.u32(), Some(6));
    assert_eq!(r.remaining(), 0);
}

#[test]
fn parses_a_canned_wl_display_error_event() {
    // wl_display.error(object_id=9, code=2, message="invalid buffer") on the display (id 1).
    let mut body = Vec::new();
    body.extend_from_slice(&9u32.to_le_bytes());
    body.extend_from_slice(&2u32.to_le_bytes());
    let msg = "invalid buffer";
    body.extend_from_slice(&((msg.len() + 1) as u32).to_le_bytes());
    body.extend_from_slice(msg.as_bytes());
    body.push(0);
    while body.len() % 4 != 0 {
        body.push(0);
    }
    let ev = canned_event(1, 0, &body);
    let mut r = ArgReader::new(&ev[Header::LEN..]);
    assert_eq!(r.u32(), Some(9));
    assert_eq!(r.u32(), Some(2));
    assert_eq!(r.string().as_deref(), Some("invalid buffer"));
}

#[test]
fn parses_a_canned_xdg_toplevel_configure_with_states_array() {
    // xdg_toplevel.configure(width=800, height=600, states=[maximized(1), activated(4)]).
    let mut body = Vec::new();
    body.extend_from_slice(&800i32.to_le_bytes());
    body.extend_from_slice(&600i32.to_le_bytes());
    // states array: length 8, then two u32 enum values.
    body.extend_from_slice(&8u32.to_le_bytes());
    body.extend_from_slice(&1u32.to_le_bytes());
    body.extend_from_slice(&4u32.to_le_bytes());
    let ev = canned_event(14, 0, &body);
    let mut r = ArgReader::new(&ev[Header::LEN..]);
    assert_eq!(r.i32(), Some(800));
    assert_eq!(r.i32(), Some(600));
    let states = r.array().unwrap();
    assert_eq!(states.len(), 8);
    assert_eq!(u32::from_le_bytes(states[0..4].try_into().unwrap()), 1);
    assert_eq!(u32::from_le_bytes(states[4..8].try_into().unwrap()), 4);
    assert_eq!(r.remaining(), 0);
}

#[test]
fn malformed_events_never_panic_and_yield_none() {
    // A header claiming a size but with a truncated body: the parser must not read past it.
    let mut ev = canned_event(2, 0, &[]);
    // Corrupt: pretend there is a string arg but supply no bytes.
    ev.truncate(Header::LEN); // header only
    let mut r = ArgReader::new(&ev[Header::LEN..]);
    assert_eq!(r.u32(), None, "no body → None, not a panic");
    assert_eq!(r.string(), None);
    assert_eq!(r.array(), None);

    // A string length larger than the remaining body is rejected.
    let mut body = Vec::new();
    body.extend_from_slice(&999u32.to_le_bytes());
    body.extend_from_slice(b"xy");
    let ev = canned_event(2, 0, &body);
    let mut r = ArgReader::new(&ev[Header::LEN..]);
    assert_eq!(r.string(), None, "over-long length rejected");

    // A sub-header-length buffer parses to None.
    assert!(Header::parse(&[0u8, 1, 2]).is_none());
}
