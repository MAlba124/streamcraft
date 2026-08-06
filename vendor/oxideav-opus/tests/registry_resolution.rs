//! Registry-level payload-magic resolution — the `register(ctx)`
//! entry point's `OpusHead` claim (RFC 7845 §5.1 item 1).
//!
//! The Opus identification header opens with the fixed 8-octet
//! signature `"OpusHead"`; carriage formats that have no codec tag
//! (an Ogg logical stream's first packet is the canonical case)
//! resolve the codec from that payload prefix. These tests pin the
//! positive resolution and the refusals: the RFC 7845 §5.2 comment
//! header (`"OpusTags"`) and every proper truncation of the magic
//! must NOT resolve to `opus`.

use oxideav_core::stream::CodecResolver;
use oxideav_core::{CodecId, RuntimeContext};
use oxideav_opus::opus_head::OPUS_HEAD_MAGIC;

fn registered_context() -> RuntimeContext {
    let mut ctx = RuntimeContext::new();
    oxideav_opus::register(&mut ctx);
    ctx
}

#[test]
fn opus_head_prefix_resolves_to_opus() {
    let ctx = registered_context();
    // A realistic §5.1 identification header: magic, version 1,
    // 2 channels, pre-skip 312, input rate 48 kHz, gain 0, family 0.
    let mut head = Vec::new();
    head.extend_from_slice(OPUS_HEAD_MAGIC);
    head.extend_from_slice(&[1, 2]);
    head.extend_from_slice(&312u16.to_le_bytes());
    head.extend_from_slice(&48_000u32.to_le_bytes());
    head.extend_from_slice(&[0, 0, 0]);
    assert_eq!(
        ctx.codecs.resolve_payload_magic_ref(&head),
        Some(&CodecId::new("opus"))
    );
}

#[test]
fn exact_length_magic_resolves() {
    let ctx = registered_context();
    // A packet that is nothing but the 8-byte magic still matches —
    // prefix matching includes the exact-length case.
    assert_eq!(
        ctx.codecs.resolve_payload_magic_ref(b"OpusHead"),
        Some(&CodecId::new("opus"))
    );
}

#[test]
fn resolver_trait_surface_agrees() {
    let ctx = registered_context();
    // The dyn-facing CodecResolver path must agree with the inherent
    // method (it is what container crates actually call).
    let resolver: &dyn CodecResolver = &ctx.codecs;
    assert_eq!(
        resolver.resolve_payload_magic(b"OpusHead\x01\x02\x38\x01"),
        Some(CodecId::new("opus"))
    );
}

#[test]
fn opus_tags_comment_header_does_not_resolve() {
    let ctx = registered_context();
    // RFC 7845 §5.2: the second header packet opens with "OpusTags".
    // It shares 4 leading octets with "OpusHead" but is NOT an
    // identification header and must not resolve.
    let mut tags = Vec::new();
    tags.extend_from_slice(b"OpusTags");
    tags.extend_from_slice(&9u32.to_le_bytes());
    tags.extend_from_slice(b"oxideav 0");
    tags.extend_from_slice(&0u32.to_le_bytes());
    assert_eq!(ctx.codecs.resolve_payload_magic_ref(&tags), None);
}

#[test]
fn truncations_of_the_magic_do_not_resolve() {
    let ctx = registered_context();
    // Every proper prefix of the magic, from 7 bytes down to empty:
    // a payload SHORTER than the claimed prefix carries insufficient
    // evidence and must not match.
    for len in (0..OPUS_HEAD_MAGIC.len()).rev() {
        let truncated = &OPUS_HEAD_MAGIC[..len];
        assert_eq!(
            ctx.codecs.resolve_payload_magic_ref(truncated),
            None,
            "truncation to {len} bytes must not resolve"
        );
    }
}

#[test]
fn corrupted_final_octet_does_not_resolve() {
    let ctx = registered_context();
    let mut wrong = *OPUS_HEAD_MAGIC;
    wrong[7] ^= 0x20; // "OpusHeaD"
    assert_eq!(ctx.codecs.resolve_payload_magic_ref(wrong.as_slice()), None);
}

#[test]
fn unrelated_payloads_do_not_resolve() {
    let ctx = registered_context();
    for payload in [
        b"\x01vorbis\x00".as_slice(),
        b"RIFF\x00\x00\x00\x00".as_slice(),
        b"\x00\x00\x00\x00\x00\x00\x00\x00".as_slice(),
    ] {
        assert_eq!(ctx.codecs.resolve_payload_magic_ref(payload), None);
    }
}
