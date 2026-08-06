//! Ogg metadata for all four audio mappings this scanner recognises: Opus, Vorbis, FLAC and
//! Speex.
//!
//! One module, because at the container level they are the same file. RFC 3533 §6 defines a
//! physical Ogg bitstream as a sequence of pages, each carrying a *segment table* whose
//! lacing values split its payload into packets — a lacing value below 255 ends a packet, a
//! value of exactly 255 says it continues (§6, field 9). Every mapping then puts the same two
//! things in the same two places:
//!
//! | packet | Opus (RFC 7845)      | Vorbis (Vorbis I §4.2) | FLAC (Ogg Mapping for FLAC §3) | Speex (manual §7.3) |
//! |--------|----------------------|------------------------|--------------------------------|---------------------|
//! | 1      | `OpusHead`           | `0x01 "vorbis"` ident  | `0x7F "FLAC"` + native `fLaC` + STREAMINFO | `"Speex   "` header |
//! | 2…     | `OpusTags`           | `0x03 "vorbis"` comment| the remaining metadata blocks, one per packet | a bare Vorbis comment |
//!
//! Speex is the odd one on packet 2: it inherits the *comment format* and nothing else, so
//! that packet carries **no magic, no packet-type octet and no framing bit** — it begins
//! directly with the vendor-string length, and goes to `parse_comment_body` rather than to one
//! of the wrappers (`ogg/spec/SPEEX.md`).
//!
//! and none of them states a duration anywhere: "the position specified is the total samples
//! encoded after including all packets finished on this page" (RFC 3533 §6), so the duration
//! *is* the granule position of the last page. That is what makes the read plan below the
//! shape it is.
//!
//! ## What it costs
//!
//! **Two reads for any real file**: the prefix (both header packets, since they are the first
//! few KiB of the stream) and one tail (the last page's granule). A comment packet carrying
//! cover art — a `METADATA_BLOCK_PICTURE` is base64, so 200 KiB of JPEG is 270 KiB of text —
//! overruns a 128 KiB prefix and costs one more: [`plan`] walks page *headers* (which state
//! their own length without their payload being present) to the end of the last header packet
//! and asks for exactly that range. Past [`MAX_META`] the scan keeps the tags it can reach
//! and drops the art: tags without a cover beat no tags at all.
//!
//! ## Allocation
//!
//! A packet inside one page is contiguous in the read buffer and reaches the parser
//! **borrowed**. Only a packet spanning pages must be made contiguous, and that copy goes to
//! the per-file [`Arena`] — a bump, not the heap. Ogg cover art is base64 and therefore
//! decoded rather than sliced no matter what, so unlike FLAC and MP4 there is no zero-copy
//! picture path to lose here.

use profluens_core::event::TagSink;
use profluens_core::memory::Arena;
use pf_ogg::page::{PageHeader, CAPTURE_PATTERN, HEADER_FIXED_LEN};

use crate::{Format, Props};

/// Pages one file's header walk may visit. A `MAX_RUNS`-page packet plus its neighbours fits
/// comfortably; past this the file is not a header chain, it is an attack.
const MAX_PAGES: usize = 192;

/// Page-spanning runs one packet may be gathered from — 64 maximum-size pages is ~4 MiB of
/// cover art, past which the packet is skipped (see the module docs).
const MAX_RUNS: usize = 64;

/// How much further to read when the header walk runs out of window. Big enough that a
/// picture-carrying comment packet is reached in one more read (8 maximum-size pages), small
/// enough that a truncated file does not turn into a multi-megabyte read.
const REACH: u64 = 512 * 1024;

/// Ceiling on the head window. Four extents of [`REACH`] never exceed it, so this is the
/// backstop, not the limit that normally binds.
pub(crate) const MAX_META: u64 = 4 << 20;

/// Header packets to walk for Opus, Vorbis and Speex: the identification header and the
/// comment header, and nothing else (RFC 7845 §5, Vorbis I §4.2 — the third Vorbis header is
/// the setup codebooks, which are not metadata; Speex manual §7.3 — the comment is packet 1
/// unconditionally, whatever `extra_headers` says, because that field counts packets *after*
/// it).
const WANT_XIPH: usize = 2;

/// Cap on the FLAC mapping's declared header-packet count (§3 allows 0 = "unknown").
const MAX_FLAC_HEADERS: usize = 8;

/// Bytes of the Ogg-FLAC mapping header in packet 1 before the native FLAC stream begins:
/// `0x7F "FLAC"` (5), major + minor mapping version (2), and the 16-bit header-packet count
/// (2) — "Ogg Mapping for FLAC" §3.
const FLAC_MAP_HEADER: usize = 9;

/// What the header packets said, plus what the tail read needs to turn a granule position
/// into a duration.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Head {
    /// Serial of the logical bitstream the headers came from (RFC 3533 §6, field 5), so the
    /// tail scan times *this* stream and not a grouped video track.
    pub serial: Option<u32>,
    pub props: Props,
    /// Opus decoder priming, subtracted from the last granule (RFC 7845 §4).
    pre_skip: u16,
    /// The rate a granule position counts in. 48 kHz for Opus (always, §4), the stream's own
    /// sample rate for Vorbis and FLAC.
    granule_rate: u32,
}

/// How many packets the head window must reach for `format`.
pub(crate) fn want(window: &[u8], body: u64, format: Format) -> usize {
    match format {
        Format::OggFlac => flac_header_packets(window, body),
        _ => WANT_XIPH,
    }
}

/// The absolute file offset the head window must reach to hold `want` complete packets.
///
/// Walks page *headers* only: a page states its own length in its segment table (RFC 3533
/// §6 — `page_size = 27 + page_segments + sum(lacing values)`), so the extent of a
/// multi-page comment packet is computable from a few hundred bytes per page without any of
/// its payload being present. `None` means the bytes at `body` are not a page at all.
pub(crate) fn plan(window: &[u8], body: u64, file_len: u64, want: usize) -> Option<u64> {
    let mut at = usize::try_from(body).ok()?;
    let mut done = 0usize;
    for _ in 0..MAX_PAGES {
        let Some(h) = window.get(at..).and_then(|r| r.get(..HEADER_FIXED_LEN)) else {
            return Some(reach(at as u64, file_len));
        };
        if h[..4] != CAPTURE_PATTERN {
            return None;
        }
        let n = h[26] as usize;
        let table_end = at + HEADER_FIXED_LEN + n;
        let Some(table) = window.get(at + HEADER_FIXED_LEN..table_end) else {
            return Some(reach(table_end as u64, file_len));
        };
        // §6, field 9: a lacing value below 255 terminates a packet.
        done += table.iter().filter(|&&b| b < 255).count();
        at = table_end + table.iter().map(|&b| b as usize).sum::<usize>();
        if done >= want {
            break;
        }
    }
    Some((at as u64).min(file_len))
}

fn reach(at: u64, file_len: u64) -> u64 {
    at.saturating_add(REACH).min(file_len)
}

/// Read the header packets in `window` (which starts at file offset 0, the stream itself at
/// `body`): properties into the returned [`Head`], tags into `sink`.
pub(crate) fn scan(
    window: &[u8],
    body: u64,
    format: Format,
    arena: &Arena,
    sink: &mut impl TagSink,
) -> Head {
    let mut head = Head::default();
    match format {
        Format::OggOpus => {
            let mut ident = None;
            let serial = for_each_packet(window, body, arena, WANT_XIPH, |i, packet| {
                if i == 0 {
                    ident = pf_ogg::ident::parse_opus_head(packet);
                } else {
                    pf_ogg::comment::parse_opus_tags(packet, arena, sink);
                }
                true
            });
            head.serial = serial;
            if let Some(id) = ident {
                head.props.channels = Some(u32::from(id.channels));
                // RFC 7845 §5.1 is explicit that `input_sample_rate` "is not the sample rate
                // to use for playback": Opus always decodes at 48 kHz, which is also the rate
                // its granule positions count in (§4). Reporting the decode rate is therefore
                // the one figure that is true of the stream as it exists — and it is what
                // every player and `ffprobe` shows for an Opus file.
                head.props.sample_rate = Some(pf_ogg::duration::OPUS_GRANULE_RATE);
                head.pre_skip = id.pre_skip;
            }
            head.granule_rate = pf_ogg::duration::OPUS_GRANULE_RATE;
        }
        Format::OggVorbis => {
            let mut ident = None;
            let serial = for_each_packet(window, body, arena, WANT_XIPH, |i, packet| {
                if i == 0 {
                    ident = pf_ogg::ident::parse_vorbis_ident(packet);
                } else {
                    pf_ogg::comment::parse_vorbis_comments(packet, arena, sink);
                }
                true
            });
            head.serial = serial;
            if let Some(id) = ident {
                head.props.channels = Some(u32::from(id.channels));
                head.props.sample_rate = Some(id.sample_rate);
                head.granule_rate = id.sample_rate;
            }
        }
        Format::OggSpeex => {
            let mut ident = None;
            let serial = for_each_packet(window, body, arena, WANT_XIPH, |i, packet| {
                if i == 0 {
                    ident = pf_ogg::ident::parse_speex_head(packet);
                } else {
                    // No magic, no packet-type octet, no framing bit — the raw comment body
                    // (Speex manual §7.3, and `ogg/spec/SPEEX.md`).
                    pf_ogg::comment::parse_comment_body(packet, arena, sink);
                }
                true
            });
            head.serial = serial;
            if let Some(id) = ident {
                head.props.channels = Some(u32::from(id.channels));
                head.props.sample_rate = Some(id.sample_rate);
                // "the granulepos is the number of the last sample encoded in that packet"
                // (§7.3) — counted at the stream's own rate, exactly like Vorbis, so the two
                // share `duration`'s arithmetic below.
                head.granule_rate = id.sample_rate;
            }
        }
        Format::OggFlac => {
            // "Ogg Mapping for FLAC" §3: after the mapping header, packet 1 holds "the fLaC
            // signature and the STREAMINFO block", and packets 2..n hold "the remaining
            // metadata blocks ... in the same format as in native FLAC". Concatenating them
            // therefore rebuilds a byte-exact native metadata chain, which is what lets this
            // reuse `pf_flac::tags` — the workspace's RFC 9639 §8.6/§8.8 reader — instead of
            // a second implementation that would have to be kept in step with it.
            let mut chain: Vec<u8, &Arena> = Vec::new_in(arena);
            let serial = for_each_packet(window, body, arena, want(window, body, format), |i, packet| {
                let block = if i == 0 { packet.get(FLAC_MAP_HEADER..) } else { Some(packet) };
                let Some(block) = block else { return false };
                chain.extend_from_slice(block);
                // §8.1: the block header's top bit is the last-block flag. Stopping on it
                // bounds the walk by the file's own structure rather than by a guess.
                let header = if i == 0 { block.get(4) } else { block.first() };
                header.is_none_or(|h| h & 0x80 == 0)
            });
            head.serial = serial;
            if let Some(si) = pf_flac::tags::stream_info(&chain) {
                head.props.channels = Some(u32::from(si.channels));
                if si.sample_rate > 0 {
                    head.props.sample_rate = Some(si.sample_rate);
                    head.granule_rate = si.sample_rate;
                    if si.total_samples > 0 {
                        // u128: `total_samples` is a 36-bit field (§8.2).
                        head.props.duration_ns = u64::try_from(
                            u128::from(si.total_samples) * 1_000_000_000 / u128::from(si.sample_rate),
                        )
                        .ok();
                        head.props.duration_exact = true;
                    }
                }
            }
            pf_flac::tags::parse_into(&chain, sink);
        }
        _ => {}
    }
    head
}

/// Duration from the tail read: the last page's granule position, converted by the mapping's
/// own rule (RFC 7845 §4 for Opus; the Ogg Vorbis mapping, "Ogg Mapping for FLAC" §4 and the
/// Speex manual §7.3 all count PCM samples at the stream's sample rate, so they share the
/// arithmetic).
///
/// Speex is the one mapping where this is measurably a *floor* rather than the nominal input
/// length: its encoder subtracts its algorithmic lookahead from the granule, and unlike Opus's
/// `pre_skip` that constant is nowhere in the file. The shortfall against `ffprobe` is 5–11 ms
/// depending on the mode; `ogg/spec/SPEEX.md` records the measurements and why adding a
/// hard-coded constant back would be worse than reporting what the stream states.
pub(crate) fn duration(tail: &[u8], head: &Head, format: Format) -> Option<u64> {
    let granule = pf_ogg::duration::last_granule(tail, head.serial)?;
    match format {
        Format::OggOpus => Some(pf_ogg::duration::opus_duration_ns(granule, head.pre_skip)),
        _ => pf_ogg::duration::vorbis_duration_ns(granule, head.granule_rate),
    }
}

/// The declared header-packet count of an Ogg-FLAC stream, plus the mapping packet itself
/// ("Ogg Mapping for FLAC" §3: a 16-bit big-endian count of "the number of header packets,
/// not including this one"; 0 means the count is not known in advance).
fn flac_header_packets(window: &[u8], body: u64) -> usize {
    let fallback = MAX_FLAC_HEADERS;
    let Ok(at) = usize::try_from(body) else { return fallback };
    let Some(h) = window.get(at..).and_then(|r| r.get(..HEADER_FIXED_LEN)) else { return fallback };
    let payload = at + HEADER_FIXED_LEN + h[26] as usize;
    let Some(id) = window.get(payload..).and_then(|r| r.get(..FLAC_MAP_HEADER)) else {
        return fallback;
    };
    if id[..5] != *b"\x7fFLAC" {
        return fallback;
    }
    match u16::from_be_bytes([id[7], id[8]]) as usize {
        0 => fallback,
        n => (n + 1).min(MAX_FLAC_HEADERS),
    }
}

/// Walk the packets of the first logical bitstream in `window`, starting at `body`, calling
/// `visit(index, packet)` until it returns false or `limit` packets have been seen. Returns
/// the stream's serial number.
///
/// Every page is validated by [`PageHeader::parse`], CRC included (RFC 3533 §6, field 7), so
/// a corrupt or truncated page ends the walk rather than yielding garbage. A page belonging
/// to another serial ends it too: that is a grouped or chained stream (§4), whose headers are
/// not this one's.
fn for_each_packet<'a>(
    window: &'a [u8],
    body: u64,
    arena: &'a Arena,
    limit: usize,
    mut visit: impl FnMut(usize, &'a [u8]) -> bool,
) -> Option<u32> {
    let mut at = usize::try_from(body).ok()?;
    let mut serial: Option<u32> = None;
    // The byte runs of the packet under construction: one per page it spans.
    let mut runs = [(0usize, 0usize); MAX_RUNS];
    let mut n_runs = 0usize;
    let mut total = 0usize;
    let mut dropped = false;
    let mut index = 0usize;

    for _ in 0..MAX_PAGES {
        let Some(rest) = window.get(at..) else { break };
        let Ok(page) = PageHeader::parse(rest) else { break };
        match serial {
            None => serial = Some(page.serial()),
            Some(s) if s != page.serial() => break,
            _ => {}
        }
        let mut off = at + HEADER_FIXED_LEN + page.n_segments();
        let mut start = off;
        let mut run = 0usize;
        for &lace in page.segment_table() {
            off += lace as usize;
            run += lace as usize;
            if lace == 255 {
                continue; // §6, field 9: the packet continues into the next segment/page
            }
            if n_runs < MAX_RUNS {
                runs[n_runs] = (start, run);
                n_runs += 1;
                total += run;
            } else {
                dropped = true;
            }
            if !dropped && !visit(index, gather(window, &runs[..n_runs], total, arena)) {
                return serial;
            }
            index += 1;
            if index >= limit {
                return serial;
            }
            (n_runs, total, dropped) = (0, 0, false);
            start = off;
            run = 0;
        }
        if run > 0 {
            if n_runs < MAX_RUNS {
                runs[n_runs] = (start, run);
                n_runs += 1;
                total += run;
            } else {
                dropped = true;
            }
        }
        at += page.len();
    }
    serial
}

/// The packet described by `runs` as one contiguous slice: borrowed from `window` when it is
/// a single run (the overwhelmingly common case — a packet inside one page), copied into the
/// bump `arena` when it spans pages.
fn gather<'a>(window: &'a [u8], runs: &[(usize, usize)], total: usize, arena: &'a Arena) -> &'a [u8] {
    if let [(start, len)] = *runs {
        return start.checked_add(len).and_then(|e| window.get(start..e)).unwrap_or(&[]);
    }
    let region: &'a mut [u8] = arena.alloc_bytes(total);
    let mut n = 0;
    for &(start, len) in runs {
        let Some(src) = start.checked_add(len).and_then(|e| window.get(start..e)) else { break };
        region[n..n + len].copy_from_slice(src);
        n += len;
    }
    let out: &'a [u8] = region;
    &out[..n]
}

#[cfg(test)]
mod tests {
    #![allow(clippy::disallowed_methods)] // fixture building in tests — one-time (spec: allocation discipline)

    use super::*;
    use profluens_core::event::{TagList, TagListSink};
    use profluens_core::memory::Pool;

    use crate::fixture::{ogg_flac, ogg_opus, ogg_speex, ogg_vorbis};

    fn run(f: &[u8], format: Format) -> (Props, TagList) {
        let pool = Pool::bounded(4096, 16);
        let arena = Arena::default();
        let mut sink = TagListSink::new(&pool);
        let mut head = scan(f, 0, format, &arena, &mut sink);
        if let Some(ns) = duration(f, &head, format) {
            head.props.duration_ns = Some(ns);
            head.props.duration_exact = true;
        }
        (head.props, sink.finish())
    }

    #[test]
    fn opus_head_tags_and_exact_duration() {
        // 7 s at Opus's fixed 48 kHz granule rate, plus the usual 312-sample priming.
        let granule = 48_000 * 7 + 312;
        let f = ogg_opus(2, 312, 44_100, &["TITLE=Rain", "ARTIST=Someone"], None, granule);
        let (props, tags) = run(&f, Format::OggOpus);
        assert_eq!(props.channels, Some(2));
        assert_eq!(props.sample_rate, Some(48_000), "Opus decodes at 48 kHz (RFC 7845 §5.1)");
        assert_eq!(props.duration_ns, Some(7_000_000_000));
        assert!(props.duration_exact);
        assert_eq!(tags.get("TITLE"), Some("Rain"));
        assert_eq!(tags.get("ARTIST"), Some("Someone"));
    }

    #[test]
    fn vorbis_ident_tags_and_duration() {
        let f = ogg_vorbis(2, 44_100, &["TITLE=Vor", "ALBUM=Bis"], 44_100 * 3);
        let (props, tags) = run(&f, Format::OggVorbis);
        assert_eq!(props.channels, Some(2));
        assert_eq!(props.sample_rate, Some(44_100));
        assert_eq!(props.duration_ns, Some(3_000_000_000));
        assert_eq!(tags.get("TITLE"), Some("Vor"));
        assert_eq!(tags.get("ALBUM"), Some("Bis"));
    }

    /// Speex's comment packet has no magic, no packet-type octet and no framing bit; a parser
    /// that reached for one of the wrapper readers would find nothing at all.
    #[test]
    fn speex_head_bare_comments_and_duration() {
        let f = ogg_speex(16_000, 1, &["TITLE=Spx", "ARTIST=Someone"], None, 16_000 * 3);
        let (props, tags) = run(&f, Format::OggSpeex);
        assert_eq!(props.sample_rate, Some(16_000));
        assert_eq!(props.channels, Some(1));
        assert_eq!(props.duration_ns, Some(3_000_000_000));
        assert!(props.duration_exact);
        assert_eq!(tags.get("TITLE"), Some("Spx"));
        assert_eq!(tags.get("ARTIST"), Some("Someone"));
    }

    /// A picture rides a Speex comment the same way it rides a Vorbis one — base64 in a
    /// `METADATA_BLOCK_PICTURE` value — and one big enough to span pages still gathers.
    #[test]
    fn a_speex_comment_carries_cover_art() {
        let art = vec![0x4Du8; 120_000];
        let f = ogg_speex(8_000, 2, &["TITLE=Art"], Some(("image/jpeg", &art)), 8_000);
        let want = want(&f, 0, Format::OggSpeex);
        assert_eq!(want, WANT_XIPH);
        let (props, tags) = run(&f, Format::OggSpeex);
        assert_eq!(props.channels, Some(2));
        assert_eq!(props.duration_ns, Some(1_000_000_000));
        assert_eq!(tags.get("TITLE"), Some("Art"));
        assert_eq!(tags.pictures()[0].data.data(), &art[..]);
    }

    #[test]
    fn ogg_flac_rebuilds_the_native_chain() {
        let f = ogg_flac(48_000, 1, 48_000 * 2, &["TITLE=Chain"], Some(("image/png", &[7u8; 900])));
        let (props, tags) = run(&f, Format::OggFlac);
        assert_eq!(props.sample_rate, Some(48_000));
        assert_eq!(props.channels, Some(1));
        assert_eq!(props.duration_ns, Some(2_000_000_000));
        assert_eq!(tags.get("TITLE"), Some("Chain"));
        assert_eq!(tags.pictures().len(), 1, "a PICTURE block in its own packet still lands");
        assert_eq!(tags.pictures()[0].data.data(), &[7u8; 900][..]);
    }

    #[test]
    fn a_packet_spanning_pages_is_gathered() {
        // A picture far past one page's 65025-byte payload maximum (§6).
        let art = vec![0x2Bu8; 150_000];
        let f = ogg_opus(2, 312, 48_000, &["TITLE=Big"], Some(("image/jpeg", &art)), 48_000 + 312);
        let want = want(&f, 0, Format::OggOpus);
        assert_eq!(plan(&f, 0, f.len() as u64, want), Some(f.len() as u64 - last_page_len(&f)));

        let (props, tags) = run(&f, Format::OggOpus);
        assert_eq!(props.duration_ns, Some(1_000_000_000));
        assert_eq!(tags.get("TITLE"), Some("Big"));
        assert_eq!(tags.pictures()[0].data.data(), &art[..]);
    }

    /// Length of the file's final page, for the plan assertion above.
    fn last_page_len(f: &[u8]) -> u64 {
        let mut at = 0usize;
        let mut last = 0usize;
        while let Ok(p) = PageHeader::parse(&f[at..]) {
            last = p.len();
            at += p.len();
            if at >= f.len() {
                break;
            }
        }
        last as u64
    }

    #[test]
    fn a_short_window_asks_for_more_and_never_regresses() {
        let art = vec![0x11u8; 150_000];
        let f = ogg_opus(1, 312, 48_000, &["TITLE=Short"], Some(("image/jpeg", &art)), 48_312);
        let len = f.len() as u64;
        for cut in [0usize, 1, 26, 27, 100, 4096, 65_536, f.len() / 2] {
            let cut = cut.min(f.len());
            if let Some(end) = plan(&f[..cut], 0, len, WANT_XIPH) {
                assert!(end > cut as u64 || end == len, "cut {cut}: plan {end} makes no progress");
            }
        }
    }

    #[test]
    fn truncation_never_panics() {
        let pool = Pool::bounded(4096, 16);
        let files = [
            (ogg_opus(2, 312, 48_000, &["TITLE=T"], Some(("image/png", &[1u8; 4000])), 96_312), Format::OggOpus),
            (ogg_vorbis(1, 8_000, &["TITLE=T"], 8_000), Format::OggVorbis),
            (ogg_flac(44_100, 2, 44_100, &["TITLE=T"], None), Format::OggFlac),
            (ogg_speex(16_000, 1, &["TITLE=T"], None, 16_000), Format::OggSpeex),
        ];
        for (f, format) in files {
            for n in (0..f.len()).step_by(13).chain(0..64) {
                let n = n.min(f.len());
                let arena = Arena::default();
                let mut sink = TagListSink::new(&pool);
                let head = scan(&f[..n], 0, format, &arena, &mut sink);
                let _ = duration(&f[..n], &head, format);
                let _ = plan(&f[..n], 0, n as u64, want(&f[..n], 0, format));
            }
        }
    }

    #[test]
    fn junk_is_not_a_page() {
        let junk: Vec<u8> = (0..4096).map(|i| (i * 37 % 256) as u8).collect();
        assert_eq!(plan(&junk, 0, junk.len() as u64, 2), None);
        let arena = Arena::default();
        let pool = Pool::bounded(4096, 4);
        let mut sink = TagListSink::new(&pool);
        let head = scan(&junk, 0, Format::OggOpus, &arena, &mut sink);
        assert!(head.serial.is_none());
        assert!(sink.finish().is_empty());
    }
}
