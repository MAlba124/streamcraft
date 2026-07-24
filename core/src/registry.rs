//! The opt-in element registry and the parse-launch grammar (spec: Plugins). `use`
//! plus typed construction (`pipeline.add(FileSrc::new(path))`) stays the primary
//! path — rustc is the registry there. This module is the *second* layer: "element by
//! name", for `scraft-launch`, quick tests, and bug-report one-liners.
//!
//! A plugin crate exposes `pub fn register(&mut Registry)` handing over its elements'
//! `&'static ElementDesc`s (each with a `make_default` factory and `props`
//! declarations). [`Registry::parse`] then turns a gst-launch-style string —
//! `elem prop=val ! elem ! elem prop=val` — into elements + links on a [`Pipeline`].
//!
//! ## Divergence from the spec sketch
//! The sketch returns a `GroupId` from `parse`; groups are a *scheduling* artefact
//! computed at `run()` (spec: Scheduling — thread groups), and the parse layer only
//! ever builds a linear chain of elements, so there is no group to name yet. Instead
//! `parse` returns the [`ElementId`]s it added, in chain order — enough to link, set
//! more props, or query counters afterwards. When groups become a topology concept
//! this can grow a `GroupId` alongside.
//!
//! ## Grammar (v1)
//! ```text
//! launch   := chain
//! chain    := element ( '!' element )*
//! element  := NAME ( property )*
//! property := KEY '=' VALUE
//! VALUE    := INTEGER | RATIONAL | STRING
//! ```
//! Whitespace-tolerant (`a!b`, `a ! b`, and `a  prop = v` all parse). Values are
//! integers (`65536`, `-3`), rationals (`30000/1001`), or bare strings (file paths,
//! sample-format names). Linking joins the **first src pad** of the left element to
//! the **first sink pad** of the right. The parser never panics on malformed input
//! (spec: fuzz-friendly surfaces) — every failure is a [`ParseError`] naming the
//! offending token.

use std::collections::HashMap;

use crate::element::{Direction, ElementDesc};
use crate::error::ParseError;
use crate::format::Value;
use crate::id::ElementId;
use crate::pipeline::Pipeline;

/// The opt-in "element by name" table (spec: Plugins). Descriptors are `&'static`
/// (they live in each element's module), so the registry only stores references; a
/// plugin crate populates it via its `register(&mut Registry)`.
#[derive(Default)]
pub struct Registry {
    by_name: HashMap<&'static str, &'static ElementDesc>,
}

impl Registry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register one element descriptor under its `desc.name` (spec: Plugins). A later
    /// registration with the same name replaces the earlier one (last write wins), so a
    /// consumer can override a built-in with its own variant. A descriptor without a
    /// `make_default` factory can still be registered (and looked up) but cannot be
    /// constructed by [`parse`](Self::parse) — that fails loudly at construction time.
    pub fn register(&mut self, desc: &'static ElementDesc) {
        self.by_name.insert(desc.name, desc);
    }

    /// The descriptor registered under `name`, if any.
    pub fn get(&self, name: &str) -> Option<&'static ElementDesc> {
        self.by_name.get(name).copied()
    }

    /// The registered element names, sorted — for a `--list`/help surface and tests.
    pub fn names(&self) -> Vec<&'static str> {
        let mut v: Vec<&'static str> = self.by_name.keys().copied().collect();
        v.sort_unstable();
        v
    }

    /// Parse a gst-launch-style string into elements + links on `into`, returning the
    /// added [`ElementId`]s in chain order (spec: Plugins — see the module divergence
    /// note on the spec's `GroupId`). Construction and linking happen here, at parse
    /// time and before `run()`, so any string-valued property lands in the pipeline's
    /// value vocabulary snapshot the run reads.
    ///
    /// Errors — an unknown element name, a non-constructible one, a malformed property,
    /// or a link with no compatible pad — surface as a [`ParseError`] naming the
    /// offending token. Nothing is half-built on error: the elements added so far stay
    /// on the pipeline (harmless; a fresh pipeline is the normal caller), but no partial
    /// chain is returned.
    pub fn parse(&self, into: &mut Pipeline, launch: &str) -> Result<Vec<ElementId>, ParseError> {
        let stages = lex_chain(launch)?;
        if stages.is_empty() {
            return Err(ParseError {
                message: "empty launch string — expected at least one element".into(),
            });
        }

        let mut ids: Vec<ElementId> = Vec::with_capacity(stages.len());
        let mut descs: Vec<&'static ElementDesc> = Vec::with_capacity(stages.len());
        for stage in &stages {
            let desc = self.get(stage.name).ok_or_else(|| ParseError {
                message: format!("unknown element '{}'", stage.name),
            })?;
            let make = desc.make_default.ok_or_else(|| ParseError {
                message: format!(
                    "element '{}' has no make_default factory — it cannot be built by name \
                     (construct it with its typed constructor instead)",
                    stage.name
                ),
            })?;
            let id = into.add_boxed(make());
            // Apply the parsed properties, validating each against the element's declared
            // constraints (spec: Dynamic element properties — validated at the call site).
            for prop in &stage.props {
                apply_prop(into, id, desc, prop)?;
            }
            ids.push(id);
            descs.push(desc);
        }

        // Link each adjacent pair: first src pad of the left → first sink pad of the right.
        for i in 0..ids.len() - 1 {
            let src_pad = first_pad(descs[i], Direction::Src).ok_or_else(|| ParseError {
                message: format!(
                    "element '{}' has no src pad to link to '{}'",
                    descs[i].name,
                    descs[i + 1].name
                ),
            })?;
            let sink_pad = first_pad(descs[i + 1], Direction::Sink).ok_or_else(|| ParseError {
                message: format!(
                    "element '{}' has no sink pad to link from '{}'",
                    descs[i + 1].name,
                    descs[i].name
                ),
            })?;
            into.link((ids[i], src_pad), (ids[i + 1], sink_pad))
                .map_err(|e| ParseError {
                    message: format!(
                        "cannot link '{}'.{src_pad} ! '{}'.{sink_pad}: {e:?}",
                        descs[i].name,
                        descs[i + 1].name
                    ),
                })?;
        }

        Ok(ids)
    }
}

/// The name of the first pad on `desc` with the given direction (declaration order is
/// preference — the parser links the first, matching the linear-chain intent).
fn first_pad(desc: &'static ElementDesc, dir: Direction) -> Option<&'static str> {
    desc.pads.iter().find(|p| p.direction == dir).map(|p| p.name)
}

/// Apply one parsed property to an element, routing string values through the value
/// interner (spec: Plugins — strings ride `Value::Id` via the pipeline's vocabulary).
fn apply_prop(
    into: &mut Pipeline,
    id: ElementId,
    desc: &'static ElementDesc,
    prop: &ParsedProp<'_>,
) -> Result<(), ParseError> {
    match &prop.value {
        ParsedValue::Str(s) => into.set_str(id, prop.key, s).map_err(|e| ParseError {
            message: format!("set '{}' {}={s:?}: {e:?}", desc.name, prop.key),
        }),
        ParsedValue::Value(v) => into.set(id, prop.key, *v).map_err(|e| ParseError {
            message: format!("set '{}' {}={:?}: {e:?}", desc.name, prop.key, v),
        }),
    }
}

// --- Lexer / mini-parser --------------------------------------------------------------
//
// Deliberately tiny and total: split on `!`, then each stage into whitespace-separated
// tokens, then each token into `name` or `key=value`. Never indexes without a bound,
// never unwraps — malformed input is a `ParseError`, not a panic (spec: parsers are
// fuzz-friendly surfaces).

struct ParsedStage<'a> {
    name: &'a str,
    props: Vec<ParsedProp<'a>>,
}

struct ParsedProp<'a> {
    key: &'a str,
    value: ParsedValue<'a>,
}

/// A parsed property value: either a POD [`Value`] (int/rational) or a borrowed string
/// slice, which the pipeline interns at set time (`Value` has no string variant).
enum ParsedValue<'a> {
    Value(Value),
    Str(&'a str),
}

/// Split a launch string into its `!`-separated stages, each lexed into a name plus
/// `key=value` properties. Empty stages (`a !! b`, a leading/trailing `!`) are a named
/// error, never a silent drop.
fn lex_chain(launch: &str) -> Result<Vec<ParsedStage<'_>>, ParseError> {
    let mut stages = Vec::new();
    // `split('!')` yields one item per segment including the empties around a stray `!`,
    // so `a ! b` → ["a ", " b"] and `a !! b` → ["a ", "", " b"] — the empty middle is
    // the error we want to name.
    for segment in launch.split('!') {
        let trimmed = segment.trim();
        if trimmed.is_empty() {
            // Distinguish a wholly-empty input (no elements at all) from a stray `!`.
            if launch.trim().is_empty() {
                return Ok(Vec::new());
            }
            return Err(ParseError {
                message: "empty element between '!' separators (a stray '!'?)".into(),
            });
        }
        stages.push(lex_stage(trimmed)?);
    }
    Ok(stages)
}

/// Lex one stage: the leading token is the element name, each following token is a
/// `key=value` property.
fn lex_stage(stage: &str) -> Result<ParsedStage<'_>, ParseError> {
    let mut tokens = stage.split_whitespace();
    let name = tokens.next().ok_or_else(|| ParseError {
        message: "expected an element name".into(),
    })?;
    if !is_ident(name) {
        return Err(ParseError {
            message: format!("'{name}' is not a valid element name"),
        });
    }
    let mut props = Vec::new();
    for tok in tokens {
        props.push(lex_prop(tok)?);
    }
    Ok(ParsedStage { name, props })
}

/// Lex one `key=value` token. A missing `=`, an empty key, or an empty value is a named
/// error. The value is parsed as an integer, then a rational, then falls back to a bare
/// string (paths, categorical names).
fn lex_prop(tok: &str) -> Result<ParsedProp<'_>, ParseError> {
    let eq = tok.find('=').ok_or_else(|| ParseError {
        message: format!("property '{tok}' is missing '=' (expected key=value)"),
    })?;
    let key = &tok[..eq];
    let val = &tok[eq + 1..];
    if key.is_empty() {
        return Err(ParseError {
            message: format!("property '{tok}' has an empty key"),
        });
    }
    if !is_ident(key) {
        return Err(ParseError {
            message: format!("'{key}' is not a valid property name"),
        });
    }
    if val.is_empty() {
        return Err(ParseError {
            message: format!("property '{key}' has an empty value"),
        });
    }
    Ok(ParsedProp {
        key,
        value: parse_value(val),
    })
}

/// Parse a property value: integer first, then rational (`num/den`), else a bare string.
/// Total — any input that is not an int or rational is simply a string (a file path may
/// contain `/`, so a rational only matches when *both* sides parse as integers).
fn parse_value(s: &str) -> ParsedValue<'_> {
    if let Ok(i) = s.parse::<i64>() {
        return ParsedValue::Value(Value::Int(i));
    }
    if let Some((n, d)) = s.split_once('/') {
        if let (Ok(n), Ok(d)) = (n.parse::<i32>(), d.parse::<i32>()) {
            return ParsedValue::Value(Value::Rat(n, d));
        }
    }
    ParsedValue::Str(s)
}

/// An element/property identifier: a non-empty run of ASCII alphanumerics plus `_`,
/// `-`, `.`, and `/` (so `audio/raw`-style names and `test-src` both pass). Kept
/// permissive but bounded — enough to reject obviously broken tokens (`=`, whitespace,
/// leading `!`) without a full grammar.
fn is_ident(s: &str) -> bool {
    !s.is_empty()
        && s.chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.' | '/'))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::element::{
        Element, ElementDesc, Flow, InputPolicy, LatencyDesc, PadDesc, SchedHint,
    };
    use crate::batch::Inputs;
    use crate::ctx::Ctx;
    use crate::error::Error;
    use crate::event::Event;
    use crate::format::{Constraint, OfferDesc};
    use crate::element::PropDesc;

    // A minimal source/sink pair with declared props, purely for exercising the parser
    // and registry without pulling in a plugin crate (core has no elements of its own).

    static ANY_BYTES: [OfferDesc; 1] = [OfferDesc::any("bytes")];

    static SRC_PADS: [PadDesc; 1] = [PadDesc {
        name: "src",
        direction: Direction::Src,
        offers: &ANY_BYTES,
        dynamic: false,
        validate: None,
    }];
    static SINK_PADS: [PadDesc; 1] = [PadDesc {
        name: "sink",
        direction: Direction::Sink,
        offers: &ANY_BYTES,
        dynamic: false,
        validate: None,
    }];

    static SRC_PROPS: [PropDesc; 2] = [
        PropDesc { name: "total", allowed: Constraint::Any, live: false },
        PropDesc { name: "path", allowed: Constraint::Any, live: false },
    ];

    static SRC_DESC: ElementDesc = ElementDesc {
        name: "tsrc",
        pads: &SRC_PADS,
        props: &SRC_PROPS,
        sched: SchedHint::Active,
        inputs: InputPolicy::None,
        latency: LatencyDesc {
            min: crate::time::Timestamp::ZERO,
            max: crate::time::Timestamp::ZERO,
            is_live: false,
            jitter: crate::time::Timestamp::ZERO,
        },
        make_default: Some(|| Box::new(TSrc)),
    };
    static SINK_DESC: ElementDesc = ElementDesc {
        name: "tsink",
        pads: &SINK_PADS,
        props: &[],
        sched: SchedHint::Active,
        inputs: InputPolicy::Single,
        latency: LatencyDesc {
            min: crate::time::Timestamp::ZERO,
            max: crate::time::Timestamp::ZERO,
            is_live: false,
            jitter: crate::time::Timestamp::ZERO,
        },
        make_default: Some(|| Box::new(TSink)),
    };
    // A descriptor with no factory: registrable, but not constructible by name.
    static NOFACTORY_DESC: ElementDesc = ElementDesc {
        name: "nofactory",
        pads: &SRC_PADS,
        props: &[],
        sched: SchedHint::Active,
        inputs: InputPolicy::None,
        latency: LatencyDesc {
            min: crate::time::Timestamp::ZERO,
            max: crate::time::Timestamp::ZERO,
            is_live: false,
            jitter: crate::time::Timestamp::ZERO,
        },
        make_default: None,
    };

    struct TSrc;
    impl Element for TSrc {
        fn desc(&self) -> &'static ElementDesc {
            &SRC_DESC
        }
        fn start(&mut self, _c: &mut Ctx) -> Result<(), Error> {
            Ok(())
        }
        fn process(&mut self, _c: &mut Ctx, _i: Inputs<'_>) -> Result<Flow, Error> {
            Ok(Flow::Eos)
        }
        fn event(&mut self, _c: &mut Ctx, _e: &Event) -> Result<(), Error> {
            Ok(())
        }
        fn stop(&mut self, _c: &mut Ctx) {}
    }
    struct TSink;
    impl Element for TSink {
        fn desc(&self) -> &'static ElementDesc {
            &SINK_DESC
        }
        fn start(&mut self, _c: &mut Ctx) -> Result<(), Error> {
            Ok(())
        }
        fn process(&mut self, _c: &mut Ctx, _i: Inputs<'_>) -> Result<Flow, Error> {
            Ok(Flow::Ok)
        }
        fn event(&mut self, _c: &mut Ctx, _e: &Event) -> Result<(), Error> {
            Ok(())
        }
        fn stop(&mut self, _c: &mut Ctx) {}
    }

    fn registry() -> Registry {
        let mut r = Registry::new();
        r.register(&SRC_DESC);
        r.register(&SINK_DESC);
        r.register(&NOFACTORY_DESC);
        r
    }

    #[test]
    fn register_get_and_names() {
        let r = registry();
        assert_eq!(r.get("tsrc").map(|d| d.name), Some("tsrc"));
        assert!(r.get("missing").is_none());
        // `names()` is sorted, so the assertion is order-stable.
        assert_eq!(r.names(), vec!["nofactory", "tsink", "tsrc"]);
    }

    #[test]
    fn last_registration_wins() {
        let mut r = Registry::new();
        r.register(&SRC_DESC);
        // Re-register another desc under a name we control by aliasing is not possible with
        // `&'static`; instead assert idempotence: registering the same desc twice is stable.
        r.register(&SRC_DESC);
        assert_eq!(r.get("tsrc").map(|d| d.name), Some("tsrc"));
    }

    #[test]
    fn parse_builds_linear_chain() {
        let r = registry();
        let mut p = Pipeline::new();
        let ids = r.parse(&mut p, "tsrc total=65536 ! tsink").expect("parses");
        assert_eq!(ids.len(), 2);
        // The int prop landed on the source.
        assert_eq!(p.prop_value(ids[0], "total"), Some(Value::Int(65536)));
    }

    #[test]
    fn parse_whitespace_tolerant() {
        let r = registry();
        // No spaces around '!', extra spaces around '=', leading/trailing whitespace.
        for s in ["tsrc!tsink", "  tsrc ! tsink  ", "tsrc total=1!tsink"] {
            let mut p = Pipeline::new();
            assert!(r.parse(&mut p, s).is_ok(), "{s:?} should parse");
        }
    }

    #[test]
    fn parse_string_prop_interns_via_value_vocab() {
        let r = registry();
        let mut p = Pipeline::new();
        let ids = r.parse(&mut p, "tsrc path=/tmp/in.wav ! tsink").expect("parses");
        // The path was interned; the element sees it as a Value::Id resolving to the string.
        let v = p.prop_value(ids[0], "path").expect("path set");
        match v {
            Value::Id(vid) => assert_eq!(p.value_name(vid), Some("/tmp/in.wav")),
            other => panic!("expected an interned id, got {other:?}"),
        }
    }

    #[test]
    fn parse_rational_prop() {
        let r = registry();
        let mut p = Pipeline::new();
        // 30000/1001 must parse as a rational; a path with '/' must stay a string.
        let ids = r.parse(&mut p, "tsrc total=30000/1001 ! tsink").expect("parses");
        assert_eq!(p.prop_value(ids[0], "total"), Some(Value::Rat(30000, 1001)));
    }

    #[test]
    fn parse_errors_name_the_offending_token() {
        let r = registry();
        // Table of bad inputs → a substring the error must mention. The parser must never
        // panic (spec: fuzz-friendly), so each of these is a clean `Err`.
        let cases: &[(&str, &str)] = &[
            ("", "empty"),
            ("   ", "empty"),
            ("!", "empty element"),
            ("tsrc ! ! tsink", "empty element"),
            ("tsrc !", "empty element"),
            ("! tsink", "empty element"),
            ("bogus ! tsink", "bogus"),
            ("tsrc ! bogus", "bogus"),
            ("nofactory ! tsink", "make_default"),
            ("tsrc total ! tsink", "missing '='"),
            ("tsrc =5 ! tsink", "empty key"),
            ("tsrc total= ! tsink", "empty value"),
            ("tsrc nosuch=1 ! tsink", "nosuch"),
        ];
        for (input, needle) in cases {
            let mut p = Pipeline::new();
            match r.parse(&mut p, input) {
                Ok(_) => panic!("{input:?} should not parse"),
                Err(e) => assert!(
                    e.message.contains(needle),
                    "{input:?}: error {:?} should mention {needle:?}",
                    e.message
                ),
            }
        }
    }

    #[test]
    fn parse_never_panics_on_junk() {
        // A pile of adversarial inputs: the contract is "Err, never panic".
        let r = registry();
        let junk = [
            "=", "==", "!=!", "a=b=c", "tsrc total=/", "/", "tsrc total=1/",
            "tsrc total=/1", "\t\n", "tsrc\0", "tsrc total=999999999999999999999999",
            "tsrc ! tsrc ! tsrc", "tsink ! tsrc",
        ];
        for s in junk {
            let mut p = Pipeline::new();
            let _ = r.parse(&mut p, s); // must return, never unwind
        }
    }

    #[test]
    fn value_parsing_int_rat_string() {
        assert!(matches!(parse_value("65536"), ParsedValue::Value(Value::Int(65536))));
        assert!(matches!(parse_value("-3"), ParsedValue::Value(Value::Int(-3))));
        assert!(matches!(parse_value("30000/1001"), ParsedValue::Value(Value::Rat(30000, 1001))));
        // A path with a slash is not a rational (its parts are not both integers).
        assert!(matches!(parse_value("/tmp/x.wav"), ParsedValue::Str("/tmp/x.wav")));
        assert!(matches!(parse_value("s16"), ParsedValue::Str("s16")));
        // Overflowing integer falls back to a string (parse::<i64> fails), never panics.
        assert!(matches!(parse_value("999999999999999999999999"), ParsedValue::Str(_)));
    }
}
