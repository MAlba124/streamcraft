//! Core's own logging — no `log`/`tracing` dependency (spec: Debuggability).
//! Per-element targets, runtime-adjustable levels, `STREAMCRAFT_DEBUG=element:level`
//! env syntax like `GST_DEBUG`. Macros branch on a static level before formatting
//! anything, so logging is free when off. TODO(step 7).

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Level {
    Error,
    Warn,
    Info,
    Debug,
    Trace,
}
