//! What the application hands the engine: a file to play, plus the two per-track knobs the
//! canonical chain can carry (ReplayGain and time-stretch).
//!
//! # Why `Growing` is a constructor and not a plain variant
//!
//! The obvious shape is `TrackSource::Growing { path, total_hint }` — data only, inert until
//! the engine builds it. It cannot be built that way, for the reason
//! [`SourceSpec::growing`](pf_play::SourceSpec::growing) documents: `GrowingFileSrc::new`
//! **mints** the [`FrontierHandle`], so the handle does not exist until the element does — and
//! the caller needs the handle *first*, because the download must already be running before the
//! open can succeed (the open waits for the typefind prefix to become readable).
//!
//! So [`Track::growing`] constructs the source eagerly and hands the handle back:
//!
//! ```ignore
//! let (track, frontier) = Track::growing("/tmp/ep.mp3", Some(content_length));
//! std::thread::spawn(move || { /* download; frontier.advance(n); … frontier.finish(len) */ });
//! engine.play_now(track);
//! ```
//!
//! Keep your own copy of the handle for the life of the download. `growingfilesrc` treats "every
//! handle dropped without `finish()`/`abort()`" as an abandoned download and winds the stream
//! down — which is the behaviour that turns a crashed downloader into a track that ends early
//! and a queue that advances, rather than a pipeline parked on the frontier forever.

// Control-plane module: paths, labels and one boxed source element per track, all built once
// when the application asks for a track. Nothing here runs on a media path or inside
// `process()` — the documented `clippy.toml` exception.
#![allow(clippy::disallowed_methods)]

use std::path::{Path, PathBuf};

use pf_play::SourceSpec;
use profluens_elements::io::FrontierHandle;

/// A file a downloader is still appending to, together with the source element built over it.
///
/// Opaque by design: it owns a `SourceSpec` whose element can be taken exactly once, so a
/// `GrowingSource` is single-use and cannot be cloned into two pipelines.
pub struct GrowingSource {
    spec: SourceSpec,
}

/// Where a track's bytes come from.
pub enum TrackSource {
    /// A complete file on disk.
    File(PathBuf),
    /// A file still being written. Build one with [`Track::growing`], which returns the
    /// [`FrontierHandle`] the writer publishes progress through.
    Growing(GrowingSource),
}

/// One item of the engine's queue: a source plus the per-track chain stages it wants.
///
/// Not `Clone` — it owns a source element, and a source element belongs to exactly one
/// pipeline.
pub struct Track {
    pub source: TrackSource,
    /// Per-track ReplayGain in **decibels**, applied by an `audiogain` stage inside this
    /// track's chain (so it dies with the track, which is the scope ReplayGain wants).
    /// `Some(0.0)` inserts the stage at unity — ask for it if you intend to change the
    /// correction live through [`Engine::set_track_gain_db`](crate::Engine::set_track_gain_db).
    pub gain_db: Option<f32>,
    /// Playback rate for an `audiostretch` stage (1.5 = 1.5× faster, pitch preserved). When
    /// present, [`Engine::position`](crate::Engine::position) reports **source** time from the
    /// stretcher rather than device time — which is what a progress bar over a sped-up
    /// audiobook must show.
    pub stretch: Option<f32>,
}

impl Track {
    /// A complete file on disk.
    pub fn file(path: impl AsRef<Path>) -> Track {
        Track {
            source: TrackSource::File(path.as_ref().to_path_buf()),
            gain_db: None,
            stretch: None,
        }
    }

    /// A file a separate writer is still appending to, plus the [`FrontierHandle`] that writer
    /// publishes progress through. See the module docs: start the download *before* handing the
    /// track to the engine.
    ///
    /// `total_hint` is the final size when it is known out of band (a podcast's HTTP
    /// `Content-Length`). Supply it whenever you can — it is the denominator the seek index is
    /// built against, and the alternative (the current frontier) silently changes meaning as
    /// bytes arrive.
    ///
    /// A path that is not valid UTF-8 yields no handle: the underlying open takes a `&str`, so
    /// such a path cannot be played at all and saying so here is better than minting a handle
    /// for a download that can never be opened.
    pub fn growing(path: impl AsRef<Path>, total_hint: Option<u64>) -> Option<(Track, FrontierHandle)> {
        let path = path.as_ref().to_str()?;
        let (spec, frontier) = SourceSpec::growing(path, total_hint);
        let track = Track {
            source: TrackSource::Growing(GrowingSource { spec }),
            gain_db: None,
            stretch: None,
        };
        Some((track, frontier))
    }

    /// Add a ReplayGain correction, in decibels.
    pub fn with_gain_db(mut self, db: f32) -> Track {
        self.gain_db = Some(db);
        self
    }

    /// Add a pitch-preserving time-stretch at `rate`.
    pub fn with_stretch(mut self, rate: f32) -> Track {
        self.stretch = Some(rate);
        self
    }

    /// A short human label for this track, used in error messages.
    pub(crate) fn label(&self) -> String {
        match &self.source {
            TrackSource::File(p) => p.display().to_string(),
            TrackSource::Growing(g) => g.spec.path_str().to_string(),
        }
    }
}

impl TrackSource {
    /// The `pf-play` source spec for this track.
    ///
    /// The only failure is a path that is not valid UTF-8. `pf-play`'s open path takes a
    /// `&str` all the way down (probe, head walk, seek index and the source element all read
    /// the same path), so a lossy conversion would open a *different* file — or, more likely,
    /// none. Reporting it is the honest answer; it reaches the app as
    /// [`EngineEvent::Error`](crate::EngineEvent::Error).
    pub(crate) fn into_spec(self) -> Result<SourceSpec, String> {
        match self {
            TrackSource::File(p) => match p.to_str() {
                Some(s) => Ok(SourceSpec::path(s)),
                None => Err(format!("path is not valid UTF-8: {}", p.display())),
            },
            TrackSource::Growing(g) => Ok(g.spec),
        }
    }
}
