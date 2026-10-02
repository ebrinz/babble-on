//! Shared contracts between the babble-on app (Rust) and the experiment
//! harness (Python, `harness/babble_harness/{noise,recording}.py`).
//!
//! Everything here has a line-for-line Python mirror and is pinned by the
//! committed vectors in `docs/contract/noise_vectors.json`, which both test
//! suites check. Change one side only together with the other.

pub mod noise;
pub mod recording;
pub mod seed;

pub use noise::{bytes_to_uniforms, uniform_to_index};
pub use recording::{RecordingHeader, RecordingReader, RecordingWriter, MAGIC};
pub use seed::{SeedBundle, SeedTag};
