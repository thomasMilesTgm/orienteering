//! # Terrain
//!
//! Elevation is a scalar field `h(x, y)`, i.e. gravitational potential up to a constant. The
//! terrain is built by composing simple fields, and a map is drawn by extracting its level sets
//! (contours) at regular height intervals. Because the terrain is a scalar field, its gradient is
//! conservative by construction and every contour either closes or leaves the mapped area.
//!
//! ## Layers
//!
//! - [`scalar`]: Fields are written once, generic over a [`Scalar`] number type. Evaluating with
//!   `f64` gives heights; evaluating with [`Dual`] numbers gives exact gradients for free.
//! - [`field`]: [`Field`], a serializable expression tree of terrain primitives (bumps, ridges,
//!   waves, planes) and combinators (sums, products, scaling, and smooth rectangular
//!   [`Window`]s).
//! - [`generate`]: The [`World`] is tiled with chunks generated lazily and deterministically from
//!   `hash(seed, chunk index)`, so they can be produced in any order.
//! - [`contour`]: Marching squares over a sampled [`HeightGrid`], with optional Newton refinement
//!   of vertices onto the exact level set using the analytic gradient.
//!
//! ## Chunks and Windows
//!
//! Each chunk's field is multiplied by a [`Window`] on its bounds: tanh transitions centered on
//! each edge. Windows of rectangles that tile the plane with a common blend width sum to exactly
//! 1, so neighboring chunks blend into each other without seams, dips or bulges. A window is
//! negligible a few blend widths past its edge, so evaluating a point only involves its own chunk
//! and its eight neighbors.
//!
//! ## Sub-chunking
//!
//! Each chunk is a quadtree. Every node adds a few features sized relative to the node itself and
//! scaled by `amplitude_falloff^depth`, so deeper nodes add shorter wavelengths with smaller
//! relief. This is like octaves of noise, but localized and with no global repetition. Each node
//! is subdivided with probability equal to the chunk's `ruggedness`, so the subdivision depth
//! sets the character of the terrain (a proxy for biome):
//!
//! - Shallow: smooth, gentle relief (plains, desert, tundra).
//! - Moderate: rolling, varied relief (grassland, forest, hills).
//! - Deep: lots of small, sharp features (mountains, rocky ground).

pub mod contour;
pub mod field;
pub mod generate;
pub mod rng;
pub mod scalar;

pub use contour::{Contour, HeightGrid, contours, levels};
pub use field::{Aabb, Field, Surface, Window};
pub use generate::{Chunk, ChunkIndex, Region, TerrainParams, World};
pub use scalar::{Dual, Scalar};
