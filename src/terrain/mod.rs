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
//! - [`grid`]: [`HeightGrid`], heights (and optionally exact gradients) sampled in parallel.
//! - [`contour`]: Marching squares over a [`HeightGrid`], with optional Newton refinement of
//!   vertices onto the exact level set using the analytic gradient.
//! - [`biome`]: Classification of the terrain into [`Biome`]s by altitude and slope, cleaned up
//!   into coherent regions.
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
//! sets the character of the terrain: shallow chunks have smooth, gentle relief, deep chunks have
//! lots of small, sharp features.
//!
//! ## Biomes
//!
//! Biomes follow from the generated shape rather than being generated separately: an ordered list
//! of [`BiomeRules`] maps each point's altitude and slope to a [`Biome`] (water below the water
//! line, rock where it is too steep for vegetation, alpine above the tree line, marsh and meadow on
//! flat ground, forest elsewhere). Rugged chunks are steeper, so they naturally produce more rock.

pub mod biome;
pub mod contour;
pub mod field;
pub mod generate;
pub mod grid;
pub mod rng;
pub mod scalar;

pub use biome::{Band, Biome, BiomeMap, BiomeRegion, BiomeRule, BiomeRules, Regions};
pub use contour::{Contour, contours, levels, refine_all};
pub use field::{Aabb, Field, Surface, Window};
pub use generate::{Chunk, ChunkIndex, Region, TerrainParams, World};
pub use grid::HeightGrid;
pub use scalar::{Dual, Scalar};
