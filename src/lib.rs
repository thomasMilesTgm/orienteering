//! # Orienteering
//!
//! Procedural terrain for orienteering: generated landscapes, their contour maps and biomes.
//!
//! Elevation is a scalar field `h(x, y)`, i.e. gravitational potential up to a constant. The
//! terrain is built by composing simple fields, and a map is drawn by extracting its level sets
//! (contours) at regular height intervals. Because the terrain is a scalar field, its gradient is
//! conservative by construction and every contour either closes or leaves the mapped area.
//!
//! ## Crates
//!
//! - [`math`]: rectangles, [`math::Dual`] numbers that carry exact gradients, and a reproducible
//!   random number generator.
//! - [`field`]: [`field::Field`], a serializable expression tree of terrain primitives and
//!   combinators, evaluated with exact gradients.
//! - [`terrain`]: the [`terrain::World`], generated coarse to fine and deterministically, in any
//!   order: 8 km provinces set the broad shape and a smooth ruggedness field, then 1 km chunks add
//!   landforms and progressively finer detail as quadtrees, each level chosen in response to the
//!   terrain generated so far (mountains grow more mountainous, slopes are cut by branching
//!   channels, smooth lowlands fill in and flatten out).
//! - [`map`]: sampled height grids, contours, biomes and rasterized map images.
//! - [`flat`]: fields flattened into arrays for evaluation on the GPU, with an f32 reference
//!   evaluator.
//!
//! Every cell's and quadtree node's field is multiplied by a [`field::Window`] on its bounds: tanh
//! transitions centered on each edge. Windows of rectangles that tile the plane with a common blend
//! width sum to exactly 1, so neighboring cells blend into each other without seams.

pub use orienteering_internal::*;
