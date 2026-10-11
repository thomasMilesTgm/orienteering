//! Maps of a height field: sampled grids, contour lines, biomes, and rasterized map images.
//!
//! - [`grid`]: [`HeightGrid`], heights (and optionally exact gradients) sampled in parallel.
//! - [`contour`]: Marching squares over a [`HeightGrid`], with optional Newton refinement of
//!   vertices onto the exact level set using the analytic gradient.
//! - [`biome`]: Classification into [`Biome`]s by altitude and slope, cleaned up into coherent
//!   regions. Biomes follow from the terrain's shape rather than being generated separately: water
//!   below the water line, rock where it is too steep for vegetation, alpine above the tree line,
//!   marsh and meadow on flat ground, forest elsewhere.
//! - [`raster`]: Map images loosely following orienteering map conventions.

pub mod biome;
pub mod contour;
pub mod grid;
pub mod raster;

pub use biome::{Band, Biome, BiomeMap, BiomeRegion, BiomeRule, BiomeRules, Regions};
pub use contour::{
    CONTOUR_LADDER, Contour, ContourLabel, ContourScale, LabelSpacing, contour_rank, contours,
    levels, place_labels, refine_all,
};
pub use grid::HeightGrid;
