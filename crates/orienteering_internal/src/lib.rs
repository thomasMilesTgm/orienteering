//! Re-exports the orienteering crates as modules. Use them through the `orienteering` facade.

pub use orienteering_field as field;
pub use orienteering_flat as flat;
pub use orienteering_map as map;
pub use orienteering_math as math;
pub use orienteering_terrain as terrain;

/// The most commonly used types. (`World` and `Window` are left out, as they would clash with
/// game engines' preludes; use [`terrain::World`] and [`field::Window`].)
pub mod prelude {
    pub use crate::field::{Field, Surface};
    pub use crate::map::{Biome, BiomeMap, BiomeRules, Contour, HeightGrid};
    pub use crate::math::{Aabb, Dual, Scalar};
    pub use crate::terrain::{CellIndex, Chunk, Region, TerrainParams};
}
