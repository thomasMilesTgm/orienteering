//! # Topography
//!
//! Defines the topography of the world, i.e. the shape of terrain, along with biomes, structures,
//! features, etc.

pub mod vector_field;
pub mod vector_function;

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct WorldMap {}
