//! Minimal example of rendering an [`orienteering::topography::WorldMap`] in [`bevy`].

use orienteering::{
    proc_gen::*,
    topography::{WorldMap, vector_field::CircularField},
};

fn main() {
    let seed = MapSeed::default();
    let map = WorldMap::new(seed);
}
