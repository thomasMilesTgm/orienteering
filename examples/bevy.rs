//! Minimal example of rendering an [`orienteering::topography::WorldMap`] in [`bevy`].

use orienteering::{proc_gen::*, topography::WorldMap};

fn main() {
    let seed = MapSeed::default();
    let _map = WorldMap::new(seed);
}
