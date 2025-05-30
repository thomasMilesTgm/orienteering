//! Minimal example of rendering an [`orienteering::topography::WorldMap`] in [`bevy`].

use bevy::{
    DefaultPlugins,
    app::{App, Startup},
    ecs::{resource::Resource, system::Commands},
};
use orienteering::{proc_gen::*, topography::WorldMap};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_systems(Startup, setup)
        .init_resource::<WorldResource>()
        .run();
}

#[derive(Resource, Default)]
pub struct WorldResource {
    pub seed: MapSeed,
    pub map: Option<WorldMap>,
}

fn setup(mut commands: Commands) {
    let seed = MapSeed::default();
    let _map = WorldMap::new(seed);
}
