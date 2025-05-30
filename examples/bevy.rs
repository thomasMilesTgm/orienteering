//! Minimal example of rendering an [`orienteering::topography::WorldMap`] in [`bevy`].

use bevy::{
    DefaultPlugins,
    app::{App, Plugin, Startup},
    ecs::{
        resource::Resource,
        system::{Commands, ResMut},
    },
};
use orienteering::{proc_gen::*, topography::WorldMap};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(ProcGenPlugins)
        .run();
}

#[derive(Resource, Default)]
pub struct WorldResource {
    pub seed: MapSeed,
    pub map: Option<WorldMap>,
}

pub struct ProcGenPlugins;

impl Plugin for ProcGenPlugins {
    fn build(&self, app: &mut App) {
        app.init_resource::<WorldResource>()
            .add_systems(Startup, init_world);
    }
}

fn init_world(mut world: ResMut<WorldResource>) {
    let seed = world.seed.clone();
    world.map = Some(WorldMap::new(seed));
}
