//! Minimal example of rendering an [`orienteering::topography::WorldMap`] in [`bevy`].

use bevy::{
    DefaultPlugins,
    app::{App, Plugin, Startup},
    ecs::{resource::Resource, system::ResMut},
};
use nalgebra::Point2;
use orienteering::{
    proc_gen::*,
    topography::{AreaOF, WorldMap},
    utils::of32,
};

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
    let mut map = WorldMap::new(seed);

    let x0 = of32::from(-100.);
    let x1 = of32::from(100.);

    map.generate_area(AreaOF {
        min: Point2::new(x0, x0),
        max: Point2::new(x1, x1),
    });

    let p0 = of32::from(10.);
    map.generate_contour(Point2::new(p0, p0), 20_f32.into(), 0_f32.into());

    world.map = Some(map);
}
