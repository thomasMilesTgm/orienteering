//! Minimal example of rendering an [`orienteering::topography::WorldMap`] in [`bevy`].

use bevy::{
    DefaultPlugins,
    app::{App, Plugin, Startup, Update},
    color::Color,
    ecs::{
        resource::Resource,
        system::{Res, ResMut},
    },
    gizmos::gizmos::Gizmos,
    math::{
        Vec2,
        cubic_splines::{CubicCurve, CubicGenerator, CubicHermite},
        vec2,
    },
};
use nalgebra::Point2;
use orienteering::{
    proc_gen::*,
    topography::{AreaOF, WorldMap},
};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(ProcGenPlugins)
        .add_systems(Update, draw_contours)
        .run();
}

#[derive(Resource, Default, Debug)]
pub struct WorldResource {
    pub seed: MapSeed,
    pub map: Option<WorldMap>,
    pub contour_splines: Vec<CubicCurve<Vec2>>,
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

    let x0 = -100.;
    let x1 = 100.;

    map.generate_area(AreaOF {
        min: Point2::new(x0.into(), x0.into()),
        max: Point2::new(x1.into(), x1.into()),
    });

    let p0 = 2.;
    map.generate_contour(Point2::new(p0, p0), 20., 0.);

    world.contour_splines = map
        .contours
        .iter()
        .flat_map(|c| {
            let points = c.points.iter().map(|p| vec2(p.x, p.y)).collect::<Vec<_>>();
            let tangents = c
                .tangents
                .iter()
                .map(|p| vec2(p.x, p.y))
                .collect::<Vec<_>>();

            let spline = CubicHermite::new(points, tangents);
            spline.to_curve()
        })
        .collect::<Vec<_>>();

    world.map = Some(map);
    dbg!(&world.as_ref());
}

fn draw_contours(world: Res<WorldResource>, mut gizmos: Gizmos) {
    world.contour_splines.iter().for_each(|s| {
        let resolution = 100 * s.segments().len();
        gizmos.linestrip(
            s.iter_positions(resolution).map(|pt| pt.extend(0.0)),
            Color::srgb(1.0, 1.0, 1.0),
        );
    });
}
