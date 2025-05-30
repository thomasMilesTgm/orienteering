//! Minimal example of rendering an [`orienteering::topography::WorldMap`] in [`bevy`].

use bevy::{
    DefaultPlugins,
    app::{App, Plugin, Startup, Update},
    color::Color,
    core_pipeline::core_2d::Camera2d,
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
    prelude::*,
};
use nalgebra::Point2;
use orienteering::{
    proc_gen::*,
    topography::{AreaOF, WorldMap, vector_field::*},
};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(ProcGenPlugins)
        .add_systems(Update, (draw_contours, draw_tangents))
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
        app.init_resource::<WorldResource>().add_systems(
            Startup,
            (
                setup, init_world,
                //..
            )
                .chain(),
        );
    }
}

fn setup(mut commands: Commands) {
    commands.spawn(Camera2d);
}

fn init_world(mut world: ResMut<WorldResource>) {
    let seed = world.seed.clone();
    let mut map = WorldMap::new(seed);

    let x0 = -500.;
    let x1 = 500.;

    let field = HillyBowlField::from_rng(map.rng());
    // let field = SaddleField::from_rng(map.rng());
    // let field = SinkField::from_rng(map.rng());
    // let field = CircularField::from_rng(map.rng());

    let area = AreaOF {
        min: Point2::new(x0.into(), x0.into()),
        max: Point2::new(x1.into(), x1.into()),
    };

    map.generate_area(area, field);

    for i in 1..=5 {
        let x = i as f32 * (x1 - x0) / 4.;
        for j in 1..=10 {
            let y = j as f32 * (x1 - x0) / 4.;
            map.generate_contour(Point2::new(x0 + x, x0 + y), 2000., 0.);
        }
    }

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
    // dbg!(&world.map.as_ref());
}

fn draw_tangents(world: Res<WorldResource>, mut gizmos: Gizmos) {
    world
        .map
        .as_ref()
        .unwrap()
        .contours
        .iter()
        .flat_map(|c| c.points.first().zip(c.points.last().zip(c.tangents.last())))
        .for_each(|(p0, (p, t))| {
            let p0 = vec2(p0.x, p0.y);
            let point = vec2(p.x, p.y);
            let tangent = 20. * vec2(t.x, t.y);
            gizmos.circle_2d(p0, 1.0, Color::srgb(0.0, 1.0, 0.0));
            gizmos.circle_2d(point, 1.0, Color::srgb(1.0, 0.0, 0.0));
            gizmos.arrow_2d(point, point + tangent, Color::srgb(1., 0., 0.));
        });
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
