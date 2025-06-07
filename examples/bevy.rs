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
    topography::{AreaOF, WorldMap, vector_field::Field},
};

const X0: f32 = -500.0;
const X1: f32 = 500.0;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(ProcGenPlugins)
        .add_systems(Update, draw_contours)
        // .add_systems(Update, draw_tangents)
        .add_systems(Update, draw_field)
        .add_systems(Update, draw_divergence)
        // .add_systems(Update, draw_curl)
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
    world.seed = MapSeed::from_string("".to_string());
    let seed = world.seed.clone();

    let mut map = WorldMap::new(seed);

    let area = AreaOF {
        min: Point2::new(X0.into(), X0.into()),
        max: Point2::new(X1.into(), X1.into()),
    };

    map.generate_island(area);

    // let field = HillyBowlField::from_rng(map.rng());
    // let field = SaddleField::from_rng(map.rng());
    // let field = SinkField::from_rng(map.rng());
    // let field = CircularField::from_rng(map.rng());

    // map.generate_area(area, field);

    world.contour_splines = map
        .contours
        .iter()
        .flat_map(|c| {
            let (points, tangents): (Vec<_>, Vec<_>) = c
                .line
                .points
                .iter()
                .map(|p| (vec2(p.xy.x, p.xy.y), vec2(p.v_xy.x, p.v_xy.y)))
                .unzip();

            let spline = CubicHermite::new(points, tangents);
            spline.to_curve()
        })
        .collect::<Vec<_>>();

    // dbg!(&map.field);
    world.map = Some(map);
}

pub fn draw_divergence(world: Res<WorldResource>, mut gizmos: Gizmos) {
    for i in 1..=10 {
        let x = i as f32 * (X1 - X0) / 10.;
        for j in 1..=10 {
            let y = j as f32 * (X1 - X0) / 10.;
            let pt = Point2::new(X0 + x, X0 + y);
            let div = world.map.as_ref().unwrap().field.divergence(pt, 1.);
            let isometry = Isometry2d::from_translation(vec2(pt.x, pt.y));

            if div.is_finite() {
                let radius = div.abs().clamp(2., 20.);

                let color = if div.is_sign_negative() {
                    Color::srgb(radius / 25., 0., 0.)
                } else {
                    Color::srgb(0., 0., radius / 25.)
                };

                gizmos.cross_2d(isometry, radius, color);
            } else {
                let color = if div.is_sign_negative() {
                    Color::srgb(1., 0., 0.)
                } else {
                    Color::srgb(0., 0., 1.)
                };
                gizmos.cross_2d(isometry, 20., color);
            }
        }
    }
}
pub fn draw_curl(world: Res<WorldResource>, mut gizmos: Gizmos) {
    for i in 1..=10 {
        let x = i as f32 * (X1 - X0) / 10.;
        for j in 1..=10 {
            let y = j as f32 * (X1 - X0) / 10.;
            let pt = Point2::new(X0 + x, X0 + y);
            let curl = world.map.as_ref().unwrap().field.curl(pt, 0.01);
            let isometry = Isometry2d::from_translation(vec2(pt.x, pt.y));

            let radius = (curl.abs() * 10000.).min(20.);

            let color = if curl.is_sign_negative() {
                Color::srgb(radius / 25., 0., 0.)
            } else {
                Color::srgb(0., 0., radius / 25.)
            };

            gizmos.circle_2d(isometry, radius, color);
        }
    }
}

pub fn draw_field(world: Res<WorldResource>, mut gizmos: Gizmos) {
    for i in 1..=10 {
        let x = i as f32 * (X1 - X0) / 10.;
        for j in 1..=10 {
            let y = j as f32 * (X1 - X0) / 10.;
            let pt = Point2::new(X0 + x, X0 + y);
            let v = world.map.as_ref().unwrap().field.field_vector(pt);
            let tangent = 10. * vec2(v.x, v.y);
            let pt = vec2(pt.x, pt.y);
            gizmos.arrow_2d(pt, pt + tangent, Color::srgb(1., 0., 0.));
        }
    }
}

pub fn draw_tangents(world: Res<WorldResource>, mut gizmos: Gizmos) {
    world
        .map
        .as_ref()
        .unwrap()
        .contours
        .iter()
        .flat_map(|c| c.line.points.first().zip(c.line.points.last()))
        .for_each(|(p0, p1)| {
            let p0 = vec2(p0.xy.x, p0.xy.x);
            let point = vec2(p1.xy.x, p1.xy.y);
            let tangent = 20. * vec2(p1.v_xy.x, p1.v_xy.y);
            gizmos.circle_2d(p0, 1.0, Color::srgb(0.0, 1.0, 0.0));
            gizmos.circle_2d(point, 1.0, Color::srgb(1.0, 0.0, 0.0));
            gizmos.arrow_2d(point, point + tangent, Color::srgb(1., 0., 0.));
        });
}

pub fn draw_contours(world: Res<WorldResource>, mut gizmos: Gizmos) {
    world.contour_splines.iter().for_each(|s| {
        let resolution = 100 * s.segments().len();
        gizmos.linestrip(
            s.iter_positions(resolution).map(|pt| pt.extend(0.0)),
            Color::srgb(1.0, 1.0, 1.0),
        );
    });
}
