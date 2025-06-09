//! Scalar field plugin

use bevy::{
    asset::RenderAssetUsages,
    platform::collections::HashMap,
    prelude::*,
    render::render_resource::{Extent3d, TextureFormat},
};
use nalgebra::Point2;
use orienteering::topography::scalar_field::*;

/// Size of a map chunk, Meters
const CHUNK_SIZE: u32 = 1000;
const INIT_CHUNK_ID: Point2<i32> = Point2::new(0, 0);
const N_SHADES: usize = 32;

pub struct ScalarFieldPlugin;

impl Plugin for ScalarFieldPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(WorldSetupPlugin);
    }
}

struct WorldSetupPlugin;

impl Plugin for WorldSetupPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<WorldMap>()
            .init_resource::<GPEField>()
            .init_resource::<Assets<Image>>()
            .add_systems(Startup, init_world)
            .add_systems(Startup, spawn_chunks.after(init_world));
    }
}

/// Gravitational potential field for the world map
#[derive(Resource, Debug, Clone, Default)]
struct GPEField {
    f: CircularPotential,
}

impl GPEField {
    pub fn strength(&self, x: f32, y: f32) -> f32 {
        self.f.phi(Point2::new(x, y))
    }
    fn generate_chunk(&self, id: Point2<i32>) -> Image {
        let mut min = f32::MAX;
        let mut max = f32::MIN;

        let chunk_size = CHUNK_SIZE as i32;

        let r0 = id.x * chunk_size;
        let r1 = id.y * chunk_size + chunk_size;

        let field_stength = (r0..r1)
            .flat_map(|i| {
                let x = 10. * i as f32 / CHUNK_SIZE as f32;
                (r0..r1).map(move |j| {
                    let y = 10. * j as f32 / CHUNK_SIZE as f32;
                    self.strength(x, y)
                })
            })
            .inspect(|gpe| {
                min = min.min(*gpe);
                max = max.max(*gpe);
            })
            .collect::<Vec<_>>();

        dbg!(min, max);

        const GREEN_HUE: f32 = 120.;
        const CYAN_HUE: f32 = 180.;
        const BLUE_HUE: f32 = 240.;
        let min_depth = (-min).max(0.);
        let max_height = max.max(0.);

        let hue = |z: f32| {
            if z == 0. {
                CYAN_HUE
            } else if z.is_sign_negative() {
                BLUE_HUE
            } else {
                GREEN_HUE
            }
        };
        let saturation = |z: f32| {
            if z == 0. {
                1.
            } else if z.is_sign_negative() {
                0.8
            } else {
                0.5
            }
        };
        let brightness = |z: f32| {
            let b = if z == 0. {
                0.2
            } else if z.is_sign_negative() {
                let range = if max_height.is_sign_negative() {
                    max_height + min_depth
                } else {
                    min_depth
                };
                0.5 - 0.5 * z.abs() / range
            } else {
                let range = if min_depth.is_sign_positive() {
                    max_height - min
                } else {
                    min_depth
                };
                0.2 + 0.5 * z.abs() / range
            };
            (b * N_SHADES as f32).round() / N_SHADES as f32
        };

        let pixel_bytes = field_stength
            .into_iter()
            .map(|z| Color::hsl(hue(z), saturation(z), brightness(z)))
            .flat_map(|color| color.to_srgba().to_f32_array())
            .flat_map(|c| c.to_ne_bytes())
            .collect::<Vec<_>>();

        Image::new(
            Extent3d {
                width: CHUNK_SIZE,
                height: CHUNK_SIZE,
                depth_or_array_layers: 1,
            },
            bevy::render::render_resource::TextureDimension::D2,
            pixel_bytes,
            TextureFormat::Rgba32Float,
            RenderAssetUsages::all(),
        )
    }
}

#[derive(Resource, Debug, Clone, Default)]
struct WorldMap {
    chunks: HashMap<Point2<i32>, Handle<Image>>,
}

fn init_world(field: Res<GPEField>, mut map: ResMut<WorldMap>, mut images: ResMut<Assets<Image>>) {
    log::info!("Initializing world map with chunk size: {}", CHUNK_SIZE);
    let chunk_id = Point2::origin();
    let image = field.generate_chunk(chunk_id);
    let handle = images.add(image);
    map.chunks.insert(chunk_id, handle);
}

fn spawn_chunks(mut commands: Commands, map: Res<WorldMap>) {
    log::info!("Spawning Chunk");
    let chunk = map.chunks.get(&INIT_CHUNK_ID).unwrap();

    let _root = commands
        .spawn((
            Sprite {
                color: Color::WHITE,
                image: chunk.clone(),
                ..Default::default()
            },
            //..
        ))
        .id();
}
