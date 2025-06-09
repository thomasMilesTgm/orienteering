//! Scalar field plugin

use bevy::{
    asset::RenderAssetUsages, platform::collections::HashMap, prelude::*,
    render::render_resource::Extent3d,
};
use nalgebra::Point2;
use orienteering::topography::scalar_field::{CircularPotential, ScalarField};

/// Size of a map chunk, Meters
const CHUNK_SIZE: u32 = 1000;

pub struct ScalarFieldPlugin;

impl Plugin for ScalarFieldPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(WorldSetupPlugin);
    }
}

struct WorldSetupPlugin;

impl Plugin for WorldSetupPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<MapAssets>()
            .init_resource::<GPEField>()
            .add_systems(Startup, init_world);
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
}

#[derive(Resource, Debug, Clone, Default)]
struct MapAssets {
    chunks: HashMap<Point2<isize>, Handle<Image>>,
}

fn init_world(mut assets: ResMut<MapAssets>, field: Res<GPEField>) {
    let mut min = f32::MAX;
    let mut max = f32::MIN;
    let mut phi_ij = vec![];
    for i in 0..CHUNK_SIZE {
        let x = i as f32 / CHUNK_SIZE as f32;
        for j in 0..CHUNK_SIZE {
            let y = j as f32 / CHUNK_SIZE as f32;
            let gpe = field.strength(x, y);
            min = min.min(gpe);
            max = max.max(gpe);
            phi_ij.push(gpe);
        }
    }
    let range = max - min;

    let pixel_bytes = phi_ij
        .into_iter()
        .map(|phi| (phi - min) / range)
        .map(|pc| Color::hsl(360. * pc, 1., 0.25))
        .flat_map(|color| color.to_srgba().to_u8_array_no_alpha())
        .collect::<Vec<_>>();

    let image = Image::new(
        Extent3d {
            width: CHUNK_SIZE,
            height: CHUNK_SIZE,
            depth_or_array_layers: 0,
        },
        bevy::render::render_resource::TextureDimension::D2,
        pixel_bytes,
        bevy::render::render_resource::TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::all(),
    );
}
