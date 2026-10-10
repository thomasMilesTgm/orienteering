//! Interactive visualization of how the terrain is generated.
//!
//! `cargo run --release -p orienteering_viewer`
//!
//! - **Seed**: click the seed box, type, and press Enter (Escape cancels).
//! - **Generation steps**: drag the slider, or use ←/→ to step and Space to play. The terrain is
//!   built coarse to fine: provinces, then chunk landforms, then each level of detail. Between
//!   steps the next level fades in.
//! - **Inspect**: hover over the map to see the quadtree node under the cursor at the current step:
//!   what the terrain looked like around it when it was generated (its site), how it responded,
//!   and the features it added, which are outlined over the map.
//! - **Navigate**: scroll to zoom, drag to pan.
//! - **GPU / CPU**: press G to switch renderers.
//!
//! On the GPU (the default), the generated fields are flattened ([`orienteering::flat`]) and
//! evaluated per pixel in two passes: `assets/shaders/terrain_field.wgsl` evaluates the terrain
//! into a float texture, only when the view, stage or scene changes, and
//! `assets/shaders/terrain_style.wgsl` styles it every frame. On the CPU, rendering runs on a
//! background thread, latest request wins, at half resolution while you interact, then refined.
//! Generation always runs on the CPU.

use bevy::{
    asset::RenderAssetUsages,
    camera::{RenderTarget, visibility::RenderLayers},
    input::{
        ButtonState,
        keyboard::{Key, KeyboardInput},
        mouse::{MouseScrollUnit, MouseWheel},
    },
    prelude::*,
    reflect::TypePath,
    render::pipelined_rendering::PipelinedRenderingPlugin,
    render::{
        render_resource::{AsBindGroup, Extent3d, ShaderType, TextureDimension, TextureFormat},
        storage::ShaderBuffer,
    },
    shader::ShaderRef,
    sprite_render::{Material2d, Material2dPlugin},
    ui::RelativeCursorPosition,
    window::PrimaryWindow,
};
use nalgebra::{Point2, Vector2};
use orienteering::{
    field::{Field, Surface},
    flat::{ALWAYS, FlatCell, FlatScene, NODE_VEC4S},
    map::{
        Biome, BiomeRules,
        raster::{self, MapStyle},
    },
    math::Aabb,
    terrain::{self, CellIndex, NodeTrace, Province, TerrainParams, TracedChunk},
};
use rayon::prelude::*;
use std::collections::HashMap;
use std::num::NonZero;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Instant;

mod gpu;
mod input;
mod inspect;
mod ui;
mod worker;

use gpu::*;
use input::*;
use inspect::*;
use ui::*;
use worker::*;

const DEFAULT_SEED: &str = "misty hollow";
/// Meters per logical pixel.
const SCALE_RANGE: (f64, f64) = (0.25, 16.);
/// Seconds after the last interaction before rendering at full resolution.
const SETTLE: f32 = 0.25;
/// Steps per second while playing.
const PLAY_SPEED: f64 = 0.6;

fn main() {
    App::new()
        .add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "Orienteering: terrain generation".into(),
                        resolution: (1600, 1000).into(),
                        // Keep input-to-screen latency low when a frame is GPU-heavy.
                        desired_maximum_frame_latency: NonZero::new(1),
                        ..default()
                    }),
                    ..default()
                })
                // Pipelined rendering adds a frame of latency, for throughput this app doesn't
                // need.
                .build()
                .disable::<PipelinedRenderingPlugin>(),
        )
        .add_plugins((
            Material2dPlugin::<FieldMaterial>::default(),
            Material2dPlugin::<StyleMaterial>::default(),
        ))
        .insert_resource(Controls {
            seed: DEFAULT_SEED.into(),
            stage: steps() as f64,
            gpu: true,
            ..default()
        })
        .insert_resource(View {
            center: Point2::origin(),
            m_per_px: 4.,
        })
        .insert_resource(Worker::spawn())
        .add_systems(Startup, (setup_camera, setup_gpu, setup_cpu_map, setup_ui))
        .add_systems(
            Update,
            (
                edit_seed,
                drag_slider,
                step_keys,
                pan_and_zoom,
                request_render,
                receive_render,
                update_gpu_map,
                place_gpu_map,
                place_cpu_map,
                inspect,
                update_seed_ui,
                update_slider_ui,
                update_status_ui,
            )
                .chain(),
        )
        .run();
}

fn setup_camera(mut commands: Commands) {
    commands.spawn(Camera2d);
}

/// The last step: provinces, landforms, then each level of detail.
fn steps() -> u32 {
    TerrainParams::default().max_depth + 1
}

fn step_name(k: u32) -> String {
    match k {
        0 => "Provinces".into(),
        1 => "Landforms".into(),
        k => format!("Detail {}", k - 1),
    }
}

fn step_description(k: u32) -> String {
    let p = TerrainParams::default();
    match k {
        0 => format!(
            "{} km provinces set the broad shape: basins, uplands and ranges. A smooth \
             ruggedness field uplifts rugged ground and lets smooth ground sink.",
            p.province_size / 1000.
        ),
        1 => format!(
            "{} km chunks add hills, ridges and valleys, each scaled by the activity of the \
             ground it sits on.",
            p.chunk_size / 1000.
        ),
        k => {
            let size = p.chunk_size / 2f64.powi(k as i32 - 1);
            format!(
                "{size} m nodes respond to the terrain so far: channels cut downhill, rugged \
                 ground gets steeper, smooth ground is damped and filled."
            )
        }
    }
}

#[derive(Resource, Default)]
pub(crate) struct Controls {
    pub(crate) seed: String,
    /// The seed being typed, while editing.
    pub(crate) draft: Option<String>,
    /// Continuous position in [0, steps()].
    pub(crate) stage: f64,
    pub(crate) playing: bool,
    pub(crate) dragging_slider: bool,
    /// Last cursor position while panning.
    pub(crate) panning: Option<Vec2>,
    /// Time of the last interaction, to render at low resolution until things settle.
    pub(crate) last_interaction: f32,
    /// Render on the GPU rather than the CPU.
    pub(crate) gpu: bool,
}

#[derive(Resource, Clone, Copy, PartialEq, Debug)]
pub(crate) struct View {
    pub(crate) center: Point2<f64>,
    pub(crate) m_per_px: f64,
}

impl View {
    /// World position of a cursor position (logical pixels, top-left origin).
    pub(crate) fn world(&self, window: &Window, cursor: Vec2) -> Point2<f64> {
        let screen = Vec2::new(
            cursor.x - window.width() / 2.,
            window.height() / 2. - cursor.y,
        );
        self.center + Vector2::new(screen.x as f64, screen.y as f64) * self.m_per_px
    }

    /// Screen position (camera space: origin at the window center, y up) of a world point.
    pub(crate) fn screen(&self, p: Point2<f64>) -> Vec2 {
        let v = (p - self.center) / self.m_per_px;
        Vec2::new(v.x as f32, v.y as f32)
    }
}

/// The latest generated data, for inspection.
#[derive(Resource)]
pub(crate) struct Shown {
    pub(crate) world: terrain::World,
    pub(crate) chunks: HashMap<CellIndex, Arc<Levels>>,
    pub(crate) provinces: Arc<HashMap<CellIndex, Province>>,
}

impl Shown {
    /// The terrain at a (possibly fractional) generation stage.
    pub(crate) fn staged(&self, stage: f64) -> Staged<'_> {
        let lo = stage.floor().clamp(0., steps() as f64) as u32;
        let p = &self.world.params;
        Staged {
            provinces: &self.provinces,
            chunks: &self.chunks,
            lo,
            hi: (lo + 1).min(steps()),
            t: stage - lo as f64,
            chunk_size: p.chunk_size,
            province_size: p.province_size,
        }
    }
}

/// The CPU-rendered map image's job and size.
#[derive(Resource)]
pub(crate) struct CpuMap {
    pub(crate) job: Job,
    pub(crate) texels: UVec2,
    pub(crate) millis: f64,
}
