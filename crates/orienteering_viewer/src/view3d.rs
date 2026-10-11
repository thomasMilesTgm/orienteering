//! The 3D view: the terrain at the current step as a lit mesh, draped with the map, orbited by a
//! camera. The 2D camera keeps drawing the UI over it.

use super::*;
use bevy::{
    image::ImageSampler,
    input::mouse::AccumulatedMouseMotion,
    mesh::{Indices, PrimitiveTopology},
};

/// Mesh vertices per side.
const MESH_RESOLUTION: usize = 257;
/// Draped map texels per side.
const TEXTURE_RESOLUTION: f64 = 1024.;
/// Vertical exaggerations to cycle through: relief of a hundred meters or so over several
/// kilometers looks flat at true scale.
const EXAGGERATIONS: [f32; 4] = [1., 2., 3., 5.];

/// A terrain mesh with the map draped over it, in meters around the area's center: x east, y up,
/// z south.
pub(crate) struct TerrainMesh {
    pub(crate) positions: Vec<[f32; 3]>,
    pub(crate) normals: Vec<[f32; 3]>,
    pub(crate) uvs: Vec<[f32; 2]>,
    pub(crate) indices: Vec<u32>,
    pub(crate) rgba: Vec<u8>,
    pub(crate) texels: UVec2,
    pub(crate) center: Point2<f64>,
    pub(crate) center_height: f64,
}

impl TerrainMesh {
    /// Samples `surface` over `area` (square) with exact heights and normals; water is flat at
    /// 0 m. The map, rendered over the same area, becomes the texture.
    pub(crate) fn build<S: Surface + Sync>(surface: &S, area: Aabb) -> Self {
        let n = MESH_RESOLUTION;
        let side = area.width();
        let center = area.center();
        let samples: Vec<(f64, Vector2<f64>)> = (0..n * n)
            .into_par_iter()
            .map(|k| {
                let (i, j) = (k % n, k / n);
                let t = |i: usize| i as f64 / (n - 1) as f64;
                // Rows from north to south, so v runs down the texture.
                surface.height_and_gradient(area.lerp(t(i), 1. - t(j)))
            })
            .collect();

        let mut positions = Vec::with_capacity(n * n);
        let mut normals = Vec::with_capacity(n * n);
        let mut uvs = Vec::with_capacity(n * n);
        for (k, &(h, g)) in samples.iter().enumerate() {
            let (u, v) = (
                (k % n) as f64 / (n - 1) as f64,
                (k / n) as f64 / (n - 1) as f64,
            );
            let (x, z) = ((u - 0.5) * side, (v - 0.5) * side);
            positions.push([x as f32, h.max(0.) as f32, z as f32]);
            // The surface y = h(x, -z) has normal (-∂h/∂x, 1, ∂h/∂y).
            let normal = if h > 0. {
                Vec3::new(-g.x as f32, 1., g.y as f32).normalize()
            } else {
                Vec3::Y
            };
            normals.push(normal.to_array());
            uvs.push([u as f32, v as f32]);
        }
        let mut indices = Vec::with_capacity((n - 1) * (n - 1) * 6);
        for j in 0..n - 1 {
            for i in 0..n - 1 {
                let k = (j * n + i) as u32;
                let (right, down) = (k + 1, k + n as u32);
                indices.extend([k, down, right, right, down, down + 1]);
            }
        }

        let mut style = MapStyle::for_scale(side / TEXTURE_RESOLUTION);
        style.contours.density = style.contours.density.min(25.);
        let map = raster::render(surface, area, side / TEXTURE_RESOLUTION, &style);
        Self {
            positions,
            normals,
            uvs,
            indices,
            rgba: map
                .pixels
                .iter()
                .flat_map(|&[r, g, b]| [r, g, b, 255])
                .collect(),
            texels: UVec2::new(map.width as u32, map.height as u32),
            center,
            center_height: samples[n * n / 2].0.max(0.),
        }
    }
}

/// The 3D scene's assets, and what's currently in them.
#[derive(Resource)]
pub(crate) struct Terrain3d {
    pub(crate) mesh: Handle<Mesh>,
    pub(crate) texture: Handle<Image>,
    pub(crate) material: Handle<StandardMaterial>,
    /// The area's center and height there, and the job it was built for.
    pub(crate) shown: Option<(Point2<f64>, f64, Job)>,
}

impl Terrain3d {
    pub(crate) fn show(
        &mut self,
        terrain: TerrainMesh,
        job: &Job,
        meshes: &mut Assets<Mesh>,
        images: &mut Assets<Image>,
        materials: &mut Assets<StandardMaterial>,
    ) {
        if let Some(mut mesh) = meshes.get_mut(&self.mesh) {
            *mesh = Mesh::new(
                PrimitiveTopology::TriangleList,
                RenderAssetUsages::RENDER_WORLD | RenderAssetUsages::MAIN_WORLD,
            )
            .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, terrain.positions)
            .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, terrain.normals)
            .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, terrain.uvs)
            .with_inserted_indices(Indices::U32(terrain.indices));
        }
        if let Some(mut image) = images.get_mut(&self.texture) {
            *image = Image::new(
                Extent3d {
                    width: terrain.texels.x,
                    height: terrain.texels.y,
                    depth_or_array_layers: 1,
                },
                TextureDimension::D2,
                terrain.rgba,
                TextureFormat::Rgba8UnormSrgb,
                RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
            );
            image.sampler = ImageSampler::linear();
        }
        // The material holds the texture in its bind group: touch it so that's rebuilt with the
        // new image.
        if let Some(mut material) = materials.get_mut(&self.material) {
            material.base_color_texture = Some(self.texture.clone());
        }
        self.shown = Some((terrain.center, terrain.center_height, job.clone()));
    }
}

/// The 3D camera's orbit around the view's center.
#[derive(Resource)]
pub(crate) struct Orbit {
    /// Radians clockwise from looking north.
    pub(crate) yaw: f32,
    /// Radians above the horizon.
    pub(crate) pitch: f32,
    /// Meters from the target; reset when entering 3D.
    pub(crate) distance: f32,
    pub(crate) exaggeration: usize,
}

impl Default for Orbit {
    fn default() -> Self {
        Self {
            yaw: 0.,
            pitch: 0.6,
            distance: 0.,
            exaggeration: 1,
        }
    }
}

impl Orbit {
    pub(crate) fn exaggeration(&self) -> f32 {
        EXAGGERATIONS[self.exaggeration % EXAGGERATIONS.len()]
    }
}

#[derive(Component)]
pub(crate) struct Camera3dView;
#[derive(Component)]
pub(crate) struct TerrainModel;
/// The 2D camera, which also draws the UI.
#[derive(Component)]
pub(crate) struct Camera2dView;

pub(crate) fn setup_3d(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut images: ResMut<Assets<Image>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    commands.spawn((
        Camera3d::default(),
        Camera {
            order: 0,
            is_active: false,
            ..default()
        },
        Projection::Perspective(PerspectiveProjection {
            near: 1.,
            far: 1e6,
            ..default()
        }),
        Camera3dView,
    ));
    // The sun, from the north-west like the map's relief shading.
    commands.spawn((
        DirectionalLight {
            illuminance: 9000.,
            shadow_maps_enabled: true,
            ..default()
        },
        Transform::from_xyz(-1., 1.2, -1.).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    commands.insert_resource(GlobalAmbientLight {
        brightness: 400.,
        ..default()
    });

    // A placeholder until the first terrain arrives (it stays hidden till then).
    let mesh = meshes.add(Plane3d::default().mesh().size(1., 1.));
    let texture = images.add(Image::new_fill(
        Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[240, 240, 235, 255],
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    ));
    let material = materials.add(StandardMaterial {
        base_color_texture: Some(texture.clone()),
        perceptual_roughness: 0.95,
        reflectance: 0.1,
        ..default()
    });
    commands.spawn((
        Mesh3d(mesh.clone()),
        MeshMaterial3d(material.clone()),
        Visibility::Hidden,
        TerrainModel,
    ));
    commands.insert_resource(Terrain3d {
        mesh,
        texture,
        material,
        shown: None,
    });
}

/// Switches cameras and layers between the 2D and 3D views.
pub(crate) fn switch_views(
    controls: Res<Controls>,
    terrain: Res<Terrain3d>,
    mut camera_3d: Single<&mut Camera, (With<Camera3dView>, Without<Camera2dView>)>,
    mut camera_2d: Single<&mut Camera, (With<Camera2dView>, Without<Camera3dView>)>,
    mut model: Single<&mut Visibility, With<TerrainModel>>,
) {
    let three_d = controls.three_d;
    camera_3d.is_active = three_d;
    // Over the 3D view, the 2D camera only adds the UI.
    camera_2d.clear_color = if three_d {
        ClearColorConfig::None
    } else {
        ClearColorConfig::Default
    };
    **model = if three_d && terrain.shown.is_some() {
        Visibility::Visible
    } else {
        Visibility::Hidden
    };
}

/// Left-drag orbits, right-drag pans (moving the view, so the terrain follows), scroll changes
/// distance.
#[allow(clippy::too_many_arguments)]
pub(crate) fn orbit_controls(
    mut controls: ResMut<Controls>,
    mut orbit: ResMut<Orbit>,
    mut view: ResMut<View>,
    mouse: Res<ButtonInput<MouseButton>>,
    motion: Res<AccumulatedMouseMotion>,
    mut wheel: MessageReader<MouseWheel>,
    panels: Query<&RelativeCursorPosition, With<Panel>>,
    time: Res<Time>,
    mut dragging: Local<bool>,
) {
    let scroll: f32 = wheel
        .read()
        .map(|w| match w.unit {
            MouseScrollUnit::Line => w.y,
            MouseScrollUnit::Pixel => w.y / 40.,
        })
        .sum();
    if !controls.three_d {
        return;
    }
    let over_ui = panels.iter().any(|r| r.cursor_over);
    let pressed = mouse.any_just_pressed([MouseButton::Left, MouseButton::Right]);
    if pressed {
        *dragging = !over_ui && !controls.dragging_slider;
    }
    if !mouse.any_pressed([MouseButton::Left, MouseButton::Right]) {
        *dragging = false;
    }

    let d = motion.delta;
    if *dragging && d != Vec2::ZERO {
        if mouse.pressed(MouseButton::Left) {
            orbit.yaw -= d.x * 0.005;
            orbit.pitch = (orbit.pitch + d.y * 0.005).clamp(0.05, 1.5);
        } else {
            // Pan along the ground, relative to where the camera faces.
            let k = (orbit.distance * 0.0015) as f64;
            let (s, c) = (orbit.yaw.sin() as f64, orbit.yaw.cos() as f64);
            let right = Vector2::new(c, s);
            let forward = Vector2::new(-s, c);
            view.center += (-right * d.x as f64 + forward * d.y as f64) * k;
            controls.last_interaction = time.elapsed_secs();
        }
    }
    if scroll != 0. && !over_ui {
        orbit.distance = (orbit.distance * 1.15f32.powf(-scroll)).clamp(50., 2e5);
    }
}

/// Places the terrain around the view's center (scaled by the exaggeration) and the camera on its
/// orbit around it.
pub(crate) fn place_3d(
    controls: Res<Controls>,
    view: Res<View>,
    terrain: Res<Terrain3d>,
    mut orbit: ResMut<Orbit>,
    window: Single<&Window, With<PrimaryWindow>>,
    mut model: Single<&mut Transform, (With<TerrainModel>, Without<Camera3dView>)>,
    mut camera: Single<&mut Transform, (With<Camera3dView>, Without<TerrainModel>)>,
    mut was_3d: Local<bool>,
) {
    let three_d = controls.three_d;
    if three_d && !*was_3d {
        // Entering 3D: look at the whole area.
        orbit.distance = window.width().max(window.height()) * view.m_per_px as f32 * 0.9;
    }
    *was_3d = three_d;
    let Some((center, height, _)) = &terrain.shown else {
        return;
    };
    let exaggeration = orbit.exaggeration();
    let offset = center - view.center;
    model.translation = Vec3::new(offset.x as f32, 0., -offset.y as f32);
    model.scale = Vec3::new(1., exaggeration, 1.);

    let target = Vec3::new(0., *height as f32 * exaggeration, 0.);
    let (yaw, pitch) = (orbit.yaw, orbit.pitch);
    let eye = target
        + Vec3::new(
            yaw.sin() * pitch.cos(),
            pitch.sin(),
            yaw.cos() * pitch.cos(),
        ) * orbit.distance;
    **camera = Transform::from_translation(eye).looking_at(target, Vec3::Y);
}
