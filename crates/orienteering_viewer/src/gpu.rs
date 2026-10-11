//! The GPU map, in two passes: pass 1 evaluates the flattened terrain into a float texture when
//! its inputs change, pass 2 styles it every frame.

use super::*;

/// The GPU map, in two passes: pass 1 evaluates the terrain into `field` (height and gradient
/// per physical pixel), only when its inputs change; pass 2 styles it every frame.
#[derive(Resource)]
pub(crate) struct GpuMap {
    pub(crate) field_material: Handle<FieldMaterial>,
    pub(crate) style_material: Handle<StyleMaterial>,
    pub(crate) field: Handle<Image>,
    pub(crate) nodes: Handle<ShaderBuffer>,
    pub(crate) features: Handle<ShaderBuffer>,
    pub(crate) cells: Handle<ShaderBuffer>,
    pub(crate) scene: Option<SceneInfo>,
    /// Bumped whenever a new scene is uploaded, so pass 1 reruns.
    pub(crate) version: u64,
}

pub(crate) struct SceneInfo {
    pub(crate) seed: String,
    pub(crate) covered: Aabb,
    pub(crate) chunk_origin: Point2<f64>,
    pub(crate) province_origin: Point2<f64>,
    pub(crate) plate_origin: Point2<f64>,
    pub(crate) baseline: f32,
    pub(crate) tables: UVec4,
    pub(crate) plate_table: UVec4,
    pub(crate) nodes: usize,
    pub(crate) millis: f64,
}

/// Pass 1: evaluates the terrain from the flattened scene.
#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
pub(crate) struct FieldMaterial {
    #[uniform(0)]
    pub(crate) params: FieldParams,
    #[storage(1, read_only)]
    pub(crate) nodes: Handle<ShaderBuffer>,
    #[storage(2, read_only)]
    pub(crate) features: Handle<ShaderBuffer>,
    #[storage(3, read_only)]
    pub(crate) cells: Handle<ShaderBuffer>,
}

impl Material2d for FieldMaterial {
    fn fragment_shader() -> ShaderRef {
        "shaders/terrain_field.wgsl".into()
    }
}

/// Mirrors `Params` in terrain_field.wgsl.
#[derive(ShaderType, Debug, Clone, Default, PartialEq)]
pub(crate) struct FieldParams {
    pub(crate) view: Vec4,
    pub(crate) plates: Vec4,
    pub(crate) scale: Vec4,
    pub(crate) tables: UVec4,
    pub(crate) plate_table: UVec4,
}

/// Pass 2: styles the evaluated terrain.
#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
pub(crate) struct StyleMaterial {
    #[uniform(0)]
    pub(crate) style: StyleParams,
    #[texture(1, sample_type = "float", filterable = false)]
    pub(crate) field: Handle<Image>,
}

impl Material2d for StyleMaterial {
    fn fragment_shader() -> ShaderRef {
        "shaders/terrain_style.wgsl".into()
    }
}

/// Mirrors `Style` in terrain_style.wgsl.
#[derive(ShaderType, Debug, Clone, Default, PartialEq)]
pub(crate) struct StyleParams {
    pub(crate) style: Vec4,
    pub(crate) rules: [Vec4; 8],
    pub(crate) rule_biomes: [Vec4; 2],
    pub(crate) colors: [Vec4; 6],
}

/// The offscreen camera for pass 1.
#[derive(Component)]
pub(crate) struct FieldCamera;
/// The quad pass 1 renders.
#[derive(Component)]
pub(crate) struct FieldQuad;

/// The layer pass 1 renders on, apart from everything else.
pub(crate) const FIELD_LAYER: usize = 1;

/// Spawns both passes' cameras, quads and materials; the field is filled in once a scene arrives.
pub(crate) fn setup_gpu(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut field_materials: ResMut<Assets<FieldMaterial>>,
    mut style_materials: ResMut<Assets<StyleMaterial>>,
    mut buffers: ResMut<Assets<ShaderBuffer>>,
    window: Single<&Window, With<PrimaryWindow>>,
) {
    // GPU pass 1: an offscreen camera renders a quad evaluating the terrain into a float texture
    // the size of the window, in physical pixels (1 world unit = 1 texel).
    let field = images.add(field_image(&window));
    let nodes = buffers.add(ShaderBuffer::from(vec![[0f32; 4]; NODE_VEC4S]));
    let features = buffers.add(ShaderBuffer::from(vec![[0f32; 4]; 2]));
    let cells = buffers.add(ShaderBuffer::from(vec![[0u32; 2]]));
    let field_material = field_materials.add(FieldMaterial {
        params: FieldParams::default(),
        nodes: nodes.clone(),
        features: features.clone(),
        cells: cells.clone(),
    });
    let quad = meshes.add(Rectangle::default());
    commands.spawn((
        Camera2d,
        Camera {
            order: -1,
            is_active: false,
            clear_color: ClearColorConfig::Custom(Color::NONE),
            ..default()
        },
        RenderTarget::Image(field.clone().into()),
        Msaa::Off,
        RenderLayers::layer(FIELD_LAYER),
        FieldCamera,
    ));
    commands.spawn((
        Mesh2d(quad.clone()),
        MeshMaterial2d(field_material.clone()),
        RenderLayers::layer(FIELD_LAYER),
        FieldQuad,
    ));

    // GPU pass 2: a quad covering the window styles the field.
    let style_material = style_materials.add(StyleMaterial {
        style: StyleParams::default(),
        field: field.clone(),
    });
    commands.spawn((
        Mesh2d(quad),
        MeshMaterial2d(style_material.clone()),
        Visibility::Hidden,
        GpuQuad,
    ));
    commands.insert_resource(GpuMap {
        field_material,
        style_material,
        field,
        nodes,
        features,
        cells,
        scene: None,
        version: 0,
    });
}

/// A float render target for pass 1, the size of the window in physical pixels.
pub(crate) fn field_image(window: &Window) -> Image {
    Image::new_target_texture(
        window.physical_width().max(1),
        window.physical_height().max(1),
        TextureFormat::Rgba32Float,
        None,
    )
}

/// Updates both passes' uniforms, and runs pass 1 only when its inputs change: the view, the
/// stage, the scene or the window size. Hovering and other redraws then cost only pass 2.
#[allow(clippy::too_many_arguments)]
pub(crate) fn update_gpu_map(
    controls: Res<Controls>,
    view: Res<View>,
    gpu: Res<GpuMap>,
    window: Single<&Window, With<PrimaryWindow>>,
    mut images: ResMut<Assets<Image>>,
    mut field_materials: ResMut<Assets<FieldMaterial>>,
    mut style_materials: ResMut<Assets<StyleMaterial>>,
    camera: Single<&mut Camera, With<FieldCamera>>,
    time: Res<Time>,
    mut last: Local<(FieldParams, StyleParams, u64, UVec2)>,
    mut rerun_until: Local<f32>,
) {
    let mut camera = camera.into_inner();
    let Some(scene) = gpu.scene.as_ref().filter(|_| controls.gpu) else {
        camera.is_active = false;
        return;
    };

    // Resizing replaces the target's contents, so it needs a rerun too. (Otherwise the image is
    // never touched from here: marking it changed would re-upload its blank CPU-side data.)
    let size = UVec2::new(window.physical_width(), window.physical_height());
    if size != last.3 {
        if let Some(mut image) = images.get_mut(&gpu.field) {
            *image = field_image(&window);
        }
        last.3 = size;
        *rerun_until = time.elapsed_secs() + RERUN;
    }

    let p = TerrainParams::default();
    let offset = |origin: Point2<f64>| view.center - origin;
    let (oc, op) = (offset(scene.chunk_origin), offset(scene.province_origin));
    let ot = offset(scene.plate_origin);
    let field = FieldParams {
        view: Vec4::new(oc.x as f32, oc.y as f32, op.x as f32, op.y as f32),
        plates: Vec4::new(
            ot.x as f32,
            ot.y as f32,
            p.tectonics.plate_size as f32,
            scene.baseline,
        ),
        scale: Vec4::new(
            p.chunk_size as f32,
            p.province_size as f32,
            // Pass 1 works in physical pixels.
            (view.m_per_px / window.scale_factor() as f64) as f32,
            controls.stage as f32,
        ),
        tables: scene.tables,
        plate_table: scene.plate_table,
    };
    if field != last.0 || gpu.version != last.2 {
        if let Some(mut material) = field_materials.get_mut(&gpu.field_material) {
            material.params = field.clone();
        }
        // The first scene also waits for pass 1's pipeline, which compiles in the background
        // (draws are skipped until it's ready).
        let first = last.2 == 0;
        last.0 = field;
        last.2 = gpu.version;
        *rerun_until = time.elapsed_secs() + if first { WARM_UP } else { RERUN };
    }

    let rules = BiomeRules::default();
    let mut style = StyleParams {
        style: Vec4::new(
            map_style(view.m_per_px).contours.density as f32,
            // Half the averaging scale, in physical pixels.
            (rules.slope_scale / 2. / view.m_per_px) as f32 * window.scale_factor(),
            rules.rules.len().min(8) as f32,
            rules.fallback as usize as f32,
        ),
        ..default()
    };
    let bound = |b: Option<f64>, default: f32| b.map_or(default, |v| v as f32);
    for (k, rule) in rules.rules.iter().take(8).enumerate() {
        style.rules[k] = Vec4::new(
            bound(rule.altitude.min, -1e9),
            bound(rule.altitude.max, 1e9),
            bound(rule.slope.min, -1e9),
            bound(rule.slope.max, 1e9),
        );
        style.rule_biomes[k / 4][k % 4] = rule.biome as usize as f32;
    }
    for biome in Biome::ALL {
        let [r, g, b] = raster::fill(biome).map(|c| (c / 255.) as f32);
        let linear = Color::srgb(r, g, b).to_linear();
        style.colors[biome as usize] = Vec4::new(linear.red, linear.green, linear.blue, 1.);
    }
    if style != last.1
        && let Some(mut material) = style_materials.get_mut(&gpu.style_material)
    {
        material.style = style.clone();
        last.1 = style;
    }

    // Keep pass 1 running a little after each change: new buffers, uniforms or pipelines can
    // reach the GPU a frame or more late, and an inactive camera leaves the field as it was.
    camera.is_active = time.elapsed_secs() < *rerun_until;
}

/// Seconds pass 1 keeps running after its inputs change.
pub(crate) const RERUN: f32 = 0.25;
/// Seconds pass 1 keeps running after the first scene arrives, while its pipeline compiles.
pub(crate) const WARM_UP: f32 = 5.;

/// The GPU map covers the window whenever it's in use: pass 2 in logical pixels for the main
/// camera, pass 1 in physical pixels for its target.
pub(crate) fn place_gpu_map(
    controls: Res<Controls>,
    gpu: Res<GpuMap>,
    window: Single<&Window, With<PrimaryWindow>>,
    style: Single<(&mut Transform, &mut Visibility), With<GpuQuad>>,
    field: Single<&mut Transform, (With<FieldQuad>, Without<GpuQuad>)>,
) {
    let (mut transform, mut visibility) = style.into_inner();
    *visibility = if controls.gpu && gpu.scene.is_some() {
        Visibility::Visible
    } else {
        Visibility::Hidden
    };
    transform.scale = Vec3::new(window.width(), window.height(), 1.);
    field.into_inner().scale = Vec3::new(
        window.physical_width() as f32,
        window.physical_height() as f32,
        1.,
    );
}
