//! Generation and CPU rendering on a background thread, latest request wins.

use super::*;

#[derive(Clone, PartialEq, Debug)]
pub(crate) struct Job {
    pub(crate) seed: String,
    pub(crate) stage: f64,
    pub(crate) view: View,
    /// Logical pixels per map texel.
    pub(crate) texel: f64,
    /// Window size, logical pixels.
    pub(crate) size: (f64, f64),
    /// Flatten a scene for the GPU rather than render an image.
    pub(crate) gpu: bool,
    /// Build a draped terrain mesh for the 3D view instead.
    pub(crate) three_d: bool,
}

impl Job {
    pub(crate) fn spacing(&self) -> f64 {
        self.view.m_per_px * self.texel
    }

    pub(crate) fn area(&self) -> Aabb {
        let half = Vector2::new(self.size.0, self.size.1) * self.view.m_per_px / 2.;
        Aabb::new(self.view.center - half, self.view.center + half)
    }

    /// For 3D: a square around the view, as wide as the view's longer side.
    pub(crate) fn square(&self) -> Aabb {
        let side = self.size.0.max(self.size.1) * self.view.m_per_px;
        Aabb::centered(self.view.center, side)
    }

    /// For the GPU: the view plus a margin, so panning a little needs no new scene.
    pub(crate) fn covered(&self) -> Aabb {
        let area = self.area();
        area.grow((area.width().max(area.height()) * 0.5).min(3000.))
    }
}

pub(crate) fn contains(outer: &Aabb, inner: &Aabb) -> bool {
    outer.contains(inner.min) && outer.contains(inner.max)
}

pub(crate) struct Done {
    pub(crate) job: Job,
    pub(crate) world: terrain::World,
    pub(crate) chunks: HashMap<CellIndex, Arc<Levels>>,
    pub(crate) provinces: Arc<HashMap<CellIndex, Province>>,
    pub(crate) plates: Arc<HashMap<CellIndex, Plate>>,
    pub(crate) output: Output,
    pub(crate) millis: f64,
}

pub(crate) enum Output {
    Image { rgba: Vec<u8>, texels: UVec2 },
    Scene(FlatScene),
    Terrain(TerrainMesh),
}

#[derive(Default)]
pub(crate) struct Mailbox {
    pub(crate) request: Option<Job>,
    pub(crate) result: Option<Done>,
}

#[derive(Resource, Clone)]
pub(crate) struct Worker(pub(crate) Arc<(Mutex<Mailbox>, Condvar)>);

impl Worker {
    pub(crate) fn spawn() -> Self {
        let worker = Self(Arc::default());
        let shared = worker.0.clone();
        std::thread::spawn(move || {
            let mut cache = Cache::default();
            loop {
                let job = {
                    let (lock, ready) = &*shared;
                    let mut mail = lock.lock().unwrap();
                    while mail.request.is_none() {
                        mail = ready.wait(mail).unwrap();
                    }
                    mail.request.take().unwrap()
                };
                let done = cache.render(job);
                shared.0.lock().unwrap().result = Some(done);
            }
        });
        worker
    }

    pub(crate) fn send(&self, job: Job) {
        let (lock, ready) = &*self.0;
        lock.lock().unwrap().request = Some(job);
        ready.notify_one();
    }

    pub(crate) fn take(&self) -> Option<Done> {
        self.0.0.lock().unwrap().result.take()
    }
}

/// A traced chunk, its field at each generation depth, and flattened for the GPU.
pub(crate) struct Levels {
    pub(crate) traced: TracedChunk,
    /// `fields[d]`: the chunk with levels down to depth `d`.
    pub(crate) fields: Vec<Field>,
    pub(crate) flat: FlatCell,
}

#[derive(Default)]
pub(crate) struct Cache {
    pub(crate) seed: String,
    pub(crate) world: Option<terrain::World>,
    pub(crate) provinces: HashMap<CellIndex, Province>,
    pub(crate) province_flats: HashMap<CellIndex, FlatCell>,
    pub(crate) plates: HashMap<CellIndex, Plate>,
    pub(crate) plate_flats: HashMap<CellIndex, FlatCell>,
    pub(crate) chunks: HashMap<CellIndex, Arc<Levels>>,
}

impl Cache {
    pub(crate) fn render(&mut self, job: Job) -> Done {
        let start = Instant::now();
        if self.world.is_none() || self.seed != job.seed {
            *self = Self {
                seed: job.seed.clone(),
                world: Some(terrain::World::from_phrase(
                    &job.seed,
                    TerrainParams::default(),
                )),
                ..default()
            };
        }
        let world = self.world.clone().unwrap();
        let p = &world.params;
        let area = if job.three_d {
            job.square()
        } else if job.gpu {
            job.covered()
        } else {
            job.area()
        };

        // Forget what's far away, so wandering doesn't grow the cache without bound.
        if self.chunks.len() > 4000 {
            let keep = area.grow(4. * p.chunk_size);
            self.chunks
                .retain(|i, _| keep.contains(i.bounds(p.chunk_size).center()));
        }

        // Plates and provinces, covering the chunks' own generation contexts too. Provinces fade in
        // over the first step; plates are always there.
        let context = area.grow(2. * p.chunk_size);
        let plate_size = p.tectonics.plate_size;
        let plates: Vec<_> = CellIndex::covering(context, plate_size, PLATE_REACH).collect();
        for &i in &plates {
            if !self.plates.contains_key(&i) {
                let plate = world.plate(i);
                let origin = i.bounds(plate_size).min;
                let flat = FlatCell::new(&plate.elevation, origin, |_| ALWAYS);
                self.plate_flats.insert(i, flat.expect("plates flatten"));
                self.plates.insert(i, plate);
            }
        }
        let provinces: Vec<_> =
            CellIndex::covering(context, p.province_size, PROVINCE_REACH).collect();
        for &i in &provinces {
            if !self.provinces.contains_key(&i) {
                let province = world.province(i);
                let origin = i.bounds(p.province_size).min;
                let flat = FlatCell::new(&province.elevation, origin, |_| 0.);
                self.province_flats
                    .insert(i, flat.expect("provinces flatten"));
                self.provinces.insert(i, province);
            }
        }

        // Zoomed far out, chunks' detail would be invisible, and there would be far too many.
        let needed: Vec<_> = if job.view.m_per_px <= DETAIL_LIMIT {
            CellIndex::covering(area, p.chunk_size, 1).collect()
        } else {
            vec![]
        };
        let missing: Vec<_> = needed
            .iter()
            .filter(|i| !self.chunks.contains_key(i))
            .copied()
            .collect();
        let (cached_provinces, cached_plates) = (&self.provinces, &self.plates);
        let generated: Vec<_> = missing
            .par_iter()
            .map(|&i| {
                let traced = world.chunk_traced_in(i, cached_provinces, cached_plates);
                let fields = (0..=traced.chunk.depth)
                    .map(|d| traced.field_to_depth(d))
                    .collect();
                // The root is a window-less node holding the landform level's window; the
                // landforms (depth 0) fade in over step 2, each finer depth over the next.
                let origin = i.bounds(p.chunk_size).min;
                let flat = FlatCell::new(&traced.chunk.field, origin, |n| n as f32)
                    .expect("generated chunks flatten");
                (
                    i,
                    Arc::new(Levels {
                        traced,
                        fields,
                        flat,
                    }),
                )
            })
            .collect();
        self.chunks.extend(generated);

        let chunks: HashMap<_, _> = needed
            .iter()
            .filter_map(|i| Some((*i, self.chunks.get(i)?.clone())))
            .collect();
        let province_fields: HashMap<_, _> = provinces
            .iter()
            .map(|i| (*i, self.provinces[i].clone()))
            .collect();
        let plate_fields: HashMap<_, _> = plates
            .iter()
            .map(|i| (*i, self.plates[i].clone()))
            .collect();

        let staged = |stage: f64| {
            let lo = stage.floor().clamp(0., steps() as f64) as u32;
            Staged {
                baseline: p.mean_elevation,
                plates: &plate_fields,
                provinces: &province_fields,
                chunks: &chunks,
                lo,
                hi: (lo + 1).min(steps()),
                t: stage - lo as f64,
                chunk_size: p.chunk_size,
                province_size: p.province_size,
                plate_size,
            }
        };
        let output = if job.three_d {
            Output::Terrain(TerrainMesh::build(&staged(job.stage), area))
        } else if job.gpu {
            let mut scene = FlatScene::new(
                chunks.iter().map(|(i, c)| (*i, c.flat.clone())),
                p.chunk_size,
                provinces
                    .iter()
                    .map(|i| (*i, self.province_flats[i].clone())),
                p.province_size,
                plates.iter().map(|i| (*i, self.plate_flats[i].clone())),
                plate_size,
            );
            scene.baseline = p.mean_elevation as f32;
            Output::Scene(scene)
        } else {
            let surface = staged(job.stage);
            let map = raster::render(&surface, area, job.spacing(), &map_style(job.view.m_per_px));
            Output::Image {
                rgba: map
                    .pixels
                    .iter()
                    .flat_map(|&[r, g, b]| [r, g, b, 255])
                    .collect(),
                texels: UVec2::new(map.width as u32, map.height as u32),
            }
        };

        Done {
            world,
            chunks,
            provinces: Arc::new(province_fields),
            plates: Arc::new(plate_fields),
            output,
            millis: start.elapsed().as_secs_f64() * 1e3,
            job,
        }
    }
}

/// The terrain partway through generation: step `lo` cross-faded into step `hi` by `t`. Plates
/// are always present, provinces fade in over step 1, chunks' depth `d` over step `d + 2`.
pub(crate) struct Staged<'a> {
    pub(crate) baseline: f64,
    pub(crate) plates: &'a HashMap<CellIndex, Plate>,
    pub(crate) provinces: &'a HashMap<CellIndex, Province>,
    pub(crate) chunks: &'a HashMap<CellIndex, Arc<Levels>>,
    pub(crate) lo: u32,
    pub(crate) hi: u32,
    pub(crate) t: f64,
    pub(crate) chunk_size: f64,
    pub(crate) province_size: f64,
    pub(crate) plate_size: f64,
}

impl Staged<'_> {
    /// Sums `eval` over the plates, the provinces as far as they've faded in, and the chunks at
    /// steps `lo` and `hi` weighted by `1 - t` and `t`.
    pub(crate) fn sum<T: Copy + std::ops::Add<Output = T> + std::ops::Mul<f64, Output = T>>(
        &self,
        p: Point2<f64>,
        zero: T,
        eval: impl Fn(&Field) -> T,
    ) -> T {
        let plates = CellIndex::containing(p, self.plate_size)
            .within(PLATE_REACH)
            .filter_map(|i| self.plates.get(&i))
            .fold(zero, |acc, v| acc + eval(&v.elevation));
        let provinces = CellIndex::containing(p, self.province_size)
            .within(PROVINCE_REACH)
            .filter_map(|i| self.provinces.get(&i))
            .fold(zero, |acc, v| acc + eval(&v.elevation));
        let faded_in = (self.lo as f64 + self.t).clamp(0., 1.);
        let mut total = eval(&Field::Constant(self.baseline)) + plates + provinces * faded_in;

        let at = |step: u32| -> T {
            if step < 2 {
                return zero;
            }
            CellIndex::containing(p, self.chunk_size)
                .neighborhood()
                .filter_map(|i| self.chunks.get(&i))
                .fold(zero, |acc, c| {
                    let depth = (step as usize - 2).min(c.fields.len() - 1);
                    acc + eval(&c.fields[depth])
                })
        };
        if self.t > 0. && self.hi != self.lo {
            total = total + at(self.lo) * (1. - self.t) + at(self.hi) * self.t;
        } else {
            total = total + at(self.lo);
        }
        total
    }
}

/// Height and gradient, summable.
#[derive(Clone, Copy)]
pub(crate) struct Hg(pub(crate) f64, Vector2<f64>);

impl std::ops::Add for Hg {
    type Output = Self;
    fn add(self, o: Self) -> Self {
        Self(self.0 + o.0, self.1 + o.1)
    }
}

impl std::ops::Mul<f64> for Hg {
    type Output = Self;
    fn mul(self, k: f64) -> Self {
        Self(self.0 * k, self.1 * k)
    }
}

impl Surface for Staged<'_> {
    fn height(&self, p: Point2<f64>) -> f64 {
        self.sum(p, 0., |f| f.height(p))
    }

    fn height_and_gradient(&self, p: Point2<f64>) -> (f64, Vector2<f64>) {
        let Hg(h, g) = self.sum(p, Hg(0., Vector2::zeros()), |f| {
            let (h, g) = f.height_and_gradient(p);
            Hg(h, g)
        });
        (h, g)
    }
}

#[derive(Resource)]
pub(crate) struct MapImage(pub(crate) Handle<Image>);

/// Spawns the sprite showing CPU-rendered maps.
pub(crate) fn setup_cpu_map(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    let image = images.add(Image::new_fill(
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
    commands.spawn((
        Sprite {
            image: image.clone(),
            custom_size: Some(Vec2::ZERO),
            ..default()
        },
        MapSprite,
    ));
    commands.insert_resource(MapImage(image));
}

pub(crate) fn request_render(
    controls: Res<Controls>,
    view: Res<View>,
    worker: Res<Worker>,
    gpu: Res<GpuMap>,
    window: Single<&Window, With<PrimaryWindow>>,
    time: Res<Time>,
    mut sent: Local<Option<Job>>,
) {
    let size = (window.width() as f64, window.height() as f64);
    if size.0 < 1. || size.1 < 1. {
        return;
    }
    let settled = time.elapsed_secs() - controls.last_interaction > SETTLE;
    let mut job = Job {
        seed: controls.seed.clone(),
        stage: controls.stage,
        view: *view,
        texel: if settled { 1. } else { 2. },
        size,
        gpu: controls.gpu && !controls.three_d,
        three_d: controls.three_d,
    };

    if controls.three_d {
        job.texel = 1.;
        if sent.as_ref() == Some(&job) {
            return;
        }
    } else if controls.gpu {
        // The stage and small pans are just uniforms; only ask for a new scene when the view
        // leaves the generated area (or the seed changes).
        job.stage = 0.;
        job.texel = 1.;
        let area = job.area();
        let have = gpu
            .scene
            .as_ref()
            .is_some_and(|s| s.seed == job.seed && contains(&s.covered, &area));
        let coming = sent.as_ref().is_some_and(|s| {
            s.gpu && !s.three_d && s.seed == job.seed && contains(&s.covered(), &area)
        });
        if have || coming {
            return;
        }
    } else if sent.as_ref() == Some(&job) {
        return;
    }
    worker.send(job.clone());
    *sent = Some(job);
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn receive_render(
    mut commands: Commands,
    worker: Res<Worker>,
    image: Res<MapImage>,
    mut images: ResMut<Assets<Image>>,
    mut gpu: ResMut<GpuMap>,
    mut buffers: ResMut<Assets<ShaderBuffer>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut terrain: ResMut<Terrain3d>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let Some(done) = worker.take() else {
        return;
    };
    match done.output {
        Output::Image { rgba, texels } => {
            if let Some(mut img) = images.get_mut(&image.0) {
                *img = Image::new(
                    Extent3d {
                        width: texels.x,
                        height: texels.y,
                        depth_or_array_layers: 1,
                    },
                    TextureDimension::D2,
                    rgba,
                    TextureFormat::Rgba8UnormSrgb,
                    RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
                );
            }
            commands.insert_resource(CpuMap {
                job: done.job.clone(),
                texels,
                millis: done.millis,
            });
        }
        Output::Terrain(mesh) => {
            terrain.show(mesh, &done.job, &mut meshes, &mut images, &mut materials);
        }
        Output::Scene(scene) => {
            let mut cells = scene.chunks.ranges.clone();
            cells.extend(&scene.provinces.ranges);
            cells.extend(&scene.plates.ranges);
            let mut set = |handle: &Handle<ShaderBuffer>, data: ShaderBuffer| {
                if let Some(mut buffer) = buffers.get_mut(handle) {
                    *buffer = data;
                }
            };
            // Buffers can't be empty.
            let non_empty = |mut v: Vec<[f32; 4]>| {
                if v.is_empty() {
                    v.push([0.; 4]);
                }
                v
            };
            set(
                &gpu.nodes,
                ShaderBuffer::from(non_empty(scene.nodes.clone())),
            );
            set(
                &gpu.features,
                ShaderBuffer::from(non_empty(scene.features.clone())),
            );
            set(&gpu.cells, ShaderBuffer::from(cells));
            gpu.version += 1;
            gpu.scene = Some(SceneInfo {
                seed: done.job.seed.clone(),
                covered: done.job.covered(),
                chunk_origin: scene.chunks.origin(),
                province_origin: scene.provinces.origin(),
                plate_origin: scene.plates.origin(),
                baseline: scene.baseline,
                tables: UVec4::new(
                    scene.chunks.size[0],
                    scene.chunks.size[1],
                    scene.provinces.size[0],
                    scene.provinces.size[1],
                ),
                plate_table: UVec4::new(scene.plates.size[0], scene.plates.size[1], 0, 0),
                nodes: scene.nodes.len() / NODE_VEC4S,
                millis: done.millis,
            });
        }
    }
    commands.insert_resource(Shown {
        world: done.world,
        chunks: done.chunks,
        provinces: done.provinces,
        plates: done.plates,
    });
}

/// The CPU image, shifted and scaled from where it was rendered while a new one is on its way.
pub(crate) fn place_cpu_map(
    controls: Res<Controls>,
    view: Res<View>,
    cpu: Option<Res<CpuMap>>,
    sprite: Single<(&mut Sprite, &mut Transform, &mut Visibility), With<MapSprite>>,
) {
    let (mut sprite, mut transform, mut visibility) = sprite.into_inner();
    let Some(cpu) = cpu.filter(|_| !controls.gpu && !controls.three_d) else {
        *visibility = Visibility::Hidden;
        return;
    };
    *visibility = Visibility::Visible;
    let size = cpu.texels.as_dvec2() * cpu.job.spacing() / view.m_per_px;
    sprite.custom_size = Some(Vec2::new(size.x as f32, size.y as f32));
    transform.translation = view.screen(cpu.job.view.center).extend(0.);
}
