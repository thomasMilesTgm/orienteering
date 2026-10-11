//! Height labels on the heavier contours, placed in the background and drawn as text anchored in
//! the world, upright and in a gap in the line.

use super::*;
use bevy::text::TextLayoutInfo;
use orienteering::map::{ContourLabel, HeightGrid, LabelSpacing, contours, levels, place_labels};

/// Label text size, logical pixels.
const LABEL_SIZE: f32 = 13.;
const LABEL_COLOR: Color = Color::srgb(0.6, 0.33, 0.1);

/// What a set of labels was placed for: they're stale once any of it changes.
#[derive(Clone, PartialEq, Debug)]
pub(crate) struct LabelKey {
    pub(crate) seed: String,
    pub(crate) stage: f64,
    pub(crate) view: View,
    pub(crate) size: (f64, f64),
}

pub(crate) struct LabelJob {
    pub(crate) key: LabelKey,
    pub(crate) world: terrain::World,
    pub(crate) chunks: HashMap<CellIndex, Arc<Levels>>,
    pub(crate) provinces: Arc<HashMap<CellIndex, Province>>,
    pub(crate) plates: Arc<HashMap<CellIndex, Plate>>,
}

pub(crate) struct Placed {
    pub(crate) key: LabelKey,
    /// Each label, with the color of the map under it for its backing.
    pub(crate) labels: Vec<(ContourLabel, Color)>,
}

#[derive(Default)]
pub(crate) struct LabelMailbox {
    pub(crate) request: Option<LabelJob>,
    pub(crate) result: Option<Placed>,
}

/// Places labels on a background thread, latest request wins.
#[derive(Resource, Clone)]
pub(crate) struct Labeler(pub(crate) Arc<(Mutex<LabelMailbox>, Condvar)>);

impl Labeler {
    pub(crate) fn spawn() -> Self {
        let labeler = Self(Arc::default());
        let shared = labeler.0.clone();
        std::thread::spawn(move || {
            loop {
                let job = {
                    let (lock, ready) = &*shared;
                    let mut mail = lock.lock().unwrap();
                    while mail.request.is_none() {
                        mail = ready.wait(mail).unwrap();
                    }
                    mail.request.take().unwrap()
                };
                let placed = place(job);
                shared.0.lock().unwrap().result = Some(placed);
            }
        });
        labeler
    }

    fn send(&self, job: LabelJob) {
        let (lock, ready) = &*self.0;
        lock.lock().unwrap().request = Some(job);
        ready.notify_one();
    }

    fn take(&self) -> Option<Placed> {
        self.0.0.lock().unwrap().result.take()
    }
}

/// Traces the labelled contours of the terrain at the job's stage on a coarse grid, then places
/// labels on them, snapped onto the exact surface.
fn place(job: LabelJob) -> Placed {
    let LabelKey {
        stage, view, size, ..
    } = job.key.clone();
    let lo = stage.floor().clamp(0., steps() as f64) as u32;
    let p = &job.world.params;
    let staged = Staged {
        baseline: p.mean_elevation,
        plates: &job.plates,
        provinces: &job.provinces,
        chunks: &job.chunks,
        lo,
        hi: (lo + 1).min(steps()),
        t: stage - lo as f64,
        chunk_size: p.chunk_size,
        province_size: p.province_size,
        plate_size: p.tectonics.plate_size,
    };

    let half = Vector2::new(size.0, size.1) * view.m_per_px / 2.;
    let area = Aabb::new(view.center - half, view.center + half);
    let grid = HeightGrid::sample(&staged, area, view.m_per_px * 4.);
    let scale = map_style(view.m_per_px).contours;
    let (lo, hi) = grid.range();
    let lines: Vec<_> = contours(&grid, levels(lo, hi, scale.labelled()))
        .into_iter()
        .filter(|l| l.level > 0.)
        .collect();
    let px = |n: f64| n * view.m_per_px;
    let spacing = LabelSpacing {
        along: px(350.),
        apart: px(70.),
        text: px(LABEL_SIZE as f64),
    };
    let rules = BiomeRules::default();
    let labels = place_labels(&lines, &staged, scale.finest(), spacing)
        .into_iter()
        .map(|label| {
            let [r, g, b] = raster::fill(rules.at(&staged, label.position)).map(|c| c as u8);
            (label, Color::srgb_u8(r, g, b))
        })
        .collect();
    Placed {
        key: job.key,
        labels,
    }
}

/// A label's text entity: its place in the world, and what it was placed for.
#[derive(Component)]
pub(crate) struct LabelText {
    pub(crate) position: Point2<f64>,
    pub(crate) angle: f32,
    pub(crate) key: LabelKey,
}

/// The backing that cuts a gap in the contour under a label.
#[derive(Component)]
pub(crate) struct LabelBacking;

/// Asks for new labels once the view has settled on something they weren't placed for.
pub(crate) fn request_labels(
    controls: Res<Controls>,
    view: Res<View>,
    shown: Option<Res<Shown>>,
    labeler: Res<Labeler>,
    window: Single<&Window, With<PrimaryWindow>>,
    time: Res<Time>,
    mut sent: Local<Option<LabelKey>>,
) {
    let Some(shown) = shown else { return };
    if time.elapsed_secs() - controls.last_interaction < SETTLE {
        return;
    }
    let key = LabelKey {
        seed: controls.seed.clone(),
        stage: controls.stage,
        view: *view,
        size: (window.width() as f64, window.height() as f64),
    };
    if sent.as_ref() == Some(&key) && !shown.is_changed() {
        return;
    }
    labeler.send(LabelJob {
        key: key.clone(),
        world: shown.world.clone(),
        chunks: shown.chunks.clone(),
        provinces: shown.provinces.clone(),
        plates: shown.plates.clone(),
    });
    *sent = Some(key);
}

/// Replaces the labels with newly placed ones.
pub(crate) fn receive_labels(
    mut commands: Commands,
    labeler: Res<Labeler>,
    old: Query<Entity, With<LabelText>>,
) {
    let Some(placed) = labeler.take() else { return };
    for entity in &old {
        commands.entity(entity).despawn();
    }
    for (label, backing) in placed.labels {
        let d = label.direction;
        commands
            .spawn((
                Text2d::new(label.text()),
                TextFont::from_font_size(LABEL_SIZE),
                TextColor(LABEL_COLOR),
                Visibility::Hidden,
                LabelText {
                    position: label.position,
                    angle: d.y.atan2(d.x) as f32,
                    key: placed.key.clone(),
                },
            ))
            .with_child((
                Sprite::from_color(backing, Vec2::ZERO),
                Transform::from_xyz(0., 0., -0.5),
                LabelBacking,
            ));
    }
}

/// Keeps labels anchored to the world as the view moves, and hides them while they're stale
/// (the seed or stage has changed, or they're for another window size).
pub(crate) fn place_label_text(
    controls: Res<Controls>,
    view: Res<View>,
    mut labels: Query<(&LabelText, &mut Transform, &mut Visibility)>,
) {
    for (label, mut transform, mut visibility) in &mut labels {
        let current = label.key.seed == controls.seed && label.key.stage == controls.stage;
        *visibility = if current {
            Visibility::Visible
        } else {
            Visibility::Hidden
        };
        *transform = Transform::from_translation(view.screen(label.position).extend(5.))
            .with_rotation(Quat::from_rotation_z(label.angle));
    }
}

/// Sizes each label's backing to its text, once laid out.
pub(crate) fn fit_label_backing(
    texts: Query<(&TextLayoutInfo, &Children), (With<LabelText>, Changed<TextLayoutInfo>)>,
    mut backings: Query<&mut Sprite, With<LabelBacking>>,
) {
    for (info, children) in &texts {
        let size = info.size / info.scale_factor.max(1e-3) + Vec2::new(4., 0.);
        for child in children.iter() {
            if let Ok(mut sprite) = backings.get_mut(child) {
                sprite.custom_size = Some(size);
            }
        }
    }
}
