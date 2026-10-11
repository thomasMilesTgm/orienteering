//! The panels: seed, inspector and generation steps.

use super::*;

#[derive(Component)]
pub(crate) struct MapSprite;
#[derive(Component)]
pub(crate) struct GpuQuad;
/// UI that the map shouldn't react to the mouse under.
#[derive(Component)]
pub(crate) struct Panel;
#[derive(Component)]
pub(crate) struct SeedBox;
#[derive(Component)]
pub(crate) struct SeedText;
#[derive(Component)]
pub(crate) struct Track;
#[derive(Component)]
pub(crate) struct TrackFill;
#[derive(Component)]
pub(crate) struct StepLabel;
#[derive(Component)]
pub(crate) struct InfoText;
#[derive(Component)]
pub(crate) struct StatusText;

pub(crate) const PANEL: Color = Color::srgba(0.08, 0.08, 0.1, 0.85);
pub(crate) const ACCENT: Color = Color::srgb(0.95, 0.6, 0.2);
pub(crate) const DIM: Color = Color::srgb(0.65, 0.65, 0.7);

pub(crate) fn setup_ui(mut commands: Commands) {
    let text = |size: f32, color: Color| (TextFont::from_font_size(size), TextColor(color));

    // Seed, top left.
    commands.spawn((
        Node {
            position_type: PositionType::Absolute,
            left: px(12),
            top: px(12),
            padding: px(10).all(),
            flex_direction: FlexDirection::Column,
            row_gap: px(6),
            ..default()
        },
        BackgroundColor(PANEL),
        RelativeCursorPosition::default(),
        Panel,
        children![
            (
                Node {
                    column_gap: px(8),
                    align_items: AlignItems::Center,
                    ..default()
                },
                children![
                    (Text::new("Seed"), text(16., DIM)),
                    (
                        Node {
                            width: px(260),
                            padding: UiRect::axes(px(8), px(4)),
                            border: px(1).all(),
                            ..default()
                        },
                        BackgroundColor(Color::srgb(0.15, 0.15, 0.18)),
                        BorderColor::all(DIM),
                        RelativeCursorPosition::default(),
                        SeedBox,
                        children![(Text::new(DEFAULT_SEED), text(16., Color::WHITE), SeedText)],
                    ),
                ],
            ),
            (
                Text::new("click to edit, Enter to generate"),
                text(12., DIM)
            ),
        ],
    ));

    // Inspector, top right.
    commands.spawn((
        Node {
            position_type: PositionType::Absolute,
            right: px(12),
            top: px(12),
            width: px(420),
            padding: px(10).all(),
            ..default()
        },
        BackgroundColor(PANEL),
        RelativeCursorPosition::default(),
        Panel,
        children![(Text::new(""), text(14., Color::WHITE), InfoText)],
    ));

    // Steps slider, along the bottom.
    let ticks: Vec<_> = (0..=steps()).map(step_name).collect();
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                left: px(12),
                right: px(12),
                bottom: px(12),
                padding: px(10).all(),
                flex_direction: FlexDirection::Column,
                row_gap: px(6),
                ..default()
            },
            BackgroundColor(PANEL),
            RelativeCursorPosition::default(),
            Panel,
        ))
        .with_children(|panel| {
            panel
                .spawn(Node {
                    justify_content: JustifyContent::SpaceBetween,
                    ..default()
                })
                .with_children(|row| {
                    row.spawn((Text::new(""), text(16., ACCENT), StepLabel));
                    row.spawn((Text::new(""), text(14., DIM), StatusText));
                });
            panel
                .spawn((
                    Node {
                        height: px(16),
                        ..default()
                    },
                    BackgroundColor(Color::srgb(0.2, 0.2, 0.24)),
                    RelativeCursorPosition::default(),
                    Track,
                ))
                .with_child((
                    Node {
                        width: percent(100),
                        height: percent(100),
                        ..default()
                    },
                    BackgroundColor(ACCENT),
                    TrackFill,
                ));
            panel
                .spawn(Node {
                    justify_content: JustifyContent::SpaceBetween,
                    ..default()
                })
                .with_children(|row| {
                    for name in ticks {
                        row.spawn((Text::new(name), text(12., DIM)));
                    }
                });
            panel.spawn((
                Text::new(
                    "Left/Right step | Space play | scroll zoom | drag pan | hover to inspect | \
                     G GPU/CPU | T plates | 3 3D view",
                ),
                text(12., DIM),
            ));
        });
}

pub(crate) fn update_seed_ui(
    controls: Res<Controls>,
    time: Res<Time>,
    mut text: Single<&mut Text, With<SeedText>>,
    mut border: Single<&mut BorderColor, With<SeedBox>>,
) {
    text.0 = match &controls.draft {
        // A blinking caret while editing.
        Some(draft) if time.elapsed_secs().fract() < 0.5 => format!("{draft}|"),
        Some(draft) => format!("{draft} "),
        None => controls.seed.clone(),
    };
    **border = BorderColor::all(if controls.draft.is_some() {
        ACCENT
    } else {
        DIM
    });
}

pub(crate) fn update_slider_ui(
    controls: Res<Controls>,
    mut label: Single<&mut Text, With<StepLabel>>,
    mut fill: Single<&mut Node, With<TrackFill>>,
) {
    let stage = controls.stage;
    let (lo, t) = (stage.floor() as u32, stage.fract());
    label.0 = if t > 0.01 {
        format!(
            "{} -> {} fading in ({:.0}%)",
            step_name(lo),
            step_name(lo + 1),
            t * 100.
        )
    } else {
        step_name(stage.round() as u32)
    };
    fill.width = percent((stage / steps() as f64 * 100.) as f32);
}

pub(crate) fn update_status_ui(
    controls: Res<Controls>,
    view: Res<View>,
    cpu: Option<Res<CpuMap>>,
    gpu: Res<GpuMap>,
    time: Res<Time>,
    mut fps: Local<f32>,
    mut status: Single<&mut Text, With<StatusText>>,
) {
    let dt = time.delta_secs().max(1e-4);
    *fps = if *fps == 0. {
        1. / dt
    } else {
        *fps * 0.95 + 0.05 / dt
    };

    let scale = map_style(view.m_per_px).contours;
    let legend = format!(
        "contours {} m, labelled every {} m",
        scale.finest(),
        scale.labelled()
    );
    status.0 = if controls.gpu {
        match &gpu.scene {
            Some(s) => format!(
                "{}GPU | {:.0} fps | ({:.0}, {:.0}) at {:.1} m/px | {} nodes | scene {:.0} ms",
                if s.seed == controls.seed {
                    ""
                } else {
                    "generating... | "
                },
                *fps,
                view.center.x,
                view.center.y,
                view.m_per_px,
                s.nodes,
                s.millis,
            ),
            None => "GPU | generating...".into(),
        }
    } else {
        match &cpu {
            Some(c) => {
                let current = c.job.seed == controls.seed
                    && c.job.stage == controls.stage
                    && c.job.view == *view;
                format!(
                    "{}CPU | ({:.0}, {:.0}) at {:.1} m/px | {}x{} px | {:.0} ms{}",
                    if current { "" } else { "rendering... | " },
                    view.center.x,
                    view.center.y,
                    view.m_per_px,
                    c.texels.x,
                    c.texels.y,
                    c.millis,
                    if c.job.texel > 1. { " | preview" } else { "" }
                )
            }
            None => "CPU | rendering...".into(),
        }
    };
    status.0 = format!("{legend} | {}", status.0);
}
