//! Hover inspection: the point and the quadtree node under the cursor, explained and outlined.

use super::*;

pub(crate) fn inspect(
    controls: Res<Controls>,
    view: Res<View>,
    shown: Option<Res<Shown>>,
    window: Single<&Window, With<PrimaryWindow>>,
    panels: Query<&RelativeCursorPosition, With<Panel>>,
    mut info: Single<&mut Text, With<InfoText>>,
    mut gizmos: Gizmos,
) {
    if controls.three_d {
        info.0 = format!(
            "{}\n\n3D view\n  left-drag orbit | right-drag pan | scroll distance\n  \
             E exaggeration | 3 back to 2D\n  (hover inspection is in 2D)",
            step_name(controls.stage.ceil() as u32).to_uppercase()
        );
        return;
    }
    let mut out = format!(
        "{}\n{}\n",
        step_name(controls.stage.ceil() as u32).to_uppercase(),
        step_description(controls.stage.ceil() as u32)
    );

    let Some(shown) = shown else {
        info.0 = out + "\nGenerating...";
        return;
    };
    let cursor = window
        .cursor_position()
        .filter(|_| !panels.iter().any(|r| r.cursor_over));
    let Some(cursor) = cursor else {
        info.0 = out + "\nHover over the map to inspect it.";
        return;
    };
    let p = view.world(&window, cursor);

    // The point, evaluated exactly at the current stage. (The biome here is the raw rule, before
    // the map's smoothing.)
    let staged = shown.staged(controls.stage);
    let (h, g) = staged.height_and_gradient(p);
    out += &format!(
        "\nAt ({:.0}, {:.0}) m\n  height {:.1} m | slope {:.1} deg | rho {:.2} | {:?}\n",
        p.x,
        p.y,
        h,
        g.norm().atan().to_degrees(),
        shown.world.ruggedness_at(p),
        BiomeRules::default().at(&staged, p),
    );

    // Before any chunk: the plate, and the province once it's fading in.
    if controls.stage <= 1. {
        out += &describe_plate(&shown, p);
        if controls.stage > 0. {
            let province = CellIndex::containing(p, shown.world.params.province_size);
            out += &format!("\nProvince ({}, {})\n", province.x, province.y);
            let bounds = province.bounds(shown.world.params.province_size);
            draw_rect(&mut gizmos, &view, bounds, Color::srgb(0.9, 0.4, 0.1));
        }
        info.0 = out;
        return;
    }
    let depth = controls.stage.ceil() as u32 - 2;
    let index = CellIndex::containing(p, shown.world.params.chunk_size);
    let Some(levels) = shown.chunks.get(&index) else {
        info.0 = out;
        return;
    };
    let traced = &levels.traced;
    let Some(id) = traced.node_at(p, depth) else {
        info.0 = out;
        return;
    };
    let node = &traced.nodes[id];
    out += &describe_node(index, node, depth);

    // Outline the chunk, the node's parent and the node, its downhill direction, and what it
    // added.
    let chunk_bounds = traced.nodes[0].bounds;
    draw_rect(
        &mut gizmos,
        &view,
        chunk_bounds,
        Color::srgba(1., 1., 1., 0.4),
    );
    if let Some(parent) = node.parent {
        draw_rect(
            &mut gizmos,
            &view,
            traced.nodes[parent].bounds,
            Color::srgba(0.1, 0.1, 0.1, 0.5),
        );
    }
    draw_rect(
        &mut gizmos,
        &view,
        node.bounds,
        Color::srgb(0.95, 0.15, 0.6),
    );
    let c = node.bounds.center();
    if node.site.downhill != Vector2::zeros() {
        let tip = c + node.site.downhill * node.bounds.width() * 0.3;
        gizmos.arrow_2d(view.screen(c), view.screen(tip), Color::srgb(0.1, 0.4, 0.9));
    }
    draw_features(&mut gizmos, &view, node);

    info.0 = out;
}

pub(crate) fn describe_node(index: CellIndex, node: &NodeTrace, shown_depth: u32) -> String {
    let s = &node.site;
    let r = &node.response;
    let level = if node.depth == 0 {
        "landform".to_string()
    } else {
        format!("detail {}", node.depth)
    };
    let concavity = match s.concavity {
        c if c > 0.2 => "valley",
        c if c < -0.2 => "spur",
        _ => "even",
    };
    let downhill = if s.downhill == Vector2::zeros() {
        "flat".to_string()
    } else {
        // Compass bearing, north up.
        let bearing = s
            .downhill
            .x
            .atan2(s.downhill.y)
            .to_degrees()
            .rem_euclid(360.);
        format!("{bearing:.0} deg")
    };

    let mut out = format!(
        "\nNode in chunk ({}, {}) | {level} | {:.0} m\n",
        index.x,
        index.y,
        node.bounds.width()
    );
    out += &format!(
        "\nSite: the terrain so far around its center\n  ruggedness {:.2} | slope {:.1} deg | downhill {downhill}\n  concavity {:+.2} ({concavity})\n",
        s.ruggedness,
        s.slope.atan().to_degrees(),
        s.concavity,
    );
    out += &format!(
        "\nResponse\n  activity x{:.2} | steepness x{:.2} | deposition {:.0}%\n",
        r.activity,
        r.steepness,
        r.deposition * 100.,
    );
    out += &match r.channels {
        Some(d) => format!("  channels cut {:.0}% of the node's relief\n", d * 100.),
        None if node.depth == 0 => "  no channels at the landform level\n".into(),
        None => "  no channels: too flat\n".into(),
    };

    let mut counts: [(usize, &str); 6] = [
        (0, "hill"),
        (0, "hollow"),
        (0, "ridge/valley"),
        (0, "scarp"),
        (0, "wave"),
        (0, "channel"),
    ];
    for f in primitives(node) {
        let k = match f {
            Feature::Shape(Field::Bump { height, .. }) if *height >= 0. => 0,
            Feature::Shape(Field::Bump { .. }) => 1,
            Feature::Shape(Field::Ridge { .. }) => 2,
            Feature::Shape(Field::Scarp { .. }) => 3,
            Feature::Shape(_) => 4,
            Feature::Channel(_) => 5,
        };
        counts[k].0 += 1;
    }
    let added: Vec<_> = counts
        .iter()
        .filter(|(n, _)| *n > 0)
        .map(|(n, what)| format!("{n} {what}{}", if *n > 1 { "s" } else { "" }))
        .collect();
    out += &format!(
        "\nAdded (outlined, before warping)\n  {}\n",
        if added.is_empty() {
            "nothing".to_string()
        } else {
            added.join(", ")
        }
    );

    out += &match node.subdivision {
        Some(sub) if node.depth < shown_depth || sub.subdivided => format!(
            "\nSubdivision: {:.0}% chance -> {}\n",
            sub.probability * 100.,
            if sub.subdivided {
                "split into 4"
            } else {
                "kept whole"
            }
        ),
        Some(sub) => format!(
            "\nSubdivision: {:.0}% chance -> kept whole\n",
            sub.probability * 100.
        ),
        None => "\nSubdivision: deepest level\n".into(),
    };
    out
}

pub(crate) enum Feature<'a> {
    /// A warped landform feature.
    Shape(&'a Field),
    /// An unwarped channel cut downhill.
    Channel(&'a Field),
}

/// The primitive features a node added: its warped shapes, then its channels.
pub(crate) fn primitives(node: &NodeTrace) -> Vec<Feature<'_>> {
    let mut out = vec![];
    for f in &node.features {
        match f {
            Field::Warped { field, .. } => {
                if let Field::Sum(shapes) = field.as_ref() {
                    out.extend(shapes.iter().map(Feature::Shape));
                }
            }
            other => out.push(Feature::Channel(other)),
        }
    }
    out
}

pub(crate) fn draw_features(gizmos: &mut Gizmos, view: &View, node: &NodeTrace) {
    let half = node.bounds.width() / 2.;
    let up = Color::srgb(0.75, 0.3, 0.05);
    let down = Color::srgb(0.1, 0.45, 0.85);
    let line = |gizmos: &mut Gizmos, at: Point2<f64>, dir: Vector2<f64>, color: Color| {
        gizmos.line_2d(
            view.screen(at - dir * half),
            view.screen(at + dir * half),
            color,
        );
    };
    for f in primitives(node) {
        match f {
            Feature::Shape(Field::Bump {
                center,
                radius,
                height,
            }) => {
                let color = if *height >= 0. { up } else { down };
                gizmos.circle_2d(
                    view.screen(*center),
                    (*radius / view.m_per_px) as f32,
                    color,
                );
            }
            Feature::Shape(Field::Ridge {
                origin,
                direction,
                height,
                ..
            }) => line(
                gizmos,
                *origin,
                *direction,
                if *height >= 0. { up } else { down },
            ),
            Feature::Shape(Field::Scarp { origin, normal, .. }) => {
                let along = Vector2::new(-normal.y, normal.x);
                line(gizmos, *origin, along, Color::srgb(0.5, 0.1, 0.1));
            }
            Feature::Channel(Field::Ridge {
                origin, direction, ..
            }) => line(gizmos, *origin, *direction, Color::srgb(0., 0.3, 1.)),
            _ => {}
        }
    }
}

pub(crate) fn draw_rect(gizmos: &mut Gizmos, view: &View, bounds: Aabb, color: Color) {
    let center = view.screen(bounds.center());
    let size = Vec2::new(bounds.width() as f32, bounds.height() as f32) / view.m_per_px as f32;
    gizmos.rect_2d(center, size, color);
}

/// The plate under `p`, its motion, and the nearest boundary.
pub(crate) fn describe_plate(shown: &Shown, p: Point2<f64>) -> String {
    let world = &shown.world;
    let index = world.plate_at(p);
    let plate = shown
        .plates
        .get(&index)
        .cloned()
        .unwrap_or_else(|| world.plate(index));
    let v = plate.velocity;
    let bearing = v.x.atan2(v.y).to_degrees().rem_euclid(360.);
    let mut out = format!(
        "\nPlate ({}, {}): drifting {bearing:.0} deg at {:.2}\n",
        index.x,
        index.y,
        v.norm()
    );
    let nearest = shown
        .plates
        .values()
        .flat_map(|plate| plate.boundaries.iter())
        .map(|b| (distance_to_segment(p, b.a, b.b), b))
        .min_by(|a, b| a.0.total_cmp(&b.0));
    if let Some((distance, b)) = nearest {
        let kind = match b.kind() {
            BoundaryKind::Convergent => "converging: a mountain range",
            BoundaryKind::Divergent => "diverging: a rift",
            BoundaryKind::Transform => "sliding past",
        };
        out += &format!(
            "  nearest boundary {:.1} km away, {kind}\n  (closing {:+.2}, sliding {:.2})\n",
            distance / 1000.,
            b.convergence,
            b.shear.abs()
        );
    }
    out
}

fn distance_to_segment(p: Point2<f64>, a: Point2<f64>, b: Point2<f64>) -> f64 {
    let ab = b - a;
    let t = ((p - a).dot(&ab) / ab.norm_squared()).clamp(0., 1.);
    (p - (a + ab * t)).norm()
}

/// Overlays plate boundaries, colored by kind, and plate motion. (Boundaries are drawn straight;
/// the ranges along them are gently warped.)
pub(crate) fn draw_plates(
    controls: Res<Controls>,
    view: Res<View>,
    shown: Option<Res<Shown>>,
    mut gizmos: Gizmos,
) {
    let Some(shown) =
        shown.filter(|_| !controls.three_d && (controls.plates || controls.stage < 1.))
    else {
        return;
    };
    let size = shown.world.params.tectonics.plate_size;
    for plate in shown.plates.values() {
        for b in &plate.boundaries {
            let color = match b.kind() {
                BoundaryKind::Convergent => Color::srgb(0.85, 0.1, 0.1),
                BoundaryKind::Divergent => Color::srgb(0.1, 0.3, 0.9),
                BoundaryKind::Transform => Color::srgb(0.45, 0.45, 0.45),
            };
            gizmos.line_2d(view.screen(b.a), view.screen(b.b), color);
        }
        let tip = plate.center + plate.velocity * size * 0.3;
        gizmos.arrow_2d(
            view.screen(plate.center),
            view.screen(tip),
            Color::srgb(0.15, 0.15, 0.15),
        );
    }
}
