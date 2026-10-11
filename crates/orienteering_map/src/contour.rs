//! Extracting contour lines (level sets of the height field) with marching squares.

use crate::grid::HeightGrid;
use nalgebra::{Point2, Vector2};
use orienteering_field::Surface;
use rayon::prelude::*;
use std::collections::HashMap;

/// A polyline along which the height equals `level`.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Contour {
    pub level: f64,
    pub points: Vec<Point2<f64>>,
    /// Closed contours form a loop (the last point connects back to the first); open ones end at
    /// the edge of the sampled area.
    pub closed: bool,
}

impl Contour {
    /// Snaps each vertex onto the exact level set with Newton steps along the gradient, removing
    /// the error from linear interpolation between grid samples.
    #[tracing::instrument(
        level = "debug",
        skip_all,
        fields(level = self.level, points = self.points.len(), iterations = iterations)
    )]
    pub fn refine<S: Surface + ?Sized>(&mut self, surface: &S, iterations: usize) {
        for p in &mut self.points {
            for _ in 0..iterations {
                let (h, g) = surface.height_and_gradient(*p);
                let g2 = g.norm_squared();
                if g2 < 1e-12 {
                    break;
                }
                *p -= g * ((h - self.level) / g2);
            }
        }
    }
}

/// [`Contour::refine`]s every contour, in parallel.
#[tracing::instrument(level = "info", skip_all, fields(contours = contours.len()))]
pub fn refine_all<S: Surface + Sync + ?Sized>(
    contours: &mut [Contour],
    surface: &S,
    iterations: usize,
) {
    let span = tracing::Span::current();
    contours.par_iter_mut().for_each(|c| {
        let _parent = span.enter();
        c.refine(surface, iterations);
    });
}

/// Contour intervals, each a multiple of the one before: every contour of a coarser interval is
/// also a contour of every finer one, so zooming in only ever adds lines between those already
/// drawn.
pub const CONTOUR_LADDER: [f64; 10] = [1., 5., 10., 50., 100., 500., 1000., 5000., 10000., 50000.];

/// A contour level's rank: the coarsest interval of the [`CONTOUR_LADDER`] it lies on (so 150 m
/// has rank 50 m, and 200 m has rank 100 m). Rank decides how prominently it's drawn.
pub fn contour_rank(level: f64) -> f64 {
    CONTOUR_LADDER
        .iter()
        .rev()
        .copied()
        .find(|&l| {
            let k = level / l;
            (k - k.round()).abs() < 1e-6
        })
        .unwrap_or(0.)
}

/// Which contours to draw at a map scale, and how heavily: each level fades in, then thickens, as
/// the scale makes its rank more widely spaced on screen. Coarser contours are never dropped when
/// zooming in, only joined by finer ones.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ContourScale {
    /// The height interval at which contours are as close as is legible: finer intervals are
    /// hidden, and coarser ones drawn heavier the coarser they are.
    pub density: f64,
}

impl ContourScale {
    /// Contours legible at `m_per_px`: on moderately steep ground, a few pixels apart.
    pub fn for_scale(m_per_px: f64) -> Self {
        Self {
            density: 2.5 * m_per_px,
        }
    }

    /// The finest interval drawn at all.
    pub fn finest(&self) -> f64 {
        Self::step(self.density, false)
    }

    /// The finest interval whose contours are heavy enough to label with their heights.
    pub fn labelled(&self) -> f64 {
        Self::step(4. * self.density, true)
    }

    fn step(above: f64, inclusive: bool) -> f64 {
        CONTOUR_LADDER
            .into_iter()
            .find(|&l| if inclusive { l >= above } else { l > above })
            .unwrap_or(CONTOUR_LADDER[CONTOUR_LADDER.len() - 1])
    }

    /// How the contour at `level` is drawn: its opacity in [0, 1] and half-width in pixels.
    pub fn weight(&self, level: f64) -> (f64, f64) {
        let r = contour_rank(level) / self.density;
        let t = ((r - 1.) / 1.).clamp(0., 1.);
        let opacity = t * t * (3. - 2. * t);
        let width = (0.6 + 0.45 * r.max(1e-9).log2()).clamp(0.6, 1.8);
        (opacity, width)
    }
}

/// Where to label a contour with its height.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ContourLabel {
    /// On the contour, exactly.
    pub position: Point2<f64>,
    pub level: f64,
    /// Unit vector along the text's baseline, along the contour and reading left to right, so
    /// the text is upright on a north-up map.
    pub direction: Vector2<f64>,
}

impl ContourLabel {
    /// The text to label the contour with.
    pub fn text(&self) -> String {
        format!("{:.0}", self.level)
    }
}

/// How to space contour labels, in meters on the ground.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LabelSpacing {
    /// Distance between labels along a contour.
    pub along: f64,
    /// Smallest distance between any two labels.
    pub apart: f64,
    /// Height of the label text: labels are skipped where neighboring contours are closer than
    /// about this, as the text wouldn't fit between them.
    pub text: f64,
}

/// Places height labels along `contours` (typically the index contours), spaced along each line
/// and kept apart from each other, snapped exactly onto their level with Newton steps on
/// `surface`. `interval` is the spacing between all contours, used to avoid crowded places.
pub fn place_labels<S: Surface + ?Sized>(
    contours: &[Contour],
    surface: &S,
    interval: f64,
    spacing: LabelSpacing,
) -> Vec<ContourLabel> {
    let mut labels: Vec<ContourLabel> = vec![];
    for contour in contours {
        let n = contour.points.len();
        let segments = if contour.closed {
            n
        } else {
            n.saturating_sub(1)
        };
        let mut lengths = Vec::with_capacity(segments);
        for k in 0..segments {
            lengths.push((contour.points[(k + 1) % n] - contour.points[k]).norm());
        }
        let total: f64 = lengths.iter().sum();
        if total < spacing.along / 2. {
            continue;
        }

        // Candidates every `along`, starting half way in (and clear of open lines' ends).
        let mut next = spacing.along / 2.;
        let mut walked = 0.;
        for (k, &len) in lengths.iter().enumerate() {
            while next <= walked + len && next <= total - spacing.along / 4. {
                let t = (next - walked) / len.max(1e-12);
                let a = contour.points[k];
                let b = contour.points[(k + 1) % n];
                next += spacing.along;

                let mut on = Contour {
                    level: contour.level,
                    points: vec![a + (b - a) * t],
                    closed: false,
                };
                on.refine(surface, 3);
                let position = on.points[0];
                let (_, g) = surface.height_and_gradient(position);
                let slope = g.norm();
                if slope < 1e-9 || interval / slope < 1.6 * spacing.text {
                    continue;
                }
                if labels
                    .iter()
                    .any(|l| (l.position - position).norm() < spacing.apart)
                {
                    continue;
                }
                let along = Vector2::new(g.y, -g.x) / slope;
                labels.push(ContourLabel {
                    position,
                    level: contour.level,
                    direction: if along.x < 0. { -along } else { along },
                });
            }
            walked += len;
        }
    }
    labels
}

/// Contour levels: every multiple of `interval` within `[min, max]`.
pub fn levels(min: f64, max: f64, interval: f64) -> impl Iterator<Item = f64> {
    let first = (min / interval).ceil() as i64;
    let last = (max / interval).floor() as i64;
    (first..=last).map(move |k| k as f64 * interval)
}

/// A grid edge: horizontal edges run from `(i, j)` to `(i + 1, j)`, vertical from `(i, j)` to
/// `(i, j + 1)`. Contour vertices lie on edges, and adjacent cells share them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Edge {
    H(usize, usize),
    V(usize, usize),
}

/// Extracts all contours of `grid` at each of `levels`, one level per task in parallel.
#[tracing::instrument(level = "info", skip_all, fields(levels, contours))]
pub fn contours(grid: &HeightGrid, levels: impl IntoIterator<Item = f64>) -> Vec<Contour> {
    let levels: Vec<f64> = levels.into_iter().collect();
    let span = tracing::Span::current();
    let lines: Vec<_> = levels
        .par_iter()
        .flat_map_iter(|&level| {
            let _parent = span.enter();
            contours_at(grid, level)
        })
        .collect();
    span.record("levels", levels.len())
        .record("contours", lines.len());
    lines
}

#[tracing::instrument(level = "debug", skip(grid), fields(segments, contours))]
fn contours_at(grid: &HeightGrid, level: f64) -> Vec<Contour> {
    let segments = segments(grid, level);
    tracing::Span::current().record("segments", segments.len());

    let mut adjacent: HashMap<Edge, Vec<usize>> = HashMap::new();
    for (s, &(a, b)) in segments.iter().enumerate() {
        adjacent.entry(a).or_default().push(s);
        adjacent.entry(b).or_default().push(s);
    }

    let mut used = vec![false; segments.len()];
    let walk = |start: Edge, used: &mut [bool]| -> Vec<Edge> {
        let mut path = vec![start];
        let mut at = start;
        while let Some(&s) = adjacent[&at].iter().find(|&&s| !used[s]) {
            used[s] = true;
            let (a, b) = segments[s];
            at = if a == at { b } else { a };
            path.push(at);
        }
        path
    };

    let mut lines = vec![];

    // Open lines start and end on edges touched by only one segment (the grid boundary).
    let mut ends: Vec<Edge> = adjacent
        .iter()
        .filter(|(_, s)| s.len() == 1)
        .map(|(e, _)| *e)
        .collect();
    ends.sort_by_key(|e| match *e {
        Edge::H(i, j) => (0, j, i),
        Edge::V(i, j) => (1, j, i),
    });
    for end in ends {
        if !used[adjacent[&end][0]] {
            lines.push((walk(end, &mut used), false));
        }
    }

    // Everything left forms closed loops.
    for s in 0..segments.len() {
        if !used[s] {
            let mut path = walk(segments[s].0, &mut used);
            path.pop(); // Back at the start.
            lines.push((path, true));
        }
    }

    tracing::Span::current().record("contours", lines.len());
    lines
        .into_iter()
        .map(|(edges, closed)| Contour {
            level,
            points: edges.iter().map(|&e| crossing(grid, e, level)).collect(),
            closed,
        })
        .collect()
}

/// Where the contour crosses `edge`, by linear interpolation.
fn crossing(grid: &HeightGrid, edge: Edge, level: f64) -> Point2<f64> {
    let ((i0, j0), (i1, j1)) = match edge {
        Edge::H(i, j) => ((i, j), (i + 1, j)),
        Edge::V(i, j) => ((i, j), (i, j + 1)),
    };
    let (v0, v1) = (grid.get(i0, j0), grid.get(i1, j1));
    let t = (level - v0) / (v1 - v0);
    let (p0, p1) = (grid.point(i0, j0), grid.point(i1, j1));
    p0 + (p1 - p0) * t
}

/// The marching-squares segments in every cell, as pairs of the edges they join.
fn segments(grid: &HeightGrid, level: f64) -> Vec<(Edge, Edge)> {
    let mut out = vec![];
    for j in 0..grid.ny - 1 {
        for i in 0..grid.nx - 1 {
            // Corners counter-clockwise from bottom-left.
            let c = [
                grid.get(i, j),
                grid.get(i + 1, j),
                grid.get(i + 1, j + 1),
                grid.get(i, j + 1),
            ];
            let case = c
                .iter()
                .enumerate()
                .fold(0, |k, (n, &v)| k | (usize::from(v >= level) << n));

            let (b, r, t, l) = (
                Edge::H(i, j),
                Edge::V(i + 1, j),
                Edge::H(i, j + 1),
                Edge::V(i, j),
            );
            // Disambiguate saddles by the cell's mean height.
            let center_high = c.iter().sum::<f64>() / 4. >= level;

            match case {
                0 | 15 => {}
                1 | 14 => out.push((l, b)),
                2 | 13 => out.push((b, r)),
                3 | 12 => out.push((l, r)),
                4 | 11 => out.push((r, t)),
                6 | 9 => out.push((b, t)),
                7 | 8 => out.push((t, l)),
                5 if center_high => out.extend([(b, r), (t, l)]),
                5 => out.extend([(l, b), (r, t)]),
                10 if center_high => out.extend([(l, b), (r, t)]),
                10 => out.extend([(b, r), (t, l)]),
                _ => unreachable!(),
            }
        }
    }
    out
}

#[cfg(test)]
mod test {
    use super::*;
    use approx::assert_relative_eq;
    use orienteering_field::Field;
    use orienteering_math::Aabb;

    #[test]
    fn hill_gives_closed_circular_contours() {
        let hill = Field::bump(Point2::new(3., -7.), 40., 100.);
        let area = Aabb::centered(Point2::origin(), 300.);
        let grid = HeightGrid::sample(&hill, area, 5.);

        let mut lines = contours(&grid, levels(1., 99., 20.));
        assert_eq!(lines.len(), 4); // 20, 40, 60, 80

        for line in &mut lines {
            assert!(line.closed);
            line.refine(&hill, 3);
            // h = 100 exp(-r² / 2σ²)  ⇒  r = σ √(2 ln(100 / h))
            let r = 40. * (2. * (100. / line.level).ln()).sqrt();
            for p in &line.points {
                assert_relative_eq!((p - Point2::new(3., -7.)).norm(), r, epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn slope_gives_open_contours() {
        let ramp = Field::plane(Point2::origin(), nalgebra::Vector2::new(0.1, 0.));
        let grid = HeightGrid::sample(&ramp, Aabb::centered(Point2::origin(), 100.), 3.);
        let lines = contours(&grid, levels(-4.9, 4.9, 1.));
        assert_eq!(lines.len(), 9);
        assert!(lines.iter().all(|l| !l.closed && l.points.len() > 30));
    }

    #[test]
    fn labels_sit_on_contours_upright() {
        let hill = Field::bump(Point2::new(3., -7.), 120., 100.);
        let area = Aabb::centered(Point2::origin(), 800.);
        let grid = HeightGrid::sample(&hill, area, 4.);
        let lines = contours(&grid, levels(1., 99., 25.));
        let spacing = LabelSpacing {
            along: 300.,
            apart: 40.,
            text: 5.,
        };
        let labels = place_labels(&lines, &hill, 25., spacing);
        assert!(labels.len() >= 4, "{} labels", labels.len());
        for l in &labels {
            approx::assert_relative_eq!(hill.height(l.position), l.level, epsilon = 1e-6);
            // Along the contour (across the slope), reading left to right.
            let g = hill.gradient(l.position).normalize();
            approx::assert_relative_eq!(l.direction.dot(&g), 0., epsilon = 1e-6);
            assert!(l.direction.x >= 0.);
            for other in labels.iter().filter(|o| *o != l) {
                assert!((other.position - l.position).norm() >= 40.);
            }
        }
    }

    #[test]
    fn zooming_in_only_adds_contours() {
        assert_eq!(contour_rank(150.), 50.);
        assert_eq!(contour_rank(200.), 100.);
        assert_eq!(contour_rank(35.), 5.);
        // Every level drawn at a coarse scale is drawn at least as heavily at a finer one.
        let levels: Vec<f64> = (1..=400).map(|k| k as f64).collect();
        let mut previous = ContourScale::for_scale(64.);
        for m_per_px in [32., 16., 8., 4., 2., 1., 0.5] {
            let scale = ContourScale::for_scale(m_per_px);
            for &level in &levels {
                let (a0, w0) = previous.weight(level);
                let (a1, w1) = scale.weight(level);
                assert!(a1 >= a0 && w1 >= w0, "{level} m at {m_per_px} m/px");
            }
            previous = scale;
        }
    }
}
