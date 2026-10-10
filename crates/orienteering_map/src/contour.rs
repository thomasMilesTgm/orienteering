//! Extracting contour lines (level sets of the height field) with marching squares.

use crate::grid::HeightGrid;
use nalgebra::Point2;
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
}
