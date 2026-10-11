//! Plate tectonics: the largest scale of the terrain, raising mountain ranges where plates collide.
//!
//! The world is divided into plates: one per cell of a coarse lattice (about 48 km by default),
//! centered at a point jittered within the cell and drifting with a random velocity, all from the
//! seed alone. Plate boundaries are the edges of the plates' Voronoi diagram. Where two plates
//! converge, a mountain range is pushed up along their boundary, as high as they collide fast and
//! head-on; where they diverge, a rift valley sinks; where they slide past each other, little
//! happens.
//!
//! Ranges also raise the ruggedness of the ground around them, so the finer levels of generation,
//! which respond to ruggedness, sculpt them into steep, craggy, deeply channelled mountains: the
//! counterpart to erosion and deposition.
//!
//! Ranges are chains of overlapping [`Field::Range`] pieces with varied heights, widths and
//! offsets, all bent by a gentle, world-wide warp so that they curve rather than following the
//! Voronoi edges' straight lines.

use crate::{CellIndex, World, smoothstep};
use nalgebra::{Point2, Vector2};
use orienteering_field::Field;
use orienteering_math::{
    Aabb,
    rng::{self, Rng64},
};
use std::f64::consts::TAU;

/// Tunable parameters for plate tectonics.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TectonicParams {
    /// Spacing of the plate lattice, meters; plates are roughly this size.
    pub plate_size: f64,
    /// Height of a mountain range where plates converge head-on at full speed, meters.
    pub uplift: f64,
    /// Depth of a rift valley where plates diverge at full speed, meters.
    pub rift: f64,
    /// Width (Gaussian σ) of mountain ranges, meters; rifts are half as wide.
    pub range_width: f64,
    /// Raw ruggedness added along a full-height range (see [`crate::ruggedness`]).
    pub ruggedness: f64,
    /// Approximate length of the pieces a range is built from, meters.
    pub piece_length: f64,
}

impl Default for TectonicParams {
    fn default() -> Self {
        Self {
            plate_size: 48_000.,
            uplift: 220.,
            rift: 50.,
            range_width: 4_000.,
            ruggedness: 1.2,
            piece_length: 12_000.,
        }
    }
}

/// Rings of plates around a point that can influence it: a plate's boundaries lie within about two
/// cells of it, and their ranges reach a little further.
pub const PLATE_REACH: i64 = 3;

const PLATE: u64 = 10;
const BOUNDARY: u64 = 11;
const WARP: u64 = 12;

/// A plate, with the boundaries it owns: those with neighbors that come after it in `(x, y)`
/// order, so each boundary belongs to exactly one plate.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Plate {
    pub index: CellIndex,
    pub center: Point2<f64>,
    /// Drift; `|velocity| ≤ 1`.
    pub velocity: Vector2<f64>,
    pub boundaries: Vec<Boundary>,
    /// The ranges and rifts along the owned boundaries.
    pub elevation: Field,
    /// Raw ruggedness raised along the owned boundaries' ranges.
    pub ruggedness: Field,
}

/// The boundary between two plates: an edge of their Voronoi diagram.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Boundary {
    /// The plate on the other side.
    pub neighbor: CellIndex,
    pub a: Point2<f64>,
    pub b: Point2<f64>,
    /// Closing speed across the boundary: positive where the plates converge, negative where
    /// they diverge.
    pub convergence: f64,
    /// Speed of the plates sliding past each other along the boundary.
    pub shear: f64,
}

impl Boundary {
    pub fn kind(&self) -> BoundaryKind {
        match self.convergence {
            c if c > 0.1 && c >= self.shear.abs() / 2. => BoundaryKind::Convergent,
            c if c < -0.1 && -c >= self.shear.abs() / 2. => BoundaryKind::Divergent,
            _ => BoundaryKind::Transform,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryKind {
    /// Colliding: a mountain range.
    Convergent,
    /// Pulling apart: a rift valley.
    Divergent,
    /// Sliding past each other.
    Transform,
}

impl World {
    /// A plate's center and drift, from the seed alone.
    fn plate_motion(&self, index: CellIndex) -> (Point2<f64>, Vector2<f64>) {
        let size = self.params.tectonics.plate_size;
        let mut rng = Rng64::new(index.seed(self.seed, PLATE));
        let center = index
            .bounds(size)
            .lerp(rng.range(0.15, 0.85), rng.range(0.15, 0.85));
        let angle = rng.range(0., TAU);
        let speed = rng.range(0.2, 1.);
        (center, Vector2::new(angle.cos(), angle.sin()) * speed)
    }

    /// A world-wide warp, bending ranges so they curve rather than follow straight Voronoi edges.
    /// Returns the x and y displacement waves and the most either can displace a point.
    fn tectonic_warp(&self) -> (Field, Field, f64) {
        let size = self.params.tectonics.plate_size;
        let mut rng = Rng64::new(rng::hash(&[self.seed, WARP]));
        let amplitude = 0.08 * size;
        let mut wave = || {
            let angle = rng.range(0., TAU);
            Field::wave(
                Vector2::new(angle.cos(), angle.sin()),
                size * rng.range(0.8, 1.4),
                rng.range(0., TAU),
                amplitude,
            )
        };
        (wave(), wave(), amplitude)
    }

    pub fn plate(&self, index: CellIndex) -> Plate {
        let t = &self.params.tectonics;
        let (center, velocity) = self.plate_motion(index);
        // Neighbors are within two cells; a point on a boundary is within about one and a half,
        // so any plate that could be closer to it, and cut the boundary short, is within three.
        let others: Vec<_> = index
            .within(3)
            .filter(|&j| j != index)
            .map(|j| (j, self.plate_motion(j)))
            .collect();

        let boundaries: Vec<Boundary> = others
            .iter()
            .filter(|(j, _)| (j.x - index.x).abs() <= 2 && (j.y - index.y).abs() <= 2)
            .filter(|(j, _)| (index.x, index.y) < (j.x, j.y))
            .filter_map(|&(j, (cj, vj))| {
                let (a, b) = voronoi_edge(center, cj, others.iter().map(|(_, (c, _))| *c))?;
                let normal = (cj - center).normalize();
                let tangent = Vector2::new(-normal.y, normal.x);
                let closing = velocity - vj;
                Some(Boundary {
                    neighbor: j,
                    a,
                    b,
                    convergence: closing.dot(&normal),
                    shear: closing.dot(&tangent),
                })
            })
            .collect();

        let (dx, dy, warp) = self.tectonic_warp();
        let mut elevation = vec![];
        let mut ruggedness = vec![];
        for boundary in &boundaries {
            let mut rng = Rng64::new(rng::hash(&[
                self.seed,
                BOUNDARY,
                index.x as u64,
                index.y as u64,
                boundary.neighbor.x as u64,
                boundary.neighbor.y as u64,
            ]));
            let up = t.uplift * smoothstep(0.1, 1.2, boundary.convergence);
            let down = t.rift * smoothstep(0.1, 1.2, -boundary.convergence);
            if up.max(down) < 1. {
                continue;
            }
            let (height, width) = if up > 0. {
                (up, t.range_width)
            } else {
                (-down, t.range_width / 2.)
            };
            let rough = t.ruggedness * up / t.uplift;

            // A chain of overlapping pieces, each a little different.
            let along = boundary.b - boundary.a;
            let across = Vector2::new(-along.y, along.x).normalize();
            let pieces = (along.norm() / t.piece_length).ceil().max(1.) as usize;
            let (mut ups, mut roughs, mut support) = (vec![], vec![], None::<Aabb>);
            for k in 0..pieces {
                let overlap = 0.25 / pieces as f64;
                let t0 = k as f64 / pieces as f64 - overlap;
                let t1 = (k + 1) as f64 / pieces as f64 + overlap;
                let offset = across * width * rng.range(-0.4, 0.4);
                let a = boundary.a + along * t0 + offset;
                let b = boundary.a + along * t1 + offset;
                let scale = rng.range(0.6, 1.2);
                let w = width * rng.range(0.7, 1.3);

                ups.push(Field::range(a, b, w, height * scale));
                let mut reach = Field::range_support(a, b, w, height * scale);
                if rough > 0. {
                    // Ruggedness spreads wider than the range itself, into the foothills.
                    roughs.push(Field::range(a, b, w * 1.6, rough * scale));
                    let s = Field::range_support(a, b, w * 1.6, rough * scale);
                    reach = union(reach, s);
                }
                support = Some(support.map_or(reach, |s| union(s, reach)));
            }
            // Warping moves the ranges by up to the warp's amplitude on each axis.
            let support = support.unwrap().grow(warp * 1.5);
            let warped = |pieces: Vec<Field>| {
                Field::Sum(pieces)
                    .warped(dx.clone(), dy.clone())
                    .bounded(support)
            };
            elevation.push(warped(ups));
            if !roughs.is_empty() {
                ruggedness.push(warped(roughs));
            }
        }

        // One support check skips a distant plate entirely.
        let bound_all = |fields: Vec<Field>| {
            let support = fields
                .iter()
                .filter_map(|f| match f {
                    Field::Bounded { support, .. } => Some(*support),
                    _ => None,
                })
                .reduce(union);
            match support {
                Some(s) => Field::Sum(fields).bounded(s),
                None => Field::Sum(vec![]),
            }
        };

        Plate {
            index,
            center,
            velocity,
            boundaries,
            elevation: bound_all(elevation),
            ruggedness: bound_all(ruggedness),
        }
    }

    /// Every plate that can influence the terrain within `area`.
    pub(crate) fn plates(&self, area: Aabb) -> Vec<Plate> {
        CellIndex::covering(area, self.params.tectonics.plate_size, PLATE_REACH)
            .map(|i| self.plate(i))
            .collect()
    }

    /// The plate whose territory `p` is in: the one with the nearest center.
    pub fn plate_at(&self, p: Point2<f64>) -> CellIndex {
        CellIndex::containing(p, self.params.tectonics.plate_size)
            .within(2)
            .min_by(|&i, &j| {
                let d = |k: CellIndex| (self.plate_motion(k).0 - p).norm_squared();
                d(i).total_cmp(&d(j))
            })
            .unwrap()
    }
}

/// The Voronoi edge between the cells of `ci` and `cj`, if they're neighbors: their bisector,
/// clipped to where no other center (in `others`) is closer.
fn voronoi_edge(
    ci: Point2<f64>,
    cj: Point2<f64>,
    others: impl Iterator<Item = Point2<f64>>,
) -> Option<(Point2<f64>, Point2<f64>)> {
    let mid = nalgebra::center(&ci, &cj);
    let normal = cj - ci;
    let dir = Vector2::new(-normal.y, normal.x).normalize();
    let (mut lo, mut hi) = (-1e9, 1e9);
    for ck in others.filter(|&ck| ck != cj) {
        // Closer to ci than to ck: (p - mid_ik) · (ck - ci) ≤ 0, along p = mid + t dir.
        let n = ck - ci;
        let a = dir.dot(&n);
        let b = (mid - nalgebra::center(&ci, &ck)).dot(&n);
        if a.abs() < 1e-12 {
            if b > 0. {
                return None;
            }
        } else if a > 0. {
            hi = f64::min(hi, -b / a);
        } else {
            lo = f64::max(lo, -b / a);
        }
    }
    (hi - lo > 1.).then(|| (mid + dir * lo, mid + dir * hi))
}

fn union(a: Aabb, b: Aabb) -> Aabb {
    Aabb::new(
        Point2::new(a.min.x.min(b.min.x), a.min.y.min(b.min.y)),
        Point2::new(a.max.x.max(b.max.x), a.max.y.max(b.max.y)),
    )
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::TerrainParams;
    use orienteering_field::{Surface, WINDOW_TOLERANCE};

    fn world() -> World {
        World::from_phrase("tectonics", TerrainParams::default())
    }

    #[test]
    fn plates_are_deterministic() {
        let w = world();
        assert_eq!(
            w.plate(CellIndex::new(2, -1)),
            w.plate(CellIndex::new(2, -1))
        );
    }

    #[test]
    fn boundaries_are_shared_voronoi_edges() {
        let w = world();
        let size = w.params.tectonics.plate_size;
        let mut boundaries = 0;
        for i in CellIndex::new(0, 0).within(2) {
            let plate = w.plate(i);
            for b in &plate.boundaries {
                boundaries += 1;
                let neighbor = w.plate_motion(b.neighbor).0;
                // Every point of the edge is equidistant from both centers, and no other plate is
                // closer. (The ends are junctions, where a third plate is just as close.)
                for t in [0., 0.25, 0.5, 0.75, 1.] {
                    let p = b.a + (b.b - b.a) * t;
                    let (di, dj) = ((p - plate.center).norm(), (p - neighbor).norm());
                    approx::assert_relative_eq!(di, dj, epsilon = 1e-6 * size);
                    let nearest = (p - w.plate_motion(w.plate_at(p)).0).norm();
                    assert!(nearest >= di - 1e-6 * size, "{p} is nearer another plate");
                }
            }
        }
        // A plate has about six neighbors, each boundary owned by one of them.
        assert!(boundaries > 25 * 2, "only {boundaries} boundaries");
    }

    #[test]
    fn ranges_rise_where_plates_converge() {
        let w = world();
        let t = &w.params.tectonics;
        let (mut converging, mut diverging) = (vec![], vec![]);
        for i in CellIndex::new(0, 0).within(3) {
            let plate = w.plate(i);
            for b in &plate.boundaries {
                // Away from the warp's shift, sample across the boundary's middle.
                let mid = nalgebra::center(&b.a, &b.b);
                let peak = (-6..=6)
                    .map(|k| {
                        let across = (b.b - b.a).normalize();
                        let p = mid + Vector2::new(-across.y, across.x) * (k as f64 * 1000.);
                        plate.elevation.height(p)
                    })
                    .fold(f64::NEG_INFINITY, f64::max);
                match b.kind() {
                    BoundaryKind::Convergent => converging.push(peak),
                    BoundaryKind::Divergent => diverging.push(peak),
                    BoundaryKind::Transform => {}
                }
            }
        }
        let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
        assert!(converging.len() > 3 && diverging.len() > 3);
        assert!(mean(&converging) > 0.3 * t.uplift, "ranges {converging:?}");
        assert!(mean(&diverging) < 0.1 * t.uplift, "rifts {diverging:?}");
    }

    #[test]
    fn plate_fields_vanish_outside_their_support() {
        let w = world();
        let plate = w.plate(CellIndex::new(1, 1));
        let Field::Bounded { support, field } = &plate.elevation else {
            panic!("plate elevation is bounded");
        };
        let outside = support.max + Vector2::new(1., 1.);
        assert_eq!(plate.elevation.height(outside), 0.);
        // Just inside the support, what's dropped beyond it is negligible.
        let edge = Point2::new(support.max.x - 1e-6, support.center().y);
        assert!(field.height(edge).abs() < 10. * WINDOW_TOLERANCE);
    }
}
