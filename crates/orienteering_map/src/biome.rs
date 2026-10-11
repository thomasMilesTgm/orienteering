//! Classifying terrain into biomes by altitude and slope.
//!
//! Classification works per sample: [`BiomeRules`] maps an altitude and slope to a [`Biome`].
//! Raw per-sample classes are noisy along boundaries, so a [`BiomeMap`] can then be cleaned up into
//! coherent regions: [`BiomeMap::smooth`] removes speckle, and [`BiomeMap::absorb_small`] merges
//! regions below a minimum mappable area into their surroundings.

use crate::grid::HeightGrid;
use nalgebra::{Point2, Vector2};
use orienteering_field::Surface;
use orienteering_math::Aabb;
use rayon::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Biome {
    /// Below the water line.
    Water,
    /// Low, flat, waterlogged ground.
    Marsh,
    /// Open, gentle ground.
    Meadow,
    Forest,
    /// Ground too steep for vegetation: crags, scree, cliffs.
    Rock,
    /// High ground above the tree line.
    Alpine,
}

impl Biome {
    pub const ALL: [Self; 6] = [
        Self::Water,
        Self::Marsh,
        Self::Meadow,
        Self::Forest,
        Self::Rock,
        Self::Alpine,
    ];

    fn index(self) -> usize {
        self as usize
    }
}

/// A half-open interval `[min, max)`; a missing bound is unbounded.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Band {
    pub min: Option<f64>,
    pub max: Option<f64>,
}

impl Band {
    pub const ANY: Self = Self {
        min: None,
        max: None,
    };

    pub fn below(max: f64) -> Self {
        Self {
            min: None,
            max: Some(max),
        }
    }

    pub fn at_least(min: f64) -> Self {
        Self {
            min: Some(min),
            max: None,
        }
    }

    pub fn between(min: f64, max: f64) -> Self {
        Self {
            min: Some(min),
            max: Some(max),
        }
    }

    pub fn contains(&self, v: f64) -> bool {
        self.min.is_none_or(|min| v >= min) && self.max.is_none_or(|max| v < max)
    }
}

/// Slope (rise over run, as used by [`BiomeRule::slope`]) of an incline of `degrees`.
pub fn slope_of_degrees(degrees: f64) -> f64 {
    degrees.to_radians().tan()
}

/// Matches samples whose altitude (meters) and slope (`|∇h|`, rise over run) both fall within
/// the given bands.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BiomeRule {
    pub biome: Biome,
    pub altitude: Band,
    pub slope: Band,
}

impl BiomeRule {
    pub fn new(biome: Biome, altitude: Band, slope: Band) -> Self {
        Self {
            biome,
            altitude,
            slope,
        }
    }
}

/// An ordered list of rules: a sample gets the biome of the first rule it matches, or `fallback`.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BiomeRules {
    pub rules: Vec<BiomeRule>,
    pub fallback: Biome,

    /// Half-width (meters) of the square over which slope is averaged before classifying; 0 uses
    /// the slope at the point itself.
    ///
    /// Land cover responds to the lie of the land over a few meters, not to the exact gradient at
    /// a point. Without averaging, every summit, hollow and saddle is momentarily flat and becomes
    /// a speck of flat-ground biome.
    pub slope_scale: f64,
}

impl Default for BiomeRules {
    /// Rules suited to the relief produced by the default terrain parameters: smooth lowlands
    /// sink into basins below the water line, and only the high mountain ranges rise above the
    /// 280 m tree line.
    fn default() -> Self {
        use Biome::*;
        Self {
            rules: vec![
                BiomeRule::new(Water, Band::below(0.), Band::ANY),
                BiomeRule::new(Rock, Band::ANY, Band::at_least(slope_of_degrees(17.))),
                BiomeRule::new(Alpine, Band::at_least(280.), Band::ANY),
                BiomeRule::new(Marsh, Band::below(3.), Band::below(slope_of_degrees(1.5))),
                BiomeRule::new(Meadow, Band::ANY, Band::below(slope_of_degrees(2.5))),
            ],
            fallback: Forest,
            slope_scale: 10.,
        }
    }
}

impl BiomeRules {
    pub fn classify(&self, altitude: f64, slope: f64) -> Biome {
        self.rules
            .iter()
            .find(|r| r.altitude.contains(altitude) && r.slope.contains(slope))
            .map_or(self.fallback, |r| r.biome)
    }

    /// The biome at a single point of a surface. The slope is averaged over a 5 × 5 stencil
    /// spanning [`BiomeRules::slope_scale`], approximating what [`BiomeMap::classify`] does on a
    /// grid.
    pub fn at<S: Surface + ?Sized>(&self, surface: &S, p: Point2<f64>) -> Biome {
        let h = surface.height(p);
        if self.slope_scale <= 0. {
            return self.classify(h, surface.gradient(p).norm());
        }
        let offsets = (-2..=2).map(|k| k as f64 * self.slope_scale / 2.);
        let slopes: Vec<f64> = offsets
            .clone()
            .flat_map(|dy| offsets.clone().map(move |dx| Vector2::new(dx, dy)))
            .map(|d| surface.gradient(p + d).norm())
            .collect();
        self.classify(h, slopes.iter().sum::<f64>() / slopes.len() as f64)
    }
}

/// A biome for every sample of a [`HeightGrid`].
#[derive(Debug, Clone, PartialEq)]
pub struct BiomeMap {
    pub origin: Point2<f64>,
    pub spacing: f64,
    pub nx: usize,
    pub ny: usize,
    cells: Vec<Biome>,
}

/// A connected (4-neighbor) area of a single biome.
#[derive(Debug, Clone, PartialEq)]
pub struct BiomeRegion {
    pub biome: Biome,
    pub cells: usize,
    /// Square meters.
    pub area: f64,
    /// Bounds of the region's sample points.
    pub bounds: Aabb,
}

/// The connected regions of a [`BiomeMap`].
#[derive(Debug, Clone)]
pub struct Regions {
    pub regions: Vec<BiomeRegion>,
    /// For each cell, the index of its region in `regions`.
    pub labels: Vec<usize>,
}

impl BiomeMap {
    /// Builds a map from row-major `cells`.
    pub fn new(origin: Point2<f64>, spacing: f64, nx: usize, ny: usize, cells: Vec<Biome>) -> Self {
        assert_eq!(cells.len(), nx * ny, "expected {nx} × {ny} cells");
        Self {
            origin,
            spacing,
            nx,
            ny,
            cells,
        }
    }

    /// Classifies every sample of `grid`, using exact gradients if the grid has them, averaged
    /// over [`BiomeRules::slope_scale`].
    #[tracing::instrument(level = "info", skip_all, fields(cells = grid.nx * grid.ny))]
    pub fn classify(grid: &HeightGrid, rules: &BiomeRules) -> Self {
        let (nx, ny) = (grid.nx, grid.ny);
        let mut slopes = vec![0.; nx * ny];
        slopes.par_chunks_mut(nx).enumerate().for_each(|(j, row)| {
            for (i, s) in row.iter_mut().enumerate() {
                *s = grid.gradient(i, j).norm();
            }
        });
        let radius = (rules.slope_scale / grid.spacing).round() as usize;
        if radius > 0 {
            slopes = box_mean(&slopes, nx, ny, radius);
        }

        let mut cells = vec![rules.fallback; nx * ny];
        cells.par_chunks_mut(nx).enumerate().for_each(|(j, row)| {
            for (i, cell) in row.iter_mut().enumerate() {
                *cell = rules.classify(grid.get(i, j), slopes[j * nx + i]);
            }
        });
        Self::new(grid.origin, grid.spacing, nx, ny, cells)
    }

    pub fn get(&self, i: usize, j: usize) -> Biome {
        self.cells[j * self.nx + i]
    }

    pub fn cells(&self) -> &[Biome] {
        &self.cells
    }

    pub fn point(&self, i: usize, j: usize) -> Point2<f64> {
        self.origin + Vector2::new(i as f64, j as f64) * self.spacing
    }

    /// The fraction of the map covered by each biome, in [`Biome::ALL`] order.
    pub fn coverage(&self) -> [(Biome, f64); Biome::ALL.len()] {
        let mut counts = [0usize; Biome::ALL.len()];
        for c in &self.cells {
            counts[c.index()] += 1;
        }
        Biome::ALL.map(|b| (b, counts[b.index()] as f64 / self.cells.len() as f64))
    }

    /// Replaces each cell with the most common biome within `radius` cells of it (a majority
    /// filter), keeping its own biome on ties. Removes speckle and smooths ragged boundaries.
    #[tracing::instrument(level = "info", skip(self))]
    pub fn smooth(&mut self, radius: usize) {
        let (nx, ny) = (self.nx, self.ny);
        let mut out = self.cells.clone();
        out.par_chunks_mut(nx).enumerate().for_each(|(j, row)| {
            for (i, cell) in row.iter_mut().enumerate() {
                let mut counts = [0u32; Biome::ALL.len()];
                for y in j.saturating_sub(radius)..=(j + radius).min(ny - 1) {
                    for x in i.saturating_sub(radius)..=(i + radius).min(nx - 1) {
                        counts[self.cells[y * nx + x].index()] += 1;
                    }
                }
                let mut best = *cell;
                for b in Biome::ALL {
                    if counts[b.index()] > counts[best.index()] {
                        best = b;
                    }
                }
                *cell = best;
            }
        });
        self.cells = out;
    }

    /// Labels the map's connected (4-neighbor) regions.
    #[tracing::instrument(level = "info", skip(self), fields(regions))]
    pub fn regions(&self) -> Regions {
        const UNLABELED: usize = usize::MAX;
        let mut labels = vec![UNLABELED; self.cells.len()];
        let mut regions = vec![];
        let mut stack = vec![];

        for start in 0..self.cells.len() {
            if labels[start] != UNLABELED {
                continue;
            }
            let label = regions.len();
            let biome = self.cells[start];
            let mut cells = 0;
            let (mut lo, mut hi) = ((usize::MAX, usize::MAX), (0, 0));

            labels[start] = label;
            stack.push(start);
            while let Some(c) = stack.pop() {
                let (i, j) = (c % self.nx, c / self.nx);
                cells += 1;
                lo = (lo.0.min(i), lo.1.min(j));
                hi = (hi.0.max(i), hi.1.max(j));

                for n in self.neighbors(i, j) {
                    if labels[n] == UNLABELED && self.cells[n] == biome {
                        labels[n] = label;
                        stack.push(n);
                    }
                }
            }

            regions.push(BiomeRegion {
                biome,
                cells,
                area: cells as f64 * self.spacing * self.spacing,
                bounds: Aabb::new(self.point(lo.0, lo.1), self.point(hi.0, hi.1)),
            });
        }

        tracing::Span::current().record("regions", regions.len());
        Regions { regions, labels }
    }

    /// Merges every region smaller than `min_area` (square meters) into the biome it borders
    /// most, repeating until none remain (or, in pathological cases, a few passes have run).
    #[tracing::instrument(level = "info", skip(self))]
    pub fn absorb_small(&mut self, min_area: f64) {
        for _ in 0..8 {
            let Regions { regions, labels } = self.regions();
            let small: Vec<bool> = regions.iter().map(|r| r.area < min_area).collect();
            if !small.contains(&true) {
                return;
            }

            // How often each small region borders each other biome.
            let mut votes = vec![[0u32; Biome::ALL.len()]; regions.len()];
            for (c, &label) in labels.iter().enumerate() {
                if small[label] {
                    for n in self.neighbors(c % self.nx, c / self.nx) {
                        if labels[n] != label {
                            votes[label][self.cells[n].index()] += 1;
                        }
                    }
                }
            }

            let into: Vec<Option<Biome>> = votes
                .iter()
                .map(|v| {
                    Biome::ALL
                        .into_iter()
                        .filter(|b| v[b.index()] > 0)
                        .max_by_key(|b| v[b.index()])
                })
                .collect();

            let mut changed = false;
            for (cell, &label) in self.cells.iter_mut().zip(&labels) {
                if small[label]
                    && let Some(b) = into[label]
                {
                    changed |= *cell != b;
                    *cell = b;
                }
            }
            if !changed {
                return;
            }
        }
    }

    fn neighbors(&self, i: usize, j: usize) -> impl Iterator<Item = usize> + use<> {
        let (nx, ny) = (self.nx, self.ny);
        [
            (i > 0).then(|| j * nx + i - 1),
            (i + 1 < nx).then(|| j * nx + i + 1),
            (j > 0).then(|| (j - 1) * nx + i),
            (j + 1 < ny).then(|| (j + 1) * nx + i),
        ]
        .into_iter()
        .flatten()
    }
}

/// The mean of `values` over the `(2r + 1)²` square around each cell (clipped at the edges),
/// using a summed-area table.
fn box_mean(values: &[f64], nx: usize, ny: usize, r: usize) -> Vec<f64> {
    // sum[(j, i)] is the sum of values[..j, ..i], with a zero row and column in front.
    let w = nx + 1;
    let mut sum = vec![0.; w * (ny + 1)];
    for j in 0..ny {
        let mut row = 0.;
        for i in 0..nx {
            row += values[j * nx + i];
            sum[(j + 1) * w + i + 1] = sum[j * w + i + 1] + row;
        }
    }

    let mut out = vec![0.; nx * ny];
    out.par_chunks_mut(nx).enumerate().for_each(|(j, row)| {
        let (j0, j1) = (j.saturating_sub(r), (j + r + 1).min(ny));
        for (i, v) in row.iter_mut().enumerate() {
            let (i0, i1) = (i.saturating_sub(r), (i + r + 1).min(nx));
            let total = sum[j1 * w + i1] - sum[j0 * w + i1] - sum[j1 * w + i0] + sum[j0 * w + i0];
            *v = total / ((j1 - j0) * (i1 - i0)) as f64;
        }
    });
    out
}

#[cfg(test)]
mod test {
    use super::Biome::*;
    use super::*;
    use orienteering_field::Field;

    /// A map from rows of characters, top row first: `~` water, `m` marsh, `.` meadow,
    /// `f` forest, `#` rock, `^` alpine.
    fn map(rows: &[&str]) -> BiomeMap {
        let (nx, ny) = (rows[0].len(), rows.len());
        let cells = rows
            .iter()
            .rev()
            .flat_map(|r| r.chars())
            .map(|c| match c {
                '~' => Water,
                'm' => Marsh,
                '.' => Meadow,
                'f' => Forest,
                '#' => Rock,
                '^' => Alpine,
                _ => panic!("unknown biome {c:?}"),
            })
            .collect();
        BiomeMap::new(Point2::origin(), 1., nx, ny, cells)
    }

    #[test]
    fn bands_are_half_open() {
        let b = Band::between(1., 2.);
        assert!(!b.contains(0.999) && b.contains(1.) && b.contains(1.999) && !b.contains(2.));
        assert!(Band::ANY.contains(f64::MAX) && Band::below(0.).contains(f64::MIN));
    }

    #[test]
    fn first_matching_rule_wins() {
        let rules = BiomeRules::default();
        assert_eq!(rules.classify(-1., 2.), Water);
        assert_eq!(rules.classify(40., 2.), Rock);
        assert_eq!(rules.classify(200., 2.), Rock);
        assert_eq!(rules.classify(200., 0.1), Forest);
        assert_eq!(rules.classify(400., 0.1), Alpine);
        assert_eq!(rules.classify(1., 0.), Marsh);
        assert_eq!(rules.classify(20., 0.), Meadow);
        assert_eq!(rules.classify(20., 0.2), Forest);
    }

    #[test]
    fn box_mean_matches_brute_force() {
        let (nx, ny, r) = (7, 5, 2);
        let values: Vec<f64> = (0..nx * ny).map(|k| ((k * 37) % 11) as f64).collect();
        let fast = box_mean(&values, nx, ny, r);
        for j in 0..ny {
            for i in 0..nx {
                let mut window = vec![];
                for y in j.saturating_sub(r)..=(j + r).min(ny - 1) {
                    for x in i.saturating_sub(r)..=(i + r).min(nx - 1) {
                        window.push(values[y * nx + x]);
                    }
                }
                let mean = window.iter().sum::<f64>() / window.len() as f64;
                approx::assert_relative_eq!(fast[j * nx + i], mean, epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn summits_are_not_flat_ground() {
        // A gentle 6 m hill: its summit has zero gradient, but the ground around it slopes at
        // ~4° (well short of rock).
        let hill = Field::bump(Point2::origin(), 25., 6.);
        let area = Aabb::centered(Point2::origin(), 100.);
        let grid = HeightGrid::sample_with_gradients(&hill, area, 1.);
        let summit = (grid.nx / 2, grid.ny / 2);
        assert_eq!(grid.point(summit.0, summit.1), Point2::origin());

        let point_slope = BiomeRules {
            slope_scale: 0.,
            ..Default::default()
        };
        let rules = BiomeRules::default();
        assert_eq!(
            BiomeMap::classify(&grid, &point_slope).get(summit.0, summit.1),
            Meadow
        );
        assert_eq!(
            BiomeMap::classify(&grid, &rules).get(summit.0, summit.1),
            Forest
        );
        assert_eq!(rules.at(&hill, Point2::origin()), Forest);
    }

    #[test]
    fn classifies_a_shoreline() {
        // Rising 0.1 m/m eastwards, crossing sea level at x = 0: steep enough for forest.
        let ramp = Field::plane(Point2::origin(), Vector2::new(0.1, 0.));
        let grid = HeightGrid::sample_with_gradients(
            &ramp,
            Aabb::new(Point2::new(-50.5, 0.), Point2::new(49.5, 10.)),
            1.,
        );
        let biomes = BiomeMap::classify(&grid, &BiomeRules::default());
        for i in 0..grid.nx {
            let expected = if grid.point(i, 0).x < 0. {
                Water
            } else {
                Forest
            };
            assert_eq!(biomes.get(i, 5), expected, "at x = {}", grid.point(i, 0).x);
        }
        assert_eq!(biomes.regions().regions.len(), 2);
    }

    #[test]
    fn smoothing_removes_speckle() {
        let mut m = map(&["fffff", "ff#ff", "fffff", "~~~~~", "~~~~~"]);
        m.smooth(1);
        assert_eq!(m, map(&["fffff", "fffff", "fffff", "~~~~~", "~~~~~"]));
    }

    #[test]
    fn labels_connected_regions() {
        let r = map(&["ff~ff", "ff~ff", "~~~~~", "ff~f#"]).regions();
        let mut found: Vec<_> = r.regions.iter().map(|r| (r.biome, r.cells)).collect();
        found.sort();
        assert_eq!(
            found,
            [
                (Water, 8),
                (Forest, 1),
                (Forest, 2),
                (Forest, 4),
                (Forest, 4),
                (Rock, 1)
            ]
        );
        assert_eq!(r.labels[0], r.labels[1]); // Bottom-left forest is one region.
    }

    #[test]
    fn absorbs_small_regions_into_their_surroundings() {
        let mut m = map(&[
            "ffffff~~", //
            "f##fff~~", //
            "f##ff.~~", //
            "ffffff~~", //
        ]);
        m.absorb_small(3.);
        assert_eq!(
            m,
            map(&[
                "ffffff~~", //
                "f##fff~~", //
                "f##fff~~", //
                "ffffff~~", //
            ])
        );
    }
}
