//! Procedural generation of the world's height field, chunk by chunk.

use super::field::{Aabb, Field, Surface, WINDOW_CUTOFF, Window};
use super::rng::{self, Rng64};
use nalgebra::{Point2, Vector2};
use std::collections::HashMap;
use std::f64::consts::TAU;

/// Tunable parameters for terrain generation.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TerrainParams {
    /// Side length of a chunk, meters.
    pub chunk_size: f64,

    /// Width of each node's window transition, as a fraction of the node's size. Must be at most
    /// `1 / WINDOW_CUTOFF` so a chunk only influences its immediate neighbors.
    pub blend_ratio: f64,

    /// Range of each chunk's base elevation, meters.
    pub base_elevation: [f64; 2],

    /// Typical relief of features at the chunk level, meters.
    pub amplitude: f64,

    /// Relief multiplier per level of sub-chunking.
    pub amplitude_falloff: f64,

    /// Deepest level of sub-chunking.
    pub max_depth: u32,

    /// Range of the number of features generated per node.
    pub features_per_node: [u32; 2],
}

impl Default for TerrainParams {
    fn default() -> Self {
        Self {
            chunk_size: 1000.,
            blend_ratio: 0.12,
            base_elevation: [0., 60.],
            amplitude: 30.,
            amplitude_falloff: 0.45,
            max_depth: 4,
            features_per_node: [2, 4],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ChunkIndex {
    pub x: i64,
    pub y: i64,
}

impl ChunkIndex {
    pub fn new(x: i64, y: i64) -> Self {
        Self { x, y }
    }

    pub fn containing(p: Point2<f64>, chunk_size: f64) -> Self {
        Self::new(
            (p.x / chunk_size).floor() as i64,
            (p.y / chunk_size).floor() as i64,
        )
    }

    /// Chunk `(i, j)` spans `[i, i + 1) × [j, j + 1)` chunk sizes, in world coordinates.
    pub fn bounds(&self, chunk_size: f64) -> Aabb {
        let min = Point2::new(self.x as f64, self.y as f64) * chunk_size;
        Aabb::new(min, min + Vector2::repeat(chunk_size))
    }

    /// This chunk and its eight neighbors.
    pub fn neighborhood(self) -> impl Iterator<Item = Self> {
        (-1..=1).flat_map(move |dy| (-1..=1).map(move |dx| Self::new(self.x + dx, self.y + dy)))
    }
}

/// A generated chunk of terrain. Its field extends a little beyond its bounds, blending into its
/// neighbors.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Chunk {
    pub index: ChunkIndex,

    /// In [0, 1]: the probability each node is subdivided further. Rugged chunks have more,
    /// smaller, higher-frequency features (a proxy for biome: plains → hills → mountains).
    pub ruggedness: f64,

    /// Number of nodes in the chunk's quadtree.
    pub nodes: usize,

    /// Deepest level of sub-chunking reached.
    pub depth: u32,

    pub field: Field,
}

impl Surface for Chunk {
    fn height(&self, p: Point2<f64>) -> f64 {
        self.field.height(p)
    }
    fn height_and_gradient(&self, p: Point2<f64>) -> (f64, Vector2<f64>) {
        self.field.height_and_gradient(p)
    }
}

/// A seed plus parameters: everything needed to generate any chunk, deterministically and in any
/// order.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct World {
    pub seed: u64,
    pub params: TerrainParams,
}

impl World {
    pub fn new(seed: u64, params: TerrainParams) -> Self {
        assert!(
            params.blend_ratio * WINDOW_CUTOFF <= 1.,
            "blend_ratio too large: chunks would reach beyond their neighbors"
        );
        Self { seed, params }
    }

    pub fn from_phrase(phrase: &str, params: TerrainParams) -> Self {
        Self::new(rng::seed_from_phrase(phrase), params)
    }

    pub fn chunk(&self, index: ChunkIndex) -> Chunk {
        let mut rng = Rng64::new(rng::hash(&[self.seed, index.x as u64, index.y as u64]));
        let p = &self.params;

        let ruggedness = rng.unit();
        let base = rng.range(p.base_elevation[0], p.base_elevation[1]);

        let mut nodes = NodeGen {
            rng,
            params: p,
            ruggedness,
            nodes: 0,
            depth: 0,
        };
        let bounds = index.bounds(p.chunk_size);
        let field = nodes.node(bounds, 0, vec![Field::Constant(base)]);

        Chunk {
            index,
            ruggedness,
            nodes: nodes.nodes,
            depth: nodes.depth,
            field,
        }
    }

    /// Generates every chunk needed to evaluate the terrain anywhere within `area`.
    pub fn region(&self, area: Aabb) -> Region {
        let size = self.params.chunk_size;
        let lo = ChunkIndex::containing(area.min, size);
        let hi = ChunkIndex::containing(area.max, size);

        let chunks = (lo.y - 1..=hi.y + 1)
            .flat_map(|y| (lo.x - 1..=hi.x + 1).map(move |x| ChunkIndex::new(x, y)))
            .map(|index| (index, self.chunk(index)))
            .collect();

        Region {
            area,
            chunk_size: size,
            chunks,
        }
    }
}

/// Recursively builds a chunk's quadtree of windowed features.
struct NodeGen<'a> {
    rng: Rng64,
    params: &'a TerrainParams,
    ruggedness: f64,
    nodes: usize,
    depth: u32,
}

impl NodeGen<'_> {
    /// The field of one node: its own features plus its children's, all under its window.
    ///
    /// Features are sized relative to the node, so deeper nodes add shorter wavelengths with
    /// smaller amplitudes, like octaves of noise but localized.
    fn node(&mut self, bounds: Aabb, depth: u32, mut terms: Vec<Field>) -> Field {
        self.nodes += 1;
        self.depth = self.depth.max(depth);

        let p = self.params;
        let size = bounds.width();
        let amplitude = p.amplitude * p.amplitude_falloff.powi(depth as i32);

        let [lo, hi] = p.features_per_node;
        for _ in 0..self.rng.int(lo, hi) {
            terms.push(self.feature(bounds, size, amplitude));
        }

        if depth < p.max_depth && self.rng.chance(self.ruggedness) {
            for quadrant in bounds.quadrants() {
                terms.push(self.node(quadrant, depth + 1, vec![]));
            }
        }

        Field::Sum(terms).masked(Window::new(bounds, size * p.blend_ratio))
    }

    fn feature(&mut self, bounds: Aabb, size: f64, amplitude: f64) -> Field {
        let rng = &mut self.rng;
        let angle = rng.range(0., TAU);
        let direction = Vector2::new(angle.cos(), angle.sin());
        let anywhere = bounds.lerp(rng.unit(), rng.unit());

        match rng.int(0, 2) {
            0 => Field::wave(
                direction,
                rng.range(size / 2., size),
                rng.range(0., TAU),
                amplitude * rng.range(0.2, 0.6),
            ),
            1 => Field::bump(
                anywhere,
                size * rng.range(0.08, 0.25),
                amplitude * rng.range(0.5, 1.5) * rng.sign(),
            ),
            _ => Field::ridge(
                anywhere,
                direction,
                size * rng.range(0.05, 0.15),
                amplitude * rng.range(0.4, 1.0) * rng.sign(),
            ),
        }
    }
}

/// A set of generated chunks, valid for evaluating the terrain within `area`.
#[derive(Debug, Clone)]
pub struct Region {
    pub area: Aabb,
    pub chunk_size: f64,
    pub chunks: HashMap<ChunkIndex, Chunk>,
}

impl Region {
    fn nearby(&self, p: Point2<f64>) -> impl Iterator<Item = &Chunk> {
        ChunkIndex::containing(p, self.chunk_size)
            .neighborhood()
            .filter_map(|i| self.chunks.get(&i))
    }
}

impl Surface for Region {
    fn height(&self, p: Point2<f64>) -> f64 {
        self.nearby(p).map(|c| c.height(p)).sum()
    }

    fn height_and_gradient(&self, p: Point2<f64>) -> (f64, Vector2<f64>) {
        self.nearby(p)
            .map(|c| c.height_and_gradient(p))
            .fold((0., Vector2::zeros()), |(h, g), (ch, cg)| (h + ch, g + cg))
    }
}

#[cfg(test)]
mod test {
    use super::super::field::numeric_gradient;
    use super::*;
    use approx::assert_relative_eq;

    fn world() -> World {
        World::from_phrase("test world", TerrainParams::default())
    }

    #[test]
    fn chunks_are_deterministic_and_order_independent() {
        let w = world();
        let a = w.chunk(ChunkIndex::new(3, -7));
        let _ = w.chunk(ChunkIndex::new(0, 0));
        let b = w.chunk(ChunkIndex::new(3, -7));
        assert_eq!(a, b);
        assert_ne!(a.field, w.chunk(ChunkIndex::new(-7, 3)).field);
    }

    #[test]
    fn terrain_is_smooth_across_chunk_seams() {
        let w = world();
        let region = w.region(Aabb::new(Point2::new(-500., -500.), Point2::new(1500., 1500.)));

        // Walk across the seams at x = 0 and x = 1000 in 1 m steps; neighboring samples must be
        // close and the analytic gradient must agree with finite differences.
        for y in [-300., 0., 250., 999.5] {
            let mut prev = region.height(Point2::new(-50., y));
            for i in -49..=1050 {
                let p = Point2::new(i as f64, y);
                let h = region.height(p);
                assert!((h - prev).abs() < 5., "jump of {} m at {p}", h - prev);
                prev = h;
            }
            for x in [-0.5, 0., 0.5, 1000.] {
                let p = Point2::new(x, y);
                let g = region.gradient(p);
                let n = numeric_gradient(&region, p, 1e-3);
                assert_relative_eq!(g, n, epsilon = 1e-4);
            }
        }
    }
}
