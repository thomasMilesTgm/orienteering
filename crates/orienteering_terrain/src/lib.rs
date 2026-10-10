//! Procedural generation of the world's height field.
//!
//! Terrain is generated coarse to fine, in two sizes of square cell:
//!
//! - **Provinces** (8 km by default) set the broad shape of the land: wide basins, uplands and
//!   ranges, and a smooth *ruggedness* field ρ ∈ [0, 1] saying how mountainous the ground is.
//!   Rugged ground is uplifted, and smooth ground sinks into basins.
//! - **Chunks** (1 km by default) add landforms and then progressively finer detail as a quadtree.
//!
//! Every chunk node is generated after its coarser ancestors and chooses its features from the
//! terrain already generated around its center (its *site*), so the large-scale character of the
//! land shapes the detail:
//!
//! - **Activity**: rugged, steep ground gets taller, narrower features and subdivides further, so
//!   mountains become more mountainous; smooth ground gets subdued features and stays coarse.
//! - **Erosion**: on slopes, channels (narrow valleys) are cut running downhill along the slope
//!   of the terrain so far. Their depth follows a stream-power proxy (slope × concavity: valleys
//!   gather water) and they are narrower on rugged ground, from shallow swales to ravines. Finer
//!   levels follow the slopes of the channels above them, so tributaries run down channel sides
//!   and join at an angle: the drainage branches.
//! - **Deposition**: on flat, smooth ground, especially basins, sediment fills hollows (new
//!   negative features are suppressed) and detail is damped, so plains flatten out.
//!
//! The result is still a static [`Field`]: the terrain is only *consulted* while generating, so
//! evaluation keeps exact gradients and costs no more than before. Generation stays deterministic
//! and order-independent: a chunk's context comes from its provinces, its neighbors' coarsest
//! (landform) level, and its own coarser levels, all of which depend only on the seed.

use nalgebra::{Point2, Vector2};
use orienteering_field::{Field, Surface, WINDOW_CUTOFF, Window};
use orienteering_math::{
    Aabb,
    rng::{self, Rng64},
};
use rayon::prelude::*;
use std::collections::HashMap;
use std::f64::consts::TAU;

/// Tunable parameters for terrain generation.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TerrainParams {
    /// Side length of a chunk, meters.
    pub chunk_size: f64,

    /// Side length of a province, meters.
    pub province_size: f64,

    /// Width of each cell's window transition, as a fraction of the cell's size. Must be at most
    /// `1 / WINDOW_CUTOFF` so a cell only influences its immediate neighbors.
    pub blend_ratio: f64,

    /// Average elevation of the world, meters.
    pub mean_elevation: f64,

    /// Relief of province-scale basins, uplands and ranges, meters.
    pub province_relief: f64,

    /// Extra elevation of fully rugged ground (ρ = 1), meters.
    pub uplift: f64,

    /// Relief of chunk-scale landforms at average activity, meters.
    pub amplitude: f64,

    /// Relief multiplier per level of sub-chunking.
    pub amplitude_falloff: f64,

    /// Deepest level of sub-chunking.
    pub max_depth: u32,

    /// Range of the number of features generated per node.
    pub features_per_node: [u32; 2],

    /// Strength of gully erosion; 0 disables it.
    pub erosion: f64,

    /// Strength of deposition on flat, smooth ground, in [0, 1]; 0 disables it.
    pub deposition: f64,
}

impl Default for TerrainParams {
    fn default() -> Self {
        Self {
            chunk_size: 1000.,
            province_size: 8000.,
            blend_ratio: 0.12,
            mean_elevation: 10.,
            province_relief: 35.,
            uplift: 60.,
            amplitude: 22.,
            amplitude_falloff: 0.4,
            max_depth: 4,
            features_per_node: [1, 3],
            erosion: 1.,
            deposition: 0.8,
        }
    }
}

/// A square cell of the world: a chunk or a province. Cell `(i, j)` of side `size` spans
/// `[i, i + 1) × [j, j + 1)` cell sizes in world coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CellIndex {
    pub x: i64,
    pub y: i64,
}

impl CellIndex {
    pub fn new(x: i64, y: i64) -> Self {
        Self { x, y }
    }

    pub fn containing(p: Point2<f64>, size: f64) -> Self {
        Self::new((p.x / size).floor() as i64, (p.y / size).floor() as i64)
    }

    pub fn bounds(&self, size: f64) -> Aabb {
        let min = Point2::new(self.x as f64, self.y as f64) * size;
        Aabb::new(min, min + Vector2::repeat(size))
    }

    /// This cell and its eight neighbors.
    pub fn neighborhood(self) -> impl Iterator<Item = Self> {
        self.within(1)
    }

    /// The `(2r + 1)²` cells centered on this one.
    pub fn within(self, r: i64) -> impl Iterator<Item = Self> {
        (-r..=r).flat_map(move |dy| (-r..=r).map(move |dx| Self::new(self.x + dx, self.y + dy)))
    }

    /// Every cell of side `size` that touches `area`, plus `ring` rings of neighbors.
    pub fn covering(area: Aabb, size: f64, ring: i64) -> impl Iterator<Item = Self> {
        let lo = Self::containing(area.min, size);
        let hi = Self::containing(area.max, size);
        (lo.y - ring..=hi.y + ring)
            .flat_map(move |y| (lo.x - ring..=hi.x + ring).map(move |x| Self::new(x, y)))
    }

    fn seed(&self, world: u64, salt: u64) -> u64 {
        rng::hash(&[world, salt, self.x as u64, self.y as u64])
    }
}

const PROVINCE: u64 = 1;
const LANDFORM: u64 = 2;
const DETAIL: u64 = 3;

/// The broad shape of an 8 km (by default) cell of the world.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Province {
    pub index: CellIndex,

    /// Raw ruggedness: a Gaussian centered on the province. Summed over provinces, these
    /// interpolate smoothly between the provinces' values; [`ruggedness`] maps the sum to ρ.
    pub ruggedness: Field,

    /// The province's contribution to the height field, including uplift.
    pub elevation: Field,
}

/// Maps the summed raw ruggedness of the provinces to ρ ∈ [0, 1].
pub fn ruggedness(raw: f64) -> f64 {
    smoothstep(0., 1., raw)
}

/// Width of each province's ruggedness Gaussian, relative to the province size. Gaussians on a
/// grid sum to a constant (to within ±2% at this width) so they interpolate smoothly.
const RUGGEDNESS_SIGMA: f64 = 0.45;

/// Rings of provinces around a point that can influence it. The Gaussians' tails reach further
/// than the windows: two rings out they are below 1e-6 of their peak, which is negligible even
/// scaled by uplift (one ring would leave steps of ~0.4% where the neighborhood changes).
const PROVINCE_REACH: i64 = 2;

/// A generated chunk of terrain. Its field extends a little beyond its bounds, blending into its
/// neighbors.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Chunk {
    pub index: CellIndex,

    /// Ruggedness ρ at the chunk's center.
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

/// A seed plus parameters: everything needed to generate any province or chunk,
/// deterministically and in any order.
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
            "blend_ratio too large: cells would reach beyond their neighbors"
        );
        Self { seed, params }
    }

    pub fn from_phrase(phrase: &str, params: TerrainParams) -> Self {
        Self::new(rng::seed_from_phrase(phrase), params)
    }

    pub fn province(&self, index: CellIndex) -> Province {
        let p = &self.params;
        let size = p.province_size;
        let mut rng = Rng64::new(index.seed(self.seed, PROVINCE));
        let bounds = index.bounds(size);
        let window = Window::new(bounds, size * p.blend_ratio);

        // The province's ruggedness, spread smoothly over it and its neighbors. Values beyond
        // [0, 1] saturate ρ over wider areas (and push uplift and subsidence further).
        let sigma = RUGGEDNESS_SIGMA;
        let grid_sum = TAU * sigma * sigma; // Sum of unit Gaussians on a unit grid.
        let ruggedness = Field::bump(
            bounds.center(),
            size * sigma,
            rng.range(-0.5, 1.4) / grid_sum,
        );

        // Broad basins, uplands and ranges, plus uplift of the rugged ground.
        let relief = p.province_relief;
        let mut shape = vec![Field::Constant(
            p.mean_elevation + relief * rng.range(-0.5, 0.5),
        )];
        for _ in 0..rng.int(2, 4) {
            let at = bounds.lerp(rng.unit(), rng.unit());
            let angle = rng.range(0., TAU);
            shape.push(if rng.chance(0.6) {
                Field::bump(
                    at,
                    size * rng.range(0.08, 0.25),
                    relief * rng.range(0.4, 1.) * rng.sign(),
                )
            } else {
                Field::ridge(
                    at,
                    Vector2::new(angle.cos(), angle.sin()),
                    size * rng.range(0.03, 0.08),
                    relief * rng.range(0.3, 0.8) * rng.sign(),
                )
            });
        }
        let elevation = Field::Sum(shape).masked(window) + ruggedness.clone() * p.uplift;

        Province {
            index,
            ruggedness,
            elevation,
        }
    }

    /// Every province that can influence the terrain within `area`.
    fn provinces(&self, area: Aabb) -> Vec<Province> {
        CellIndex::covering(area, self.params.province_size, PROVINCE_REACH)
            .map(|i| self.province(i))
            .collect()
    }

    /// Ruggedness ρ ∈ [0, 1] at `p`.
    pub fn ruggedness_at(&self, p: Point2<f64>) -> f64 {
        let provinces = self.provinces(Aabb::new(p, p));
        ruggedness(provinces.iter().map(|v| v.ruggedness.height(p)).sum())
    }

    pub fn chunk(&self, index: CellIndex) -> Chunk {
        self.chunk_traced(index).chunk
    }

    /// Generates a chunk and keeps the record of how each of its nodes was generated.
    #[tracing::instrument(level = "debug", skip(self), fields(ruggedness, nodes, depth))]
    pub fn chunk_traced(&self, index: CellIndex) -> TracedChunk {
        let p = &self.params;
        let bounds = index.bounds(p.chunk_size);
        let provinces = self.provinces(bounds.grow(p.chunk_size));

        // Everything coarser than this chunk's own detail: provinces and neighboring landforms.
        let mut ctx = Context::new(&provinces);
        let landforms: Vec<Field> = index
            .neighborhood()
            .filter(|&n| n != index)
            .map(|n| self.landform(n, &ctx).own_field(p.blend_ratio))
            .collect();
        ctx.base.extend(landforms);

        let mut tree = Tree::new(self.landform(index, &ctx));
        let mut rng = Rng64::new(index.seed(self.seed, DETAIL));
        let mut frontier = vec![0];

        // Level by level, so each level sees all of the chunk's coarser levels.
        for depth in 1..=p.max_depth {
            ctx.own = tree.field(p.blend_ratio, u32::MAX);
            let mut next = vec![];
            for parent in frontier {
                let bounds = tree.nodes[parent].bounds;
                let site = ctx.site(bounds.center(), bounds.width());
                let probability = respond(p, &site, depth - 1).subdivide;
                let subdivided = rng.chance(probability);
                tree.nodes[parent].subdivision = Some(Subdivision {
                    site,
                    probability,
                    subdivided,
                });
                if !subdivided {
                    continue;
                }
                for quadrant in bounds.quadrants() {
                    let site = ctx.site(quadrant.center(), quadrant.width());
                    let (features, response) = features(&mut rng, p, quadrant, depth, &site);
                    next.push(tree.add(NodeTrace {
                        bounds: quadrant,
                        depth,
                        parent: Some(parent),
                        children: vec![],
                        site,
                        response,
                        features,
                        subdivision: None,
                    }));
                }
            }
            if next.is_empty() {
                break;
            }
            frontier = next;
        }

        let ruggedness = ctx.site(bounds.center(), p.chunk_size).ruggedness;
        let (nodes, depth) = (tree.nodes.len(), tree.depth);
        tracing::Span::current()
            .record("ruggedness", ruggedness)
            .record("nodes", nodes)
            .record("depth", depth);

        TracedChunk {
            chunk: Chunk {
                index,
                ruggedness,
                nodes,
                depth,
                field: tree.field(p.blend_ratio, u32::MAX),
            },
            nodes: tree.nodes,
            blend_ratio: p.blend_ratio,
        }
    }

    /// The coarsest level of a chunk, which depends only on the provinces.
    fn landform(&self, index: CellIndex, provinces: &Context) -> NodeTrace {
        let p = &self.params;
        let bounds = index.bounds(p.chunk_size);
        let mut rng = Rng64::new(index.seed(self.seed, LANDFORM));
        let site = provinces.site(bounds.center(), p.chunk_size);
        let (features, response) = features(&mut rng, p, bounds, 0, &site);
        NodeTrace {
            bounds,
            depth: 0,
            parent: None,
            children: vec![],
            site,
            response,
            features,
            subdivision: None,
        }
    }

    /// Generates every province and chunk needed to evaluate the terrain anywhere within `area`.
    #[tracing::instrument(level = "info", skip(self), fields(provinces, chunks))]
    pub fn region(&self, area: Aabb) -> Region {
        let p = &self.params;
        let provinces: HashMap<_, _> = CellIndex::covering(area, p.province_size, PROVINCE_REACH)
            .map(|i| (i, self.province(i)))
            .collect();
        let indices: Vec<_> = CellIndex::covering(area, p.chunk_size, 1).collect();
        let chunks: HashMap<_, _> = indices
            .into_par_iter()
            .map(|i| (i, self.chunk(i)))
            .collect();
        tracing::Span::current()
            .record("provinces", provinces.len())
            .record("chunks", chunks.len());

        Region {
            area,
            chunk_size: p.chunk_size,
            province_size: p.province_size,
            provinces,
            chunks,
        }
    }
}

/// The terrain generated so far around a chunk.
struct Context {
    ruggedness: Field,
    /// Provinces and (once generated) neighboring landforms.
    base: Vec<Field>,
    /// The chunk's own levels generated so far.
    own: Field,
}

/// What the terrain generated so far looks like around a point: what a node sees before choosing
/// its features.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Site {
    /// ρ ∈ [0, 1].
    pub ruggedness: f64,
    /// `|∇h|`, rise over run.
    pub slope: f64,
    /// Unit vector down the slope (zero on flat ground).
    pub downhill: Vector2<f64>,
    /// In (-1, 1): positive in valleys and basins, negative on spurs and summits, relative to the
    /// slope, at the scale of the node.
    pub concavity: f64,
}

impl Context {
    fn new(provinces: &[Province]) -> Self {
        Self {
            ruggedness: Field::Sum(provinces.iter().map(|v| v.ruggedness.clone()).collect()),
            base: provinces.iter().map(|v| v.elevation.clone()).collect(),
            own: Field::Constant(0.),
        }
    }

    fn height_and_gradient(&self, p: Point2<f64>) -> (f64, Vector2<f64>) {
        self.base
            .iter()
            .chain([&self.own])
            .map(|f| f.height_and_gradient(p))
            .fold((0., Vector2::zeros()), |(h, g), (fh, fg)| (h + fh, g + fg))
    }

    fn height(&self, p: Point2<f64>) -> f64 {
        self.base
            .iter()
            .chain([&self.own])
            .map(|f| f.height(p))
            .sum()
    }

    /// The site at `p`, with curvature measured at the scale of a node of side `size`.
    fn site(&self, p: Point2<f64>, size: f64) -> Site {
        let (h, g) = self.height_and_gradient(p);
        let slope = g.norm();

        let e = size / 4.;
        let around: f64 = [(e, 0.), (-e, 0.), (0., e), (0., -e)]
            .map(|(dx, dy)| self.height(p + Vector2::new(dx, dy)))
            .iter()
            .sum();
        let laplacian = (around - 4. * h) / (e * e);

        Site {
            ruggedness: ruggedness(self.ruggedness.height(p)),
            slope,
            downhill: if slope > 1e-9 {
                -g / slope
            } else {
                Vector2::zeros()
            },
            concavity: (laplacian * size / (slope + 0.05)).tanh(),
        }
    }
}

/// How a node responds to its site.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Response {
    /// Relief multiplier.
    pub activity: f64,
    /// Feature narrowing: steeper flanks on rugged ground.
    pub steepness: f64,
    /// In [0, 1]: how much sediment fills hollows and damps detail.
    pub deposition: f64,
    /// Probability of subdividing further.
    pub subdivide: f64,
    /// Channel depth relative to the node's relief, if any are cut.
    pub channels: Option<f64>,
}

fn respond(p: &TerrainParams, site: &Site, depth: u32) -> Response {
    let rho = site.ruggedness;
    let flat = 1. - smoothstep(0.02, 0.1, site.slope);
    let basin = 0.5 + 0.5 * site.concavity.max(0.);
    let deposition = p.deposition * (1. - rho) * flat * basin;

    // Steep ground gets livelier, but only up to a point: the slope seen here includes every
    // coarser level, so unbounded feedback would compound down the levels into solid cliffs.
    let activity = lerp(0.25, 1.4, rho) * (1. + site.slope.min(0.3)) * (1. - 0.8 * deposition);

    let channels = (p.erosion > 0. && site.slope > 0.04 && depth > 0).then(|| {
        // Stream power: steeper slopes and converging (concave) ground carry more water.
        let power = (site.slope / 0.2).min(1.5) * (1. + 0.5 * site.concavity);
        p.erosion * 0.35 * power * (1. - deposition)
    });

    Response {
        activity,
        steepness: 0.9 + 0.4 * rho,
        deposition,
        // Smooth ground still subdivides (its detail is just weaker), so no shape is left as a
        // bare primitive.
        subdivide: (0.55 + 0.3 * rho + site.slope.min(0.2) - 0.3 * deposition).clamp(0., 0.9),
        channels,
    }
}

/// The features of one node, chosen in response to its site.
fn features(
    rng: &mut Rng64,
    p: &TerrainParams,
    bounds: Aabb,
    depth: u32,
    site: &Site,
) -> (Vec<Field>, Response) {
    let r = respond(p, site, depth);
    let size = bounds.width();
    let relief = p.amplitude * p.amplitude_falloff.powi(depth as i32) * r.activity;
    let rho = site.ruggedness;

    let mut out = vec![];
    let [lo, hi] = p.features_per_node;
    for _ in 0..rng.int(lo, hi) {
        let at = bounds.lerp(rng.unit(), rng.unit());
        let angle = rng.range(0., TAU);
        let direction = Vector2::new(angle.cos(), angle.sin());
        // Sediment fills hollows: negative features shrink with deposition.
        let sign = rng.sign();
        let height = relief * sign * if sign < 0. { 1. - r.deposition } else { 1. };

        // Bumps (hills, hollows), ridges (spurs, valleys), scarps (only on rugged ground, and
        // below the landform level so cliff bands stay short) and occasional long waves.
        let scarps = if depth > 0 { 2. * rho } else { 0. };
        out.push(match rng.pick(&[4., 3., scarps, 0.5]) {
            0 => Field::bump(
                at,
                size * rng.range(0.08, 0.25) / r.steepness,
                height * rng.range(0.5, 1.3),
            ),
            1 => Field::ridge(
                at,
                direction,
                size * rng.range(0.05, 0.15) / r.steepness,
                height * rng.range(0.4, 1.0),
            ),
            2 => Field::scarp(
                at,
                direction,
                size * rng.range(0.02, 0.06) / r.steepness,
                height * rng.range(0.5, 1.0),
            ),
            _ => Field::wave(
                direction,
                rng.range(size / 2., size),
                rng.range(0., TAU),
                height * rng.range(0.2, 0.5),
            ),
        });
    }

    // Bend everything this node adds, so hills and valleys meander rather than being perfect
    // ellipses and straight lines. The warp is smooth at the node's scale.
    let mut warp = || {
        let angle = rng.range(0., TAU);
        Field::wave(
            Vector2::new(angle.cos(), angle.sin()),
            size * rng.range(0.6, 1.2),
            rng.range(0., TAU),
            size * rng.range(0.06, 0.12),
        )
    };
    let (dx, dy) = (warp(), warp());
    let mut out = vec![Field::Sum(out).warped(dx, dy)];

    // Channels run downhill through the node, narrower on rugged ground. They're not warped:
    // they must keep following the slope.
    if let Some(depth) = r.channels {
        for _ in 0..rng.int(1, 2) {
            out.push(Field::ridge(
                bounds.lerp(rng.range(0.2, 0.8), rng.range(0.2, 0.8)),
                site.downhill,
                size * rng.range(0.03, 0.07) / r.steepness,
                -relief * depth * rng.range(0.6, 1.),
            ));
        }
    }
    (out, r)
}

/// How one quadtree node of a chunk was generated.
#[derive(Debug, Clone, PartialEq)]
pub struct NodeTrace {
    pub bounds: Aabb,
    /// 0 for the chunk's landform level.
    pub depth: u32,
    pub parent: Option<usize>,
    pub children: Vec<usize>,
    /// The terrain so far around the node's center when its features were chosen.
    pub site: Site,
    pub response: Response,
    /// The node's own features (before its window is applied).
    pub features: Vec<Field>,
    /// The decision whether to subdivide, if the node was considered for it (nodes at the
    /// deepest level are not).
    pub subdivision: Option<Subdivision>,
}

/// A node's decision whether to subdivide, made once the whole of its level was generated.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Subdivision {
    /// The site at the node's center at that time.
    pub site: Site,
    pub probability: f64,
    pub subdivided: bool,
}

impl NodeTrace {
    /// This node's own features under its window (without children).
    fn own_field(&self, blend_ratio: f64) -> Field {
        let window = Window::new(self.bounds, self.bounds.width() * blend_ratio);
        Field::Sum(self.features.clone()).masked(window)
    }
}

/// A generated chunk together with the record of how it was generated.
#[derive(Debug, Clone, PartialEq)]
pub struct TracedChunk {
    pub chunk: Chunk,
    /// The quadtree's nodes; node 0 is the landform level, and children follow their parents.
    pub nodes: Vec<NodeTrace>,
    blend_ratio: f64,
}

impl TracedChunk {
    /// The chunk's field with only its levels down to `max_depth` (0 = landforms only), i.e. the
    /// chunk as it stood partway through generation.
    pub fn field_to_depth(&self, max_depth: u32) -> Field {
        Tree::subtree_of(&self.nodes, 0, self.blend_ratio, max_depth)
    }

    /// The deepest node, no deeper than `max_depth`, containing `p`.
    pub fn node_at(&self, p: Point2<f64>, max_depth: u32) -> Option<usize> {
        if !self.nodes[0].bounds.contains(p) {
            return None;
        }
        let mut id = 0;
        while let Some(&child) = self.nodes[id]
            .children
            .iter()
            .find(|&&c| self.nodes[c].depth <= max_depth && self.nodes[c].bounds.contains(p))
        {
            id = child;
        }
        Some(id)
    }
}

/// A chunk's quadtree under construction.
struct Tree {
    nodes: Vec<NodeTrace>,
    depth: u32,
}

impl Tree {
    fn new(root: NodeTrace) -> Self {
        Self {
            nodes: vec![root],
            depth: 0,
        }
    }

    fn add(&mut self, node: NodeTrace) -> usize {
        let id = self.nodes.len();
        let parent = node.parent.expect("only the root has no parent");
        self.depth = self.depth.max(node.depth);
        self.nodes.push(node);
        self.nodes[parent].children.push(id);
        id
    }

    /// The nested field of the tree down to `max_depth`: each node's features and children under
    /// its window.
    fn field(&self, blend_ratio: f64, max_depth: u32) -> Field {
        Self::subtree_of(&self.nodes, 0, blend_ratio, max_depth)
    }

    fn subtree_of(nodes: &[NodeTrace], id: usize, blend_ratio: f64, max_depth: u32) -> Field {
        let node = &nodes[id];
        let mut terms = node.features.clone();
        terms.extend(
            node.children
                .iter()
                .filter(|&&c| nodes[c].depth <= max_depth)
                .map(|&c| Self::subtree_of(nodes, c, blend_ratio, max_depth)),
        );
        let window = Window::new(node.bounds, node.bounds.width() * blend_ratio);
        Field::Sum(terms).masked(window)
    }
}

fn smoothstep(e0: f64, e1: f64, x: f64) -> f64 {
    let t = ((x - e0) / (e1 - e0)).clamp(0., 1.);
    t * t * (3. - 2. * t)
}

fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

/// The provinces and chunks covering an area, valid for evaluating the terrain within `area`.
#[derive(Debug, Clone)]
pub struct Region {
    pub area: Aabb,
    pub chunk_size: f64,
    pub province_size: f64,
    pub provinces: HashMap<CellIndex, Province>,
    pub chunks: HashMap<CellIndex, Chunk>,
}

impl Region {
    fn fields(&self, p: Point2<f64>) -> impl Iterator<Item = &Field> {
        let provinces = CellIndex::containing(p, self.province_size)
            .within(PROVINCE_REACH)
            .filter_map(|i| self.provinces.get(&i))
            .map(|v| &v.elevation);
        let chunks = CellIndex::containing(p, self.chunk_size)
            .neighborhood()
            .filter_map(|i| self.chunks.get(&i))
            .map(|c| &c.field);
        provinces.chain(chunks)
    }

    /// Ruggedness ρ ∈ [0, 1] at `p`.
    pub fn ruggedness(&self, p: Point2<f64>) -> f64 {
        ruggedness(
            CellIndex::containing(p, self.province_size)
                .within(PROVINCE_REACH)
                .filter_map(|i| self.provinces.get(&i))
                .map(|v| v.ruggedness.height(p))
                .sum(),
        )
    }
}

impl Surface for Region {
    fn height(&self, p: Point2<f64>) -> f64 {
        self.fields(p).map(|f| f.height(p)).sum()
    }

    fn height_and_gradient(&self, p: Point2<f64>) -> (f64, Vector2<f64>) {
        self.fields(p)
            .map(|f| f.height_and_gradient(p))
            .fold((0., Vector2::zeros()), |(h, g), (fh, fg)| (h + fh, g + fg))
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use approx::assert_relative_eq;
    use orienteering_field::{WINDOW_TOLERANCE, numeric_gradient};

    fn world() -> World {
        World::from_phrase("test world", TerrainParams::default())
    }

    #[test]
    fn chunks_are_deterministic_and_order_independent() {
        let w = world();
        let a = w.chunk(CellIndex::new(3, -7));
        let _ = w.chunk(CellIndex::new(0, 0));
        let b = w.chunk(CellIndex::new(3, -7));
        assert_eq!(a, b);
        assert_ne!(a.field, w.chunk(CellIndex::new(-7, 3)).field);
    }

    #[test]
    fn traces_reproduce_the_chunk() {
        let w = world();
        let traced = w.chunk_traced(CellIndex::new(2, 5));
        assert_eq!(traced.chunk, w.chunk(CellIndex::new(2, 5)));
        assert_eq!(traced.nodes.len(), traced.chunk.nodes);
        assert_eq!(traced.field_to_depth(u32::MAX), traced.chunk.field);

        // Truncating at each depth keeps exactly the nodes down to that depth.
        // Nodes are the masked sums; features never are.
        fn count(f: &Field) -> usize {
            match f {
                Field::Masked { field, .. } => match field.as_ref() {
                    Field::Sum(terms) => 1 + terms.iter().map(count).sum::<usize>(),
                    _ => 0,
                },
                _ => 0,
            }
        }
        for d in 0..=traced.chunk.depth {
            let expected = traced.nodes.iter().filter(|n| n.depth <= d).count();
            assert_eq!(count(&traced.field_to_depth(d)), expected, "depth {d}");
        }

        // The node under a point is the deepest one containing it.
        let p = traced.nodes[0].bounds.lerp(0.3, 0.7);
        let id = traced.node_at(p, u32::MAX).unwrap();
        assert!(traced.nodes[id].bounds.contains(p) && traced.nodes[id].children.is_empty());
        assert_eq!(traced.node_at(p, 0), Some(0));
    }

    #[test]
    fn terrain_is_smooth_across_seams() {
        let w = world();
        let region = w.region(Aabb::new(
            Point2::new(-500., -500.),
            Point2::new(1500., 1500.),
        ));

        // Across the chunk (and province) seam at x = 0 and the chunk seam at x = 1000, the height
        // may only step by the tolerance allowed for truncating window and Gaussian tails, and on
        // either side the analytic gradient must agree with finite differences.
        for y in [-300., 0.5, 250., 999.5] {
            for seam in [0., 1000.] {
                let step = region.height(Point2::new(seam + 1e-9, y))
                    - region.height(Point2::new(seam - 1e-9, y));
                assert!(
                    step.abs() < WINDOW_TOLERANCE,
                    "step of {step} m at x = {seam}"
                );

                for x in [seam - 0.5, seam + 0.5] {
                    let p = Point2::new(x, y);
                    let n = numeric_gradient(&region, p, 1e-3);
                    assert_relative_eq!(region.gradient(p), n, epsilon = 1e-4);
                }
            }
        }
    }

    #[test]
    fn ruggedness_is_smooth_and_matches_the_region() {
        let w = world();
        let region = w.region(Aabb::centered(Point2::origin(), 2000.));
        let mut prev = region.ruggedness(Point2::new(-1000., -370.));
        for i in -1000..=1000 {
            let p = Point2::new(i as f64, 0.37 * i as f64);
            let rho = region.ruggedness(p);
            assert!((0. ..=1.).contains(&rho));
            assert!(
                (rho - prev).abs() < 0.01,
                "ρ jumps by {} at {p}",
                rho - prev
            );
            prev = rho;
        }
        let p = Point2::new(123., -456.);
        assert_relative_eq!(region.ruggedness(p), w.ruggedness_at(p), epsilon = 1e-12);
    }

    #[test]
    fn rugged_ground_has_more_relief() {
        // Across many chunks, local relief (height range within the chunk) should rise with
        // ruggedness.
        let w = world();
        let mut samples: Vec<(f64, f64)> = vec![];
        for cx in -12..12 {
            for cy in [-9, 4] {
                let chunk = w.chunk(CellIndex::new(cx, cy));
                let bounds = chunk.index.bounds(1000.);
                let heights: Vec<f64> = (0..20)
                    .flat_map(|i| (0..20).map(move |j| (i, j)))
                    .map(|(i, j)| chunk.height(bounds.lerp(i as f64 / 19., j as f64 / 19.)))
                    .collect();
                let range = heights.iter().cloned().fold(f64::MIN, f64::max)
                    - heights.iter().cloned().fold(f64::MAX, f64::min);
                samples.push((chunk.ruggedness, range));
            }
        }
        samples.sort_by(|a, b| a.0.total_cmp(&b.0));
        let third = samples.len() / 3;
        let mean = |s: &[(f64, f64)]| s.iter().map(|x| x.1).sum::<f64>() / s.len() as f64;
        let (smooth, rugged) = (mean(&samples[..third]), mean(&samples[2 * third..]));
        assert!(
            rugged > 2. * smooth,
            "relief: smooth {smooth:.1} m, rugged {rugged:.1} m"
        );
    }
}
