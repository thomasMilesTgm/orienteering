//! A flat, GPU-friendly form of the terrain's fields.
//!
//! Shaders can't recurse or follow boxed enums, so a [`Field`] tree is flattened into arrays:
//!
//! - **Nodes**: one per window (each [`Field::Masked`]), warp (each [`Field::Warped`]) or bounded
//!   subtree (each [`Field::Bounded`]), plus a root per cell, in pre-order. Each node stores its own primitive features as a range of the
//!   feature array, and the index just past its subtree (`skip`), so a traversal can skip a whole
//!   subtree, outside its support or not yet generated, without a stack.
//! - **Features**: the primitives (constants, planes, bumps, ridges, scarps, waves, ranges), with any
//!   [`Field::Scaled`] factors folded into their heights.
//!
//! Coordinates are relative to each cell's origin and stored as f32, which keeps millimeter
//! precision anywhere in the world (shaders only have f32).
//!
//! # Layout
//!
//! Both arrays are packed as `[f32; 4]` (`vec4<f32>`); integers are stored as their bit patterns
//! (`bitcast<u32>` in WGSL).
//!
//! A node is [`NODE_VEC4S`] vec4s:
//!
//! | # | x | y | z | w |
//! |---|---|---|---|---|
//! | 0 | window min x | min y | max x | max y |
//! | 1 | support min x | min y | max x | max y |
//! | 2 | window blend (0: no window) | level | depth (u32) | skip (u32) |
//! | 3 | warp x: direction x | direction y | 2π / wavelength | phase |
//! | 4 | warp x: amplitude (0: no warp) | warp y: direction x | direction y | 2π / wavelength |
//! | 5 | warp y: phase | amplitude | first feature (u32) | end of features (u32) |
//!
//! A feature is [`FEATURE_VEC4S`] vec4s: `[kind (u32), a.x, a.y, b.x]`, `[b.y, p0, p1, p2]`:
//!
//! | kind | a | b | p0 | p1 | p2 |
//! |---|---|---|---|---|---|
//! | 0 constant | | | value | | |
//! | 1 plane | origin | slope | | | |
//! | 2 bump | center | | radius | height | |
//! | 3 ridge | origin | direction | width | height | |
//! | 4 scarp | origin | normal | width | height | |
//! | 5 wave | | direction | 2π / wavelength | phase | amplitude |
//! | 6 range | center | direction | width | height | half length |
//!
//! Nodes' features are weighted by `clamp(stage - level, 0, 1)`: the terrain partway through
//! generation (see [`FlatScene::height_and_gradient`]).

use nalgebra::{Point2, Vector2};
use orienteering_field::{Field, SCARP_REACH, Window};
use orienteering_math::Aabb;
use orienteering_terrain::{CellIndex, PLATE_REACH, PROVINCE_REACH};
use std::f64::consts::TAU;
use std::ops::{Add, Mul};

pub const NODE_VEC4S: usize = 6;
pub const FEATURE_VEC4S: usize = 2;

/// Deepest node nesting a traversal supports (the size of its window-product stack).
pub const MAX_NESTING: u32 = 8;

/// The level of nodes that are always fully present, whatever the stage.
pub const ALWAYS: f32 = -1e6;

/// A field that can't be flattened.
#[derive(Debug, Clone, PartialEq)]
pub struct Unsupported(pub &'static str);

impl std::fmt::Display for Unsupported {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "can't flatten field: {}", self.0)
    }
}

impl std::error::Error for Unsupported {}

/// One cell's flattened field, in coordinates relative to the cell's origin.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct FlatCell {
    nodes: Vec<[[f32; 4]; NODE_VEC4S]>,
    features: Vec<[[f32; 4]; FEATURE_VEC4S]>,
}

impl FlatCell {
    /// Flattens `field` relative to `origin`. `level(n)` gives the stage level of a node nested in
    /// `n` windows (the root has `n = 0`).
    pub fn new(
        field: &Field,
        origin: Point2<f64>,
        level: impl Fn(u32) -> f32,
    ) -> Result<Self, Unsupported> {
        let mut f = Flattener {
            cell: Self::default(),
            origin,
            level: &level,
        };
        f.node(None, None, None, 0, 0, 1., field)?;
        Ok(f.cell)
    }

    pub fn nodes(&self) -> usize {
        self.nodes.len()
    }
}

struct Flattener<'a> {
    cell: FlatCell,
    origin: Point2<f64>,
    level: &'a dyn Fn(u32) -> f32,
}

/// A child of a node: something that becomes a node of its own.
enum Child<'a> {
    Window(&'a Window, &'a Aabb, &'a Field, f64),
    Warp(&'a Field, &'a Field, &'a Field, f64),
    Bounded(&'a Aabb, &'a Field, f64),
}

impl<'a> Flattener<'a> {
    fn local(&self, p: Point2<f64>) -> [f32; 2] {
        [(p.x - self.origin.x) as f32, (p.y - self.origin.y) as f32]
    }

    /// Pushes a node, its features and then its children's subtrees.
    fn node(
        &mut self,
        window: Option<&Window>,
        support: Option<&Aabb>,
        warp: Option<(&Field, &Field)>,
        depth: u32,
        windows: u32,
        scale: f64,
        body: &'a Field,
    ) -> Result<(), Unsupported> {
        if depth >= MAX_NESTING {
            return Err(Unsupported("nested too deeply"));
        }
        let id = self.cell.nodes.len();
        self.cell.nodes.push(Default::default());

        let first = self.cell.features.len() as u32;
        let mut children = vec![];
        self.collect(body, scale, &mut children)?;
        let end = self.cell.features.len() as u32;
        if warp.is_some() && !children.is_empty() {
            return Err(Unsupported("windows or bounds inside a warp"));
        }

        let level = (self.level)(windows);
        for child in children {
            match child {
                Child::Window(w, support, f, s) => {
                    self.node(Some(w), Some(support), None, depth + 1, windows + 1, s, f)?
                }
                Child::Warp(dx, dy, f, s) => {
                    self.node(None, None, Some((dx, dy)), depth + 1, windows, s, f)?
                }
                Child::Bounded(support, f, s) => {
                    self.node(None, Some(support), None, depth + 1, windows, s, f)?
                }
            }
        }
        let skip = self.cell.nodes.len() as u32;

        let (min, max, blend) = match window {
            Some(w) => (
                self.local(w.bounds.min),
                self.local(w.bounds.max),
                w.blend as f32,
            ),
            None => ([0.; 2], [0.; 2], 0.),
        };
        let (smin, smax) = match support {
            Some(s) => (self.local(s.min), self.local(s.max)),
            None => ([-1e9; 2], [1e9; 2]),
        };
        let [wx, wy] = match warp {
            Some((dx, dy)) => [self.wave(dx)?, self.wave(dy)?],
            None => [[0.; 5]; 2],
        };
        let bits = f32::from_bits;
        self.cell.nodes[id] = [
            [min[0], min[1], max[0], max[1]],
            [smin[0], smin[1], smax[0], smax[1]],
            [blend, level, bits(depth), bits(skip)],
            [wx[0], wx[1], wx[2], wx[3]],
            [wx[4], wy[0], wy[1], wy[2]],
            [wy[3], wy[4], bits(first), bits(end)],
        ];
        Ok(())
    }

    /// Adds `field`'s primitives (scaled by `scale`) as features of the current node, and
    /// collects what must become child nodes.
    fn collect(
        &mut self,
        field: &'a Field,
        scale: f64,
        children: &mut Vec<Child<'a>>,
    ) -> Result<(), Unsupported> {
        let feature = |kind: u32, a: [f32; 2], b: [f32; 2], p: [f64; 3]| {
            [
                [f32::from_bits(kind), a[0], a[1], b[0]],
                [b[1], p[0] as f32, p[1] as f32, p[2] as f32],
            ]
        };
        let v = |v: &Vector2<f64>| [v.x as f32, v.y as f32];
        let f = match field {
            Field::Sum(terms) => {
                for t in terms {
                    self.collect(t, scale, children)?;
                }
                return Ok(());
            }
            Field::Scaled { factor, field } => {
                return self.collect(field, scale * factor, children);
            }
            Field::Masked {
                window,
                support,
                field,
            } => {
                children.push(Child::Window(window, support, field, scale));
                return Ok(());
            }
            Field::Warped { dx, dy, field } => {
                children.push(Child::Warp(dx, dy, field, scale));
                return Ok(());
            }
            Field::Bounded { support, field } => {
                children.push(Child::Bounded(support, field, scale));
                return Ok(());
            }
            Field::Product(_) => return Err(Unsupported("products")),
            Field::Range {
                center,
                direction,
                half_length,
                width,
                height,
            } => feature(
                6,
                self.local(*center),
                v(direction),
                [*width, height * scale, *half_length],
            ),
            Field::Constant(c) => feature(0, [0.; 2], [0.; 2], [c * scale, 0., 0.]),
            Field::Plane { origin, slope } => {
                feature(1, self.local(*origin), v(&(slope * scale)), [0.; 3])
            }
            Field::Bump {
                center,
                radius,
                height,
            } => feature(
                2,
                self.local(*center),
                [0.; 2],
                [*radius, height * scale, 0.],
            ),
            Field::Ridge {
                origin,
                direction,
                width,
                height,
            } => feature(
                3,
                self.local(*origin),
                v(direction),
                [*width, height * scale, 0.],
            ),
            Field::Scarp {
                origin,
                normal,
                width,
                height,
            } => feature(
                4,
                self.local(*origin),
                v(normal),
                [*width, height * scale, 0.],
            ),
            Field::Wave { .. } => {
                let [dx, dy, k, phase, amplitude] = self.wave(field)?;
                feature(
                    5,
                    [0.; 2],
                    [dx, dy],
                    [k as f64, phase as f64, amplitude as f64 * scale],
                )
            }
        };
        self.cell.features.push(f);
        Ok(())
    }

    /// A wave in local coordinates: `[direction x, direction y, 2π / wavelength, phase,
    /// amplitude]`, with the phase shifted to account for the origin.
    fn wave(&self, field: &Field) -> Result<[f32; 5], Unsupported> {
        let Field::Wave {
            direction,
            wavelength,
            phase,
            amplitude,
        } = field
        else {
            return Err(Unsupported("warps other than waves"));
        };
        let k = TAU / wavelength;
        let shifted = (phase + k * direction.dot(&self.origin.coords)).rem_euclid(TAU);
        Ok([
            direction.x as f32,
            direction.y as f32,
            k as f32,
            shifted as f32,
            *amplitude as f32,
        ])
    }
}

/// Square cells of one size, each with a range of nodes.
#[derive(Debug, Clone, PartialEq)]
pub struct CellTable {
    /// The cell at the table's minimum corner.
    pub min: CellIndex,
    /// Cells across and down.
    pub size: [u32; 2],
    pub cell_size: f64,
    /// `[start, end)` node ranges, row-major from `min`; empty for cells without data.
    pub ranges: Vec<[u32; 2]>,
}

impl CellTable {
    /// World position of the table's minimum corner.
    pub fn origin(&self) -> Point2<f64> {
        Point2::new(self.min.x as f64, self.min.y as f64) * self.cell_size
    }

    fn range(&self, i: CellIndex) -> Option<[u32; 2]> {
        let (x, y) = (i.x - self.min.x, i.y - self.min.y);
        if x < 0 || y < 0 || x >= self.size[0] as i64 || y >= self.size[1] as i64 {
            return None;
        }
        Some(self.ranges[(y as usize) * self.size[0] as usize + x as usize])
    }
}

/// Flattened plates, provinces and chunks covering an area, ready to upload.
#[derive(Debug, Clone, PartialEq)]
pub struct FlatScene {
    /// [`NODE_VEC4S`] per node.
    pub nodes: Vec<[f32; 4]>,
    /// [`FEATURE_VEC4S`] per feature.
    pub features: Vec<[f32; 4]>,
    pub chunks: CellTable,
    pub provinces: CellTable,
    pub plates: CellTable,
    /// A constant height under everything (the world's mean elevation); 0 by default.
    pub baseline: f32,
}

impl FlatScene {
    pub fn new(
        chunks: impl IntoIterator<Item = (CellIndex, FlatCell)>,
        chunk_size: f64,
        provinces: impl IntoIterator<Item = (CellIndex, FlatCell)>,
        province_size: f64,
        plates: impl IntoIterator<Item = (CellIndex, FlatCell)>,
        plate_size: f64,
    ) -> Self {
        let mut scene = Self {
            nodes: vec![],
            features: vec![],
            chunks: CellTable::empty(chunk_size),
            provinces: CellTable::empty(province_size),
            plates: CellTable::empty(plate_size),
            baseline: 0.,
        };
        scene.chunks = scene.table(chunks, chunk_size);
        scene.provinces = scene.table(provinces, province_size);
        scene.plates = scene.table(plates, plate_size);
        scene
    }

    /// Appends cells' nodes and features, rebasing their indices, and indexes them in a table.
    fn table(
        &mut self,
        cells: impl IntoIterator<Item = (CellIndex, FlatCell)>,
        cell_size: f64,
    ) -> CellTable {
        let cells: Vec<_> = cells.into_iter().collect();
        if cells.is_empty() {
            return CellTable::empty(cell_size);
        }
        let min = CellIndex::new(
            cells.iter().map(|c| c.0.x).min().unwrap(),
            cells.iter().map(|c| c.0.y).min().unwrap(),
        );
        let max = CellIndex::new(
            cells.iter().map(|c| c.0.x).max().unwrap(),
            cells.iter().map(|c| c.0.y).max().unwrap(),
        );
        let size = [(max.x - min.x + 1) as u32, (max.y - min.y + 1) as u32];
        let mut ranges = vec![[0, 0]; (size[0] * size[1]) as usize];

        for (index, cell) in cells {
            let node_base = (self.nodes.len() / NODE_VEC4S) as u32;
            let feature_base = (self.features.len() / FEATURE_VEC4S) as u32;
            let rebase = |v: f32, base: u32| f32::from_bits(v.to_bits() + base);
            for mut node in cell.nodes {
                node[2][3] = rebase(node[2][3], node_base);
                node[5][2] = rebase(node[5][2], feature_base);
                node[5][3] = rebase(node[5][3], feature_base);
                self.nodes.extend(node);
            }
            for feature in cell.features {
                self.features.extend(feature);
            }
            let end = (self.nodes.len() / NODE_VEC4S) as u32;
            let (x, y) = ((index.x - min.x) as usize, (index.y - min.y) as usize);
            ranges[y * size[0] as usize + x] = [node_base, end];
        }
        CellTable {
            min,
            size,
            cell_size,
            ranges,
        }
    }

    /// Height and gradient at `p`, partway through generation at `stage`, computed exactly as the
    /// shader does (in f32): the reference for testing both the flattening and the shader.
    pub fn height_and_gradient(&self, p: Point2<f64>, stage: f32) -> (f32, [f32; 2]) {
        let mut total = D::constant(self.baseline);
        for (table, reach) in [
            (&self.plates, PLATE_REACH),
            (&self.provinces, PROVINCE_REACH),
            (&self.chunks, 1),
        ] {
            if table.size[0] == 0 {
                continue;
            }
            let center = CellIndex::containing(p, table.cell_size);
            for cell in center.within(reach) {
                let Some([start, end]) = table.range(cell) else {
                    continue;
                };
                let origin = Point2::new(cell.x as f64, cell.y as f64) * table.cell_size;
                let q = [(p.x - origin.x) as f32, (p.y - origin.y) as f32];
                total = total + self.eval_cell(start, end, q, stage);
            }
        }
        (total.v, [total.dx, total.dy])
    }

    fn node(&self, i: u32) -> &[[f32; 4]] {
        let i = i as usize * NODE_VEC4S;
        &self.nodes[i..i + NODE_VEC4S]
    }

    fn eval_cell(&self, start: u32, end: u32, q: [f32; 2], stage: f32) -> D {
        // windows[d] is the product of the windows of the nodes enclosing depth d.
        let mut windows = [D::constant(1.); MAX_NESTING as usize + 1];
        let mut h = D::constant(0.);
        let mut i = start;
        while i < end {
            let n = self.node(i);
            let [blend, level, depth, skip] = n[2];
            let (depth, skip) = (depth.to_bits() as usize, skip.to_bits());
            let weight = (stage - level).clamp(0., 1.);
            let s = n[1];
            if weight == 0. || q[0] < s[0] || q[1] < s[1] || q[0] > s[2] || q[1] > s[3] {
                i = skip;
                continue;
            }

            let (x, y) = (D::x(q[0]), D::y(q[1]));
            let window = if blend > 0. {
                let b = n[0];
                axis(x, b[0], b[2], blend) * axis(y, b[1], b[3], blend)
            } else {
                D::constant(1.)
            };
            windows[depth + 1] = windows[depth] * window;

            let (wx, wy) = if n[4][0] != 0. || n[5][1] != 0. {
                let w0 = [n[3][0], n[3][1], n[3][2], n[3][3], n[4][0]];
                let w1 = [n[4][1], n[4][2], n[4][3], n[5][0], n[5][1]];
                (x + wave(x, y, w0), y + wave(x, y, w1))
            } else {
                (x, y)
            };

            let mut sum = D::constant(0.);
            for f in n[5][2].to_bits()..n[5][3].to_bits() {
                let f = f as usize * FEATURE_VEC4S;
                sum = sum + feature(&self.features[f..f + FEATURE_VEC4S], wx, wy);
            }
            h = h + windows[depth + 1] * sum * weight;
            i += 1;
        }
        h
    }
}

impl CellTable {
    fn empty(cell_size: f64) -> Self {
        Self {
            min: CellIndex::new(0, 0),
            size: [0, 0],
            cell_size,
            ranges: vec![],
        }
    }
}

/// f32 value and gradient, mirroring the shader's `vec3<f32>` duals.
#[derive(Debug, Clone, Copy)]
struct D {
    v: f32,
    dx: f32,
    dy: f32,
}

impl D {
    fn constant(v: f32) -> Self {
        Self { v, dx: 0., dy: 0. }
    }
    fn x(v: f32) -> Self {
        Self { v, dx: 1., dy: 0. }
    }
    fn y(v: f32) -> Self {
        Self { v, dx: 0., dy: 1. }
    }
    fn map(self, v: f32, d: f32) -> Self {
        Self {
            v,
            dx: self.dx * d,
            dy: self.dy * d,
        }
    }
    fn exp(self) -> Self {
        let e = self.v.exp();
        self.map(e, e)
    }
    fn sin(self) -> Self {
        self.map(self.v.sin(), self.v.cos())
    }
    fn tanh(self) -> Self {
        // Clamped like the shader, where tanh of large arguments can overflow.
        let t = self.v.clamp(-15., 15.).tanh();
        self.map(t, 1. - t * t)
    }
}

impl Add for D {
    type Output = Self;
    fn add(self, o: Self) -> Self {
        Self {
            v: self.v + o.v,
            dx: self.dx + o.dx,
            dy: self.dy + o.dy,
        }
    }
}

impl Mul for D {
    type Output = Self;
    fn mul(self, o: Self) -> Self {
        Self {
            v: self.v * o.v,
            dx: self.dx * o.v + o.dx * self.v,
            dy: self.dy * o.v + o.dy * self.v,
        }
    }
}

impl Mul<f32> for D {
    type Output = Self;
    fn mul(self, k: f32) -> Self {
        Self {
            v: self.v * k,
            dx: self.dx * k,
            dy: self.dy * k,
        }
    }
}

fn axis(t: D, a: f32, b: f32, blend: f32) -> D {
    let k = 1. / blend;
    (((t + D::constant(-a)) * k).tanh() + ((t + D::constant(-b)) * k).tanh() * -1.) * 0.5
}

/// `[direction x, direction y, k, phase, amplitude]`.
fn wave(x: D, y: D, w: [f32; 5]) -> D {
    ((x * w[0] + y * w[1]) * w[2] + D::constant(w[3])).sin() * w[4]
}

fn feature(f: &[[f32; 4]], x: D, y: D) -> D {
    let kind = f[0][0].to_bits();
    let (ax, ay, bx) = (f[0][1], f[0][2], f[0][3]);
    let (by, p0, p1, p2) = (f[1][0], f[1][1], f[1][2], f[1][3]);
    let (rx, ry) = (x + D::constant(-ax), y + D::constant(-ay));
    match kind {
        0 => D::constant(p0),
        1 => rx * bx + ry * by,
        2 => ((rx * rx + ry * ry) * (-0.5 / (p0 * p0))).exp() * p1,
        3 => {
            let d = rx * -by + ry * bx;
            ((d * d) * (-0.5 / (p0 * p0))).exp() * p1
        }
        4 => {
            let d = rx * bx + ry * by;
            let reach = SCARP_REACH as f32 * p0;
            let fade = ((d * d) * (-0.5 / (reach * reach))).exp();
            (d * (1. / p0)).tanh() * fade * (p1 / 2.)
        }
        5 => wave(x, y, [bx, by, p0, p1, p2]),
        6 => {
            let along = rx * bx + ry * by;
            let across = rx * -by + ry * bx;
            let k = 1. / p0;
            let taper = (((along + D::constant(p2)) * k).tanh()
                + ((along + D::constant(-p2)) * k).tanh() * -1.)
                * 0.5;
            ((across * across) * (-0.5 * k * k)).exp() * taper * p1
        }
        _ => D::constant(0.),
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use orienteering_field::Surface;
    use orienteering_terrain::{TerrainParams, TracedChunk, World};

    fn close(a: f32, b: f64, tolerance: f64) -> bool {
        (a as f64 - b).abs() <= tolerance
    }

    #[test]
    fn flattened_fields_match() {
        let w = Window::new(
            Aabb::new(Point2::new(1000., 2000.), Point2::new(1500., 2500.)),
            40.,
        );
        let warped = (Field::scarp(Point2::new(1300., 2200.), Vector2::new(-2., 1.), 12., 9.)
            + Field::bump(Point2::new(1150., 2150.), 40., -5.))
        .warped(
            Field::wave(Vector2::new(1., 2.), 120., 0.3, 15.),
            Field::wave(Vector2::new(-2., 1.), 150., 1.9, 15.),
        );
        let nested = Field::bump(Point2::new(1100., 2400.), 50., 8.).masked(Window::new(
            Aabb::new(Point2::new(1000., 2250.), Point2::new(1250., 2500.)),
            20.,
        ));
        let node = Field::bump(Point2::new(1200., 2300.), 80., 25.)
            + Field::ridge(Point2::new(1250., 2250.), Vector2::new(1., 2.), 30., -12.)
            + Field::wave(Vector2::new(3., -1.), 170., 0.4, 6.)
            + warped
            + nested;
        let field = node.masked(w) * 3.
            + Field::range(
                Point2::new(900., 2100.),
                Point2::new(1600., 2400.),
                60.,
                30.,
            )
            .warped(
                Field::wave(Vector2::new(1., 1.), 900., 0.2, 40.),
                Field::wave(Vector2::new(1., -1.), 700., 1.2, 40.),
            )
            .bounded(Aabb::centered(Point2::new(1250., 2250.), 2000.))
            + Field::plane(Point2::new(1000., 2000.), Vector2::new(0.01, 0.02))
            + Field::Constant(7.);

        // A table holding the single cell whose origin the field is flattened relative to.
        let origin = Point2::new(1000., 2000.);
        let flat = FlatCell::new(&field, origin, |_| ALWAYS).unwrap();
        let scene = FlatScene::new([(CellIndex::new(1, 2), flat)], 1000., [], 8000., [], 48000.);

        for (x, y) in [
            (1200., 2300.),
            (1010., 2490.),
            (1499., 2001.),
            (1600., 2600.),
        ] {
            let p = Point2::new(x, y);
            let (h, g) = scene.height_and_gradient(p, 0.);
            let (eh, eg) = field.height_and_gradient(p);
            assert!(close(h, eh, 1e-3), "height at {p}: {h} vs {eh}");
            assert!(
                close(g[0], eg.x, 1e-4) && close(g[1], eg.y, 1e-4),
                "gradient at {p}"
            );
        }
    }

    #[test]
    fn unsupported_fields_are_rejected() {
        let product = Field::Constant(1.) * Field::Constant(2.);
        assert!(FlatCell::new(&product, Point2::origin(), |_| ALWAYS).is_err());

        // A warp would bend a nested window's bounds, which a flat node can't express.
        let window = Window::new(Aabb::centered(Point2::origin(), 10.), 1.);
        let wave = Field::wave(Vector2::new(1., 0.), 10., 0., 1.);
        let warped_window = Field::Constant(1.)
            .masked(window)
            .warped(wave.clone(), wave);
        assert!(FlatCell::new(&warped_window, Point2::origin(), |_| ALWAYS).is_err());
    }

    /// Flattens a world's provinces and chunks the way the viewer does.
    fn scene(world: &World, area: Aabb) -> (FlatScene, Vec<TracedChunk>) {
        let p = &world.params;
        let provinces = CellIndex::covering(area, p.province_size, PROVINCE_REACH).map(|i| {
            let origin = i.bounds(p.province_size).min;
            let flat = FlatCell::new(&world.province(i).elevation, origin, |_| ALWAYS);
            (i, flat.unwrap())
        });
        let plate_size = p.tectonics.plate_size;
        let plates = CellIndex::covering(area, plate_size, PLATE_REACH).map(|i| {
            let origin = i.bounds(plate_size).min;
            let flat = FlatCell::new(&world.plate(i).elevation, origin, |_| ALWAYS);
            (i, flat.unwrap())
        });
        let traced: Vec<_> = CellIndex::covering(area, p.chunk_size, 1)
            .map(|i| world.chunk_traced(i))
            .collect();
        let chunks = traced.iter().map(|t| {
            let origin = t.chunk.index.bounds(p.chunk_size).min;
            // The root is a window-less node holding the landform level's window.
            let flat = FlatCell::new(&t.chunk.field, origin, |n| n as f32 - 1.);
            (t.chunk.index, flat.unwrap())
        });
        let mut scene = FlatScene::new(
            chunks,
            p.chunk_size,
            provinces.collect::<Vec<_>>(),
            p.province_size,
            plates.collect::<Vec<_>>(),
            plate_size,
        );
        scene.baseline = p.mean_elevation as f32;
        (scene, traced)
    }

    #[test]
    fn flattened_world_matches_at_every_stage() {
        let world = World::from_phrase("misty hollow", TerrainParams::default());
        let area = Aabb::centered(Point2::new(2600., 2600.), 1500.);
        let (scene, traced) = scene(&world, area);
        let region = world.region(area);

        let points: Vec<_> = (0..40)
            .map(|k| area.lerp((k as f64 * 0.618).fract(), (k as f64 * 0.377).fract()))
            .collect();

        // Fully generated: the region itself.
        for &p in &points {
            let (h, g) = scene.height_and_gradient(p, 1e3);
            let (eh, eg) = region.height_and_gradient(p);
            assert!(close(h, eh, 5e-3), "height at {p}: {h} vs {eh}");
            assert!(
                close(g[0], eg.x, 1e-3) && close(g[1], eg.y, 1e-3),
                "gradient at {p}"
            );
        }

        // Partway: provinces plus each chunk's levels down to `stage - 1`, the next level
        // faded in by the fractional part.
        for stage in [0., 0.5, 1., 2.25, 3.] {
            let at = |p: Point2<f64>, depth: f64| -> f64 {
                let chunks: f64 = traced
                    .iter()
                    .map(|t| {
                        if depth < 0. {
                            0.
                        } else {
                            t.field_to_depth(depth as u32).height(p)
                        }
                    })
                    .sum();
                let provinces: f64 = region
                    .provinces
                    .values()
                    .map(|v| &v.elevation)
                    .chain(region.plates.values().map(|v| &v.elevation))
                    .map(|f| f.height(p))
                    .sum();
                region.baseline + provinces + chunks
            };
            for &p in &points {
                let (lo, t) = (f64::floor(stage), stage - f64::floor(stage));
                let expected = at(p, lo - 1.) * (1. - t) + at(p, lo) * t;
                let (h, _) = scene.height_and_gradient(p, stage as f32);
                assert!(
                    close(h, expected, 5e-3),
                    "stage {stage} at {p}: {h} vs {expected}"
                );
            }
        }
    }
}
