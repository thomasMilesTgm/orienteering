//! Rasterized maps: biome fills with relief shading, and contours.

use crate::{Biome, BiomeMap, BiomeRules, Contour, HeightGrid, contours, levels, refine_all};
use nalgebra::{Point2, Vector3};
use orienteering_field::Surface;
use orienteering_math::Aabb;

const BROWN: [u8; 3] = [186, 110, 40];
const BLUE: [u8; 3] = [0, 150, 220];

/// Fill colors loosely following orienteering map conventions, as sRGB in [0, 255].
pub fn fill(biome: Biome) -> [f64; 3] {
    match biome {
        Biome::Water => [140., 205., 240.],
        Biome::Marsh => [252., 250., 245.], // Drawn as blue hatching.
        Biome::Meadow => [255., 216., 120.],
        Biome::Forest => [252., 250., 245.],
        Biome::Rock => [196., 196., 196.],
        Biome::Alpine => [232., 238., 244.],
    }
}

pub struct MapStyle {
    /// Contour interval, meters; every fifth contour is an index contour.
    pub interval: f64,
    /// Smallest biome region drawn, square meters.
    pub min_area: f64,
    /// Newton iterations snapping contour vertices onto the exact level set (0 to skip).
    pub refine: usize,
}

impl MapStyle {
    /// A style whose contour interval suits a scale of `m_per_px`: the smallest of 1, 2, 5, 10,
    /// 20, ... meters keeping contours on moderate slopes a few pixels apart.
    pub fn for_scale(m_per_px: f64) -> Self {
        let interval = [1., 2., 5., 10., 20., 50., 100., 200.]
            .into_iter()
            .find(|&i| i >= 2.5 * m_per_px)
            .unwrap_or(500.);
        Self {
            interval,
            min_area: 400.,
            refine: 0,
        }
    }
}

/// A rendered map: row-major RGB pixels, top row (north) first.
pub struct Rendered {
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<[u8; 3]>,
    pub grid: HeightGrid,
    pub biomes: BiomeMap,
    pub contours: Vec<Contour>,
}

/// Renders `area` of `surface` at one pixel per `spacing` meters.
pub fn render<S: Surface + Sync>(
    surface: &S,
    area: Aabb,
    spacing: f64,
    style: &MapStyle,
) -> Rendered {
    let grid = HeightGrid::sample_with_gradients(surface, area, spacing);
    let (lo, hi) = grid.range();
    let mut lines = contours(&grid, levels(lo, hi, style.interval));
    if style.refine > 0 {
        refine_all(&mut lines, surface, style.refine);
    }

    let mut biomes = BiomeMap::classify(&grid, &BiomeRules::default());
    biomes.smooth(2);
    biomes.absorb_small(style.min_area);

    let (width, height) = (grid.nx, grid.ny);
    let light = Vector3::new(-1., 1., 1.5).normalize();
    let mut pixels = Vec::with_capacity(width * height);
    for py in 0..height {
        for px in 0..width {
            let (i, j) = (px, height - 1 - py);
            let biome = biomes.get(i, j);
            pixels.push(if biome == Biome::Marsh && py % 4 == 0 {
                BLUE
            } else {
                let g = grid.gradient(i, j);
                let normal = Vector3::new(-g.x, -g.y, 1.).normalize();
                let shade = 0.88 + 0.12 * normal.dot(&light);
                fill(biome).map(|c| (c * shade).clamp(0., 255.) as u8)
            });
        }
    }

    let mut map = Rendered {
        width,
        height,
        pixels,
        grid,
        biomes,
        contours: vec![],
    };

    // No contours under water; the zero contour is the shoreline.
    for line in lines.iter().filter(|l| l.level >= 0.) {
        let index = (line.level / style.interval).round() as i64 % 5 == 0;
        let (color, width) = match line.level {
            0. => (BLUE, 1.2),
            _ if index => (BROWN, 1.4),
            _ => (BROWN, 0.7),
        };
        let n = line.points.len();
        let segments = if line.closed { n } else { n.saturating_sub(1) };
        for k in 0..segments {
            let a = map.to_px(line.points[k]);
            let b = map.to_px(line.points[(k + 1) % n]);
            map.draw_line(a, b, width, color);
        }
    }
    map.contours = lines;
    map
}

impl Rendered {
    /// Pixel coordinates (x right, y down) of a world point.
    pub fn to_px(&self, p: Point2<f64>) -> (f64, f64) {
        let v = (p - self.grid.origin) / self.grid.spacing;
        (v.x, (self.height - 1) as f64 - v.y)
    }

    /// Draws a roughly antialiased line by stamping a disc of radius `width` along it.
    fn draw_line(&mut self, a: (f64, f64), b: (f64, f64), width: f64, color: [u8; 3]) {
        let len = ((b.0 - a.0).powi(2) + (b.1 - a.1).powi(2)).sqrt();
        let steps = (len * 2.).ceil().max(1.) as usize;
        let r = width.ceil() as i64;
        for s in 0..=steps {
            let t = s as f64 / steps as f64;
            let (x, y) = (a.0 + (b.0 - a.0) * t, a.1 + (b.1 - a.1) * t);
            for oy in -r..=r {
                for ox in -r..=r {
                    let (px, py) = (x.round() as i64 + ox, y.round() as i64 + oy);
                    let d = ((px as f64 - x).powi(2) + (py as f64 - y).powi(2)).sqrt();
                    if d <= width
                        && (0..self.width as i64).contains(&px)
                        && (0..self.height as i64).contains(&py)
                    {
                        self.pixels[py as usize * self.width + px as usize] = color;
                    }
                }
            }
        }
    }
}
