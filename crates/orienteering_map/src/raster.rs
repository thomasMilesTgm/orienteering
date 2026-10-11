//! Rasterized maps: biome fills with relief shading, contours weighted by rank, and height labels
//! on the heavier contours.

use crate::{
    Biome, BiomeMap, BiomeRules, Contour, ContourLabel, ContourScale, HeightGrid, LabelSpacing,
    contours, levels, place_labels, refine_all,
};
use nalgebra::{Point2, Vector2, Vector3};
use orienteering_field::Surface;
use orienteering_math::Aabb;

const BROWN: [u8; 3] = [186, 110, 40];
const BLUE: [u8; 3] = [0, 150, 220];

/// Label digits are drawn this many pixels per font pixel.
const LABEL_SCALE: i64 = 2;

/// A 5 × 7 pixel font for labels: one row of 5 bits per line, top first.
fn glyph(c: char) -> Option<[u8; 7]> {
    Some(match c {
        '0' => [0x0e, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0e],
        '1' => [0x04, 0x0c, 0x04, 0x04, 0x04, 0x04, 0x0e],
        '2' => [0x0e, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1f],
        '3' => [0x1e, 0x01, 0x01, 0x0e, 0x01, 0x01, 0x1e],
        '4' => [0x02, 0x06, 0x0a, 0x12, 0x1f, 0x02, 0x02],
        '5' => [0x1f, 0x10, 0x1e, 0x01, 0x01, 0x11, 0x0e],
        '6' => [0x06, 0x08, 0x10, 0x1e, 0x11, 0x11, 0x0e],
        '7' => [0x1f, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08],
        '8' => [0x0e, 0x11, 0x11, 0x0e, 0x11, 0x11, 0x0e],
        '9' => [0x0e, 0x11, 0x11, 0x0f, 0x01, 0x02, 0x0c],
        '-' => [0x00, 0x00, 0x00, 0x1f, 0x00, 0x00, 0x00],
        _ => return None,
    })
}

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
    /// Which contours to draw, and how heavily.
    pub contours: ContourScale,
    /// Smallest biome region drawn, square meters.
    pub min_area: f64,
    /// Newton iterations snapping contour vertices onto the exact level set (0 to skip).
    pub refine: usize,
    /// Label the heavier contours with their heights.
    pub labels: bool,
}

impl MapStyle {
    /// A style for a scale of `m_per_px`.
    pub fn for_scale(m_per_px: f64) -> Self {
        Self {
            contours: ContourScale::for_scale(m_per_px),
            min_area: 400.,
            refine: 0,
            labels: true,
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
    pub labels: Vec<ContourLabel>,
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
    let finest = style.contours.finest();
    let mut lines = contours(&grid, levels(lo, hi, finest));
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
        labels: vec![],
    };

    // Heights on the heavier contours (above water), each in a gap cut into its line.
    if style.labels {
        let labelled = style.contours.labelled();
        let index: Vec<Contour> = lines
            .iter()
            .filter(|l| l.level > 0. && crate::contour_rank(l.level) >= labelled)
            .cloned()
            .collect();
        let px = |n: f64| n * spacing;
        let spacing = LabelSpacing {
            along: px(350.),
            apart: px(60.),
            text: px((7 * LABEL_SCALE) as f64),
        };
        map.labels = place_labels(&index, surface, finest, spacing);
    }
    let mut gaps = vec![false; width * height];
    for label in &map.labels {
        map.label_pixels(label, 2, |i, _| gaps[i] = true);
    }

    // No contours under water; the zero contour is the shoreline. Lines are accumulated as
    // coverage, then blended in once, so faint lines stay faint where they overlap themselves.
    let mut brown = vec![0f32; width * height];
    let mut blue = vec![0f32; width * height];
    for line in lines.iter().filter(|l| l.level >= 0.) {
        let (ink, opacity, width) = if line.level == 0. {
            (&mut blue, 1., 1.2)
        } else {
            let (opacity, width) = style.contours.weight(line.level);
            (&mut brown, opacity, width)
        };
        if opacity <= 0. {
            continue;
        }
        let n = line.points.len();
        let segments = if line.closed { n } else { n.saturating_sub(1) };
        for k in 0..segments {
            let a = map.to_px(line.points[k]);
            let b = map.to_px(line.points[(k + 1) % n]);
            map.draw_line(a, b, width, opacity as f32, ink, &gaps);
        }
    }
    for (i, pixel) in map.pixels.iter_mut().enumerate() {
        for (cover, color) in [(brown[i], BROWN), (blue[i], BLUE)] {
            if cover > 0. {
                *pixel = std::array::from_fn(|c| {
                    (pixel[c] as f32 + (color[c] as f32 - pixel[c] as f32) * cover).round() as u8
                });
            }
        }
    }

    let labels = std::mem::take(&mut map.labels);
    for label in &labels {
        let mut on = vec![];
        map.label_pixels(label, 0, |i, ink| {
            if ink {
                on.push(i);
            }
        });
        for i in on {
            map.pixels[i] = BROWN;
        }
    }
    map.labels = labels;
    map.contours = lines;
    map
}

impl Rendered {
    /// Pixel coordinates (x right, y down) of a world point.
    pub fn to_px(&self, p: Point2<f64>) -> (f64, f64) {
        let v = (p - self.grid.origin) / self.grid.spacing;
        (v.x, (self.height - 1) as f64 - v.y)
    }

    /// Calls `f(pixel index, ink)` for every pixel within `margin` pixels of `label`'s text box,
    /// with `ink` set where the text itself is drawn. The text runs along the label's direction.
    fn label_pixels(&self, label: &ContourLabel, margin: i64, mut f: impl FnMut(usize, bool)) {
        let text: Vec<[u8; 7]> = label.text().chars().filter_map(glyph).collect();
        let s = LABEL_SCALE as f64;
        let (w, h) = ((text.len() * 6) as f64 * s - s, 7. * s);
        let (half_w, half_h) = (w / 2. + margin as f64, h / 2. + margin as f64);

        // Pixel coordinates have y down: flip the world's directions.
        let d = label.direction;
        let along = Vector2::new(d.x, -d.y);
        let up = Vector2::new(-d.y, -d.x);
        let (cx, cy) = self.to_px(label.position);
        let r = half_w.hypot(half_h).ceil() as i64;
        for py in (cy as i64 - r)..=(cy as i64 + r) {
            for px in (cx as i64 - r)..=(cx as i64 + r) {
                if !(0..self.width as i64).contains(&px) || !(0..self.height as i64).contains(&py) {
                    continue;
                }
                let rel = Vector2::new(px as f64 + 0.5 - cx, py as f64 + 0.5 - cy);
                let (x, y) = (rel.dot(&along), rel.dot(&up));
                if x.abs() > half_w || y.abs() > half_h {
                    continue;
                }
                // Font pixel: columns along the text, rows down from its top.
                let (fx, fy) = (((x + w / 2.) / s).floor(), ((h / 2. - y) / s).floor());
                let ink = fx >= 0. && (0. ..7.).contains(&fy) && {
                    let (c, col) = (fx as usize / 6, fx as usize % 6);
                    col < 5 && c < text.len() && text[c][fy as usize] & (0x10 >> col) != 0
                };
                f(py as usize * self.width + px as usize, ink);
            }
        }
    }

    /// Draws a line into `ink` (coverage per pixel) by stamping a disc of radius `width` and
    /// the given opacity along it, except where `gaps` is set.
    fn draw_line(
        &self,
        a: (f64, f64),
        b: (f64, f64),
        width: f64,
        opacity: f32,
        ink: &mut [f32],
        gaps: &[bool],
    ) {
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
                        let i = py as usize * self.width + px as usize;
                        if !gaps[i] {
                            ink[i] = ink[i].max(opacity);
                        }
                    }
                }
            }
        }
    }
}
