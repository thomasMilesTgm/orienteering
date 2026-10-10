//! Renders a contour map of a generated world, colored by biome, to a PNG.
//!
//! `cargo run --release --example contour_map -- [seed phrase] [output.png]`

use image::{Rgb, RgbImage};
use nalgebra::Point2;
use orienteering::terrain::*;

/// Map extent, meters.
const EXTENT: f64 = 3000.;
/// Meters per pixel.
const SCALE: f64 = 2.;
/// Contour interval, meters; every fifth contour is an index contour.
const INTERVAL: f64 = 5.;
/// Smallest biome region drawn, square meters.
const MIN_AREA: f64 = 400.;

const BROWN: Rgb<u8> = Rgb([186, 110, 40]);
const BLUE: Rgb<u8> = Rgb([0, 150, 220]);

/// Fill colors loosely following orienteering map conventions.
fn fill(biome: Biome) -> [f64; 3] {
    match biome {
        Biome::Water => [140., 205., 240.],
        Biome::Marsh => [252., 250., 245.], // Drawn as blue hatching.
        Biome::Meadow => [255., 216., 120.],
        Biome::Forest => [252., 250., 245.],
        Biome::Rock => [196., 196., 196.],
        Biome::Alpine => [232., 238., 244.],
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let phrase = args.next().unwrap_or_else(|| "orienteering".into());
    let out = args.next().unwrap_or_else(|| "contour_map.png".into());

    let world = World::from_phrase(&phrase, TerrainParams::default());
    let area = Aabb::centered(Point2::origin(), EXTENT);
    let region = world.region(area);

    let grid = HeightGrid::sample_with_gradients(&region, area, SCALE);
    let (lo, hi) = grid.range();
    let mut lines = contours(&grid, levels(lo, hi, INTERVAL));
    refine_all(&mut lines, &region, 2);

    let mut biomes = BiomeMap::classify(&grid, &BiomeRules::default());
    biomes.smooth(2);
    biomes.absorb_small(MIN_AREA);

    let mut img = render_biomes(&grid, &biomes);
    // No contours under water; the zero contour is the shoreline.
    for line in lines.iter().filter(|l| l.level >= 0.) {
        let index = (line.level / INTERVAL).round() as i64 % 5 == 0;
        let (color, width) = match line.level {
            0. => (BLUE, 1.2),
            _ if index => (BROWN, 1.4),
            _ => (BROWN, 0.7),
        };
        let n = line.points.len();
        let segments = if line.closed { n } else { n.saturating_sub(1) };
        for k in 0..segments {
            let a = to_px(&grid, line.points[k]);
            let b = to_px(&grid, line.points[(k + 1) % n]);
            draw_line(&mut img, a, b, width, color);
        }
    }

    img.save(&out).expect("failed to write image");

    let coverage: Vec<_> = biomes
        .coverage()
        .iter()
        .map(|(b, f)| format!("{b:?} {:.1}%", f * 100.))
        .collect();
    println!(
        "{phrase:?}: {} contours, heights {lo:.0}..{hi:.0} m, {} biome regions ({}) -> {out}",
        lines.len(),
        biomes.regions().regions.len(),
        coverage.join(", "),
    );
}

fn to_px(grid: &HeightGrid, p: Point2<f64>) -> (f64, f64) {
    let v = (p - grid.origin) / grid.spacing;
    // Image y runs downwards; map north up.
    (v.x, (grid.ny - 1) as f64 - v.y)
}

/// Biome fills with soft relief shading lit from the north-west.
fn render_biomes(grid: &HeightGrid, biomes: &BiomeMap) -> RgbImage {
    let light = nalgebra::Vector3::new(-1., 1., 1.5).normalize();
    RgbImage::from_fn(grid.nx as u32, grid.ny as u32, |px, py| {
        let (i, j) = (px as usize, grid.ny - 1 - py as usize);
        let biome = biomes.get(i, j);
        if biome == Biome::Marsh && py % 4 == 0 {
            return BLUE;
        }
        let g = grid.gradient(i, j);
        let normal = nalgebra::Vector3::new(-g.x, -g.y, 1.).normalize();
        let shade = 0.88 + 0.12 * normal.dot(&light);
        Rgb(fill(biome).map(|c| (c * shade).clamp(0., 255.) as u8))
    })
}

/// Draws an antialiased-ish line by stamping a disc of radius `width` along it.
fn draw_line(img: &mut RgbImage, a: (f64, f64), b: (f64, f64), width: f64, color: Rgb<u8>) {
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
                    && px >= 0
                    && py >= 0
                    && let Some(p) = img.get_pixel_mut_checked(px as u32, py as u32)
                {
                    *p = color;
                }
            }
        }
    }
}
