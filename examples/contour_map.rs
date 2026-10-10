//! Renders a contour map of a generated world to a PNG.
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

const PAPER: [f64; 3] = [252., 250., 245.];
const BROWN: Rgb<u8> = Rgb([186, 110, 40]);

fn main() {
    let mut args = std::env::args().skip(1);
    let phrase = args.next().unwrap_or_else(|| "orienteering".into());
    let out = args.next().unwrap_or_else(|| "contour_map.png".into());

    let world = World::from_phrase(&phrase, TerrainParams::default());
    let area = Aabb::centered(Point2::origin(), EXTENT);
    let region = world.region(area);

    let grid = HeightGrid::sample(&region, area, SCALE);
    let (lo, hi) = grid.range();
    let mut lines = contours(&grid, levels(lo, hi, INTERVAL));
    for line in &mut lines {
        line.refine(&region, 2);
    }

    let mut img = hillshade(&grid);
    for line in &lines {
        let index = (line.level / INTERVAL).round() as i64 % 5 == 0;
        let n = line.points.len();
        let segments = if line.closed { n } else { n.saturating_sub(1) };
        for k in 0..segments {
            let a = to_px(&grid, line.points[k]);
            let b = to_px(&grid, line.points[(k + 1) % n]);
            draw_line(&mut img, a, b, if index { 1.4 } else { 0.7 });
        }
    }

    img.save(&out).expect("failed to write image");

    let rugged: Vec<_> = region.chunks.values().map(|c| c.depth).collect();
    println!(
        "{phrase:?}: {} contours, heights {lo:.0}..{hi:.0} m, chunk depths {rugged:?} -> {out}",
        lines.len()
    );
}

fn to_px(grid: &HeightGrid, p: Point2<f64>) -> (f64, f64) {
    let v = (p - grid.origin) / grid.spacing;
    // Image y runs downwards; map north up.
    (v.x, (grid.ny - 1) as f64 - v.y)
}

/// Soft relief shading lit from the north-west, from grid finite differences.
fn hillshade(grid: &HeightGrid) -> RgbImage {
    let light = nalgebra::Vector3::new(-1., 1., 1.5).normalize();
    RgbImage::from_fn(grid.nx as u32, grid.ny as u32, |px, py| {
        let (i, j) = (px as usize, grid.ny - 1 - py as usize);
        let h = |i: usize, j: usize| grid.get(i.min(grid.nx - 1), j.min(grid.ny - 1));
        let dx = (h(i + 1, j) - h(i.saturating_sub(1), j)) / (2. * grid.spacing);
        let dy = (h(i, j + 1) - h(i, j.saturating_sub(1))) / (2. * grid.spacing);
        let normal = nalgebra::Vector3::new(-dx, -dy, 1.).normalize();
        let shade = 0.88 + 0.12 * normal.dot(&light);
        Rgb(PAPER.map(|c| (c * shade).clamp(0., 255.) as u8))
    })
}

/// Draws an antialiased-ish line by stamping a disc of radius `width` along it.
fn draw_line(img: &mut RgbImage, a: (f64, f64), b: (f64, f64), width: f64) {
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
                    *p = BROWN;
                }
            }
        }
    }
}
