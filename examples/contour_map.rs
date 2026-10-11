//! Renders a contour map of a generated world, colored by biome, to a PNG.
//!
//! `cargo run --release --example contour_map -- [seed phrase] [output.png] [extent m] [m/px]
//! [center x] [center y]`
//!
//! Areas wider than 20 km show only the broad shape of the land (plates and provinces).

use nalgebra::Point2;
use orienteering::{
    map::{
        ContourScale,
        raster::{self, MapStyle},
    },
    math::Aabb,
    terrain::{TerrainParams, World},
};

fn main() {
    let mut args = std::env::args().skip(1);
    let phrase = args.next().unwrap_or_else(|| "orienteering".into());
    let out = args.next().unwrap_or_else(|| "contour_map.png".into());
    let mut number = |default: f64| args.next().map_or(default, |a| a.parse().expect("number"));
    let extent = number(3000.);
    let scale = number(2.);
    let center = Point2::new(number(0.), number(0.));

    let world = World::from_phrase(&phrase, TerrainParams::default());
    let area = Aabb::centered(center, extent);
    let broad = extent > 20_000.;
    let region = if broad {
        world.broad_region(area)
    } else {
        world.region(area)
    };

    // Broad views show only the large-scale relief, gentle enough for closer contours than the
    // pixel size alone would suggest.
    let style = if broad {
        MapStyle {
            contours: ContourScale { density: 25. },
            min_area: 400.,
            refine: 0,
            labels: true,
        }
    } else {
        MapStyle {
            contours: ContourScale::for_scale(scale),
            min_area: 400.,
            refine: 2,
            labels: true,
        }
    };
    let map = raster::render(&region, area, scale, &style);

    let img = image::RgbImage::from_fn(map.width as u32, map.height as u32, |x, y| {
        image::Rgb(map.pixels[y as usize * map.width + x as usize])
    });
    img.save(&out).expect("failed to write image");

    let (lo, hi) = map.grid.range();
    let coverage: Vec<_> = map
        .biomes
        .coverage()
        .iter()
        .map(|(b, f)| format!("{b:?} {:.1}%", f * 100.))
        .collect();
    println!(
        "{phrase:?}: {} contours, heights {lo:.0}..{hi:.0} m, {} biome regions ({}) -> {out}",
        map.contours.len(),
        map.biomes.regions().regions.len(),
        coverage.join(", "),
    );
}
