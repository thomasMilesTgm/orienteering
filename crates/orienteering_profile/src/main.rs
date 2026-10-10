//! Profiles terrain generation and contouring.
//!
//! `cargo run --release -p orienteering_profile -- --help` for options.
//!
//! Span timings are inclusive (a span's time includes its children's) and measured from span
//! creation to close. Spans on worker threads (e.g. `contour::refine`) run concurrently, so their
//! totals add up time across threads and can exceed their parent's.

use nalgebra::Point2;
use orienteering::{
    field::Surface,
    map::{BiomeMap, BiomeRules, HeightGrid, contours, levels, refine_all},
    math::{Aabb, rng::Rng64},
    terrain::{Region, TerrainParams, World},
};
use std::collections::BTreeMap;
use std::hint::black_box;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tracing::span::{Attributes, Id};
use tracing::{Subscriber, info_span};
use tracing_subscriber::Layer;
use tracing_subscriber::filter::LevelFilter;
use tracing_subscriber::layer::{Context, SubscriberExt};
use tracing_subscriber::registry::LookupSpan;
use tracing_subscriber::util::SubscriberInitExt;

struct Args {
    seed: String,
    extent: f64,
    spacing: f64,
    interval: f64,
    refine: usize,
    runs: usize,
    evals: usize,
    min_area: f64,
    level: LevelFilter,
    chrome: Option<String>,
}

impl Args {
    fn parse() -> Result<Self, String> {
        let mut args = Self {
            seed: "orienteering".into(),
            extent: 3000.,
            spacing: 2.,
            interval: 5.,
            refine: 2,
            runs: 3,
            evals: 200_000,
            min_area: 400.,
            level: LevelFilter::DEBUG,
            chrome: None,
        };

        let mut it = std::env::args().skip(1);
        while let Some(flag) = it.next() {
            if flag == "--help" || flag == "-h" {
                return Err(String::new());
            }
            let value = it.next().ok_or(format!("{flag} needs a value"))?;
            let bad = |e: &dyn std::fmt::Display| format!("bad value for {flag}: {e}");
            match flag.as_str() {
                "--seed" => args.seed = value,
                "--extent" => args.extent = value.parse().map_err(|e| bad(&e))?,
                "--spacing" => args.spacing = value.parse().map_err(|e| bad(&e))?,
                "--interval" => args.interval = value.parse().map_err(|e| bad(&e))?,
                "--refine" => args.refine = value.parse().map_err(|e| bad(&e))?,
                "--runs" => args.runs = value.parse().map_err(|e| bad(&e))?,
                "--evals" => args.evals = value.parse().map_err(|e| bad(&e))?,
                "--min-area" => args.min_area = value.parse().map_err(|e| bad(&e))?,
                "--level" => args.level = value.parse().map_err(|e| bad(&e))?,
                "--chrome" => args.chrome = Some(value),
                _ => return Err(format!("unknown option {flag}")),
            }
        }
        Ok(args)
    }
}

fn main() {
    let args = match Args::parse() {
        Ok(args) => args,
        Err(e) => {
            if !e.is_empty() {
                eprintln!("error: {e}\n");
            }
            eprintln!("{}", USAGE.trim());
            std::process::exit(if e.is_empty() { 0 } else { 2 });
        }
    };

    let stats = SpanStats::default();
    let (chrome, _flush_on_drop) = match &args.chrome {
        Some(path) => {
            let (layer, guard) = tracing_chrome::ChromeLayerBuilder::new()
                .file(path)
                .include_args(true)
                .build();
            (Some(layer), Some(guard))
        }
        None => (None, None),
    };
    tracing_subscriber::registry()
        .with(args.level)
        .with(stats.clone())
        .with(chrome)
        .init();

    let world = World::from_phrase(&args.seed, TerrainParams::default());
    let area = Aabb::centered(Point2::origin(), args.extent);

    let start = Instant::now();
    let mut region = None;
    for run in 0..args.runs {
        let _run = info_span!("run", run).entered();
        let r = world.region(area);
        let grid = HeightGrid::sample_with_gradients(&r, area, args.spacing);
        let (lo, hi) = grid.range();
        let mut lines = contours(&grid, levels(lo, hi, args.interval));
        refine_all(&mut lines, &r, args.refine);
        black_box(&lines);

        let mut biomes = BiomeMap::classify(&grid, &BiomeRules::default());
        biomes.smooth(2);
        biomes.absorb_small(args.min_area);
        black_box(&biomes);
        region = Some(r);
    }
    let pipeline = start.elapsed();
    let region = region.expect("--runs must be at least 1");

    let per_eval = bench_evals(&region, area, args.evals);
    drop(_flush_on_drop);

    report(&args, &region, &stats, pipeline, per_eval);
}

/// Mean time per `height` and per `height_and_gradient` call at random points in `area`.
fn bench_evals(region: &Region, area: Aabb, n: usize) -> (Duration, Duration) {
    let mut rng = Rng64::new(0);
    let points: Vec<_> = (0..n).map(|_| area.lerp(rng.unit(), rng.unit())).collect();
    let per = |d: Duration| d / n.max(1) as u32;

    let t = Instant::now();
    {
        let _s = info_span!("bench_height", n).entered();
        for &p in &points {
            black_box(region.height(black_box(p)));
        }
    }
    let height = per(t.elapsed());

    let t = Instant::now();
    {
        let _s = info_span!("bench_gradient", n).entered();
        for &p in &points {
            black_box(region.height_and_gradient(black_box(p)));
        }
    }
    (height, per(t.elapsed()))
}

fn report(
    args: &Args,
    region: &Region,
    stats: &SpanStats,
    pipeline: Duration,
    (height, gradient): (Duration, Duration),
) {
    let chunks: Vec<_> = region.chunks.values().collect();
    let nodes: usize = chunks.iter().map(|c| c.nodes).sum();
    let mut depths = [0; 16];
    for c in &chunks {
        depths[(c.depth as usize).min(15)] += 1;
    }
    let depths: Vec<String> = depths
        .iter()
        .enumerate()
        .filter(|(_, n)| **n > 0)
        .map(|(d, n)| format!("{d}:{n}"))
        .collect();

    println!(
        "seed {:?}, {} m area, {} m grid, {} m contours, {} refine iterations",
        args.seed, args.extent, args.spacing, args.interval, args.refine
    );
    println!(
        "{} chunks, {nodes} quadtree nodes, depth histogram {}",
        chunks.len(),
        depths.join(" ")
    );
    println!(
        "pipeline: {:.1} ms per run ({} runs)",
        ms(pipeline) / args.runs as f64,
        args.runs
    );
    println!("evaluation: height {height:.2?}, height + gradient {gradient:.2?}\n");

    let stats = stats.0.lock().unwrap();
    let mut rows: Vec<_> = stats.iter().collect();
    rows.sort_by(|a, b| b.1.total.cmp(&a.1.total));

    println!(
        "{:<28} {:>8} {:>11} {:>10} {:>10}",
        "span", "calls", "total ms", "mean ms", "max ms"
    );
    for (name, s) in rows {
        println!(
            "{:<28} {:>8} {:>11.2} {:>10.3} {:>10.3}",
            name,
            s.calls,
            ms(s.total),
            ms(s.total) / s.calls as f64,
            ms(s.max)
        );
    }

    if let Some(path) = &args.chrome {
        println!("\nChrome trace written to {path}");
    }
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

const USAGE: &str = "
Profiles terrain generation and contouring.

Usage: profile [OPTIONS]

  --seed <phrase>     World seed phrase                     [default: orienteering]
  --extent <m>        Side length of the profiled area      [default: 3000]
  --spacing <m>       Height grid spacing                   [default: 2]
  --interval <m>      Contour interval                      [default: 5]
  --refine <n>        Newton iterations per contour vertex  [default: 2]
  --runs <n>          Times to repeat the pipeline          [default: 3]
  --evals <n>         Points for the per-evaluation bench   [default: 200000]
  --min-area <m²>     Smallest biome region kept            [default: 400]
  --level <level>     Most verbose spans to record: info, debug or trace [default: debug]
  --chrome <path>     Also write a Chrome trace (open in https://ui.perfetto.dev)
";

/// Aggregates the wall time of every closed span by name.
#[derive(Clone, Default)]
struct SpanStats(Arc<Mutex<BTreeMap<String, Stat>>>);

#[derive(Default)]
struct Stat {
    calls: u64,
    total: Duration,
    max: Duration,
}

struct Opened(Instant);

impl<S> Layer<S> for SpanStats
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_new_span(&self, _: &Attributes<'_>, id: &Id, ctx: Context<'_, S>) {
        if let Some(span) = ctx.span(id) {
            span.extensions_mut().insert(Opened(Instant::now()));
        }
    }

    fn on_close(&self, id: Id, ctx: Context<'_, S>) {
        let Some(span) = ctx.span(&id) else { return };
        let Some(elapsed) = span.extensions().get::<Opened>().map(|o| o.0.elapsed()) else {
            return;
        };

        // e.g. `generate::chunk` for a span in `orienteering::terrain::generate`.
        let module = span.metadata().target().rsplit("::").next().unwrap_or("");
        let name = format!("{module}::{}", span.name());

        let mut stats = self.0.lock().unwrap();
        let stat = stats.entry(name).or_default();
        stat.calls += 1;
        stat.total += elapsed;
        stat.max = stat.max.max(elapsed);
    }
}
