// Pass 1 of the GPU map: evaluates the terrain partway through generation per pixel, from
// flattened fields, writing (height, dh/dx, dh/dy) to a float texture. Pass 2
// (terrain_style.wgsl) styles it. This pass only runs when the view, stage or scene changes.
//
// The data layout and traversal mirror `orienteering::terrain::flat` (see its docs); its
// `FlatScene::height_and_gradient` is the CPU reference for `terrain` below.

// Matches bevy_sprite_render's mesh2d vertex output.
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) world_position: vec4<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(5) @interpolate(flat) instance_index: u32,
}

struct Params {
    // xy: view center - chunk table origin; zw: view center - province table origin (meters).
    view: vec4<f32>,
    // chunk size, province size, meters per pixel, stage.
    scale: vec4<f32>,
    // chunk table width, height; province table width, height.
    tables: vec4<u32>,
}

@group(2) @binding(0) var<uniform> params: Params;
@group(2) @binding(1) var<storage, read> nodes: array<vec4<f32>>;
@group(2) @binding(2) var<storage, read> features: array<vec4<f32>>;
// [start, end) node ranges: the chunk table, then the province table.
@group(2) @binding(3) var<storage, read> cells: array<vec2<u32>>;

const NODE: u32 = 6u;
const FEATURE: u32 = 2u;

// --- Duals: (value, d/dx, d/dy) ---

fn c(v: f32) -> vec3<f32> {
    return vec3(v, 0.0, 0.0);
}

fn dmul(a: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
    return vec3(a.x * b.x, a.yz * b.x + b.yz * a.x);
}

fn dexp(a: vec3<f32>) -> vec3<f32> {
    let e = exp(a.x);
    return vec3(e, a.yz * e);
}

fn dsin(a: vec3<f32>) -> vec3<f32> {
    return vec3(sin(a.x), a.yz * cos(a.x));
}

fn dtanh(a: vec3<f32>) -> vec3<f32> {
    // Clamped: tanh of large arguments can overflow on some implementations.
    let t = tanh(clamp(a.x, -15.0, 15.0));
    return vec3(t, a.yz * (1.0 - t * t));
}

// --- Fields ---

fn axis(t: vec3<f32>, a: f32, b: f32, blend: f32) -> vec3<f32> {
    let k = 1.0 / blend;
    return (dtanh((t - c(a)) * k) - dtanh((t - c(b)) * k)) * 0.5;
}

fn wave(x: vec3<f32>, y: vec3<f32>, dir: vec2<f32>, k: f32, phase: f32, amplitude: f32) -> vec3<f32> {
    return dsin((x * dir.x + y * dir.y) * k + c(phase)) * amplitude;
}

fn feature(i: u32, x: vec3<f32>, y: vec3<f32>) -> vec3<f32> {
    let f0 = features[i * FEATURE];
    let f1 = features[i * FEATURE + 1u];
    let kind = bitcast<u32>(f0.x);
    let a = f0.yz;
    let b = vec2(f0.w, f1.x);
    let rx = x - c(a.x);
    let ry = y - c(a.y);
    switch kind {
        case 0u: { return c(f1.y); }
        case 1u: { return rx * b.x + ry * b.y; }
        case 2u: { return dexp((dmul(rx, rx) + dmul(ry, ry)) * (-0.5 / (f1.y * f1.y))) * f1.z; }
        case 3u: {
            let d = rx * -b.y + ry * b.x;
            return dexp(dmul(d, d) * (-0.5 / (f1.y * f1.y))) * f1.z;
        }
        case 4u: { return dtanh((rx * b.x + ry * b.y) * (1.0 / f1.y)) * (f1.z * 0.5); }
        case 5u: { return wave(x, y, b, f1.y, f1.z, f1.w); }
        default: { return c(0.0); }
    }
}

// One cell's nodes, at cell-local position q: a pre-order traversal that skips subtrees outside
// their support or not yet generated at this stage.
fn eval_cell(start: u32, end: u32, q: vec2<f32>, stage: f32) -> vec3<f32> {
    var windows: array<vec3<f32>, 9>;
    windows[0] = c(1.0);
    var h = c(0.0);
    var i = start;
    loop {
        if i >= end {
            break;
        }
        let base = i * NODE;
        let n2 = nodes[base + 2u];
        let depth = bitcast<u32>(n2.z);
        let weight = clamp(stage - n2.y, 0.0, 1.0);
        let support = nodes[base + 1u];
        if weight == 0.0 || any(q < support.xy) || any(q > support.zw) {
            i = bitcast<u32>(n2.w);
            continue;
        }

        let x = vec3(q.x, 1.0, 0.0);
        let y = vec3(q.y, 0.0, 1.0);
        var window = c(1.0);
        if n2.x > 0.0 {
            let b = nodes[base];
            window = dmul(axis(x, b.x, b.z, n2.x), axis(y, b.y, b.w, n2.x));
        }
        windows[depth + 1u] = dmul(windows[depth], window);

        let n3 = nodes[base + 3u];
        let n4 = nodes[base + 4u];
        let n5 = nodes[base + 5u];
        var wx = x;
        var wy = y;
        if n4.x != 0.0 || n5.y != 0.0 {
            wx = x + wave(x, y, n3.xy, n3.z, n3.w, n4.x);
            wy = y + wave(x, y, n4.yz, n4.w, n5.x, n5.y);
        }

        var sum = c(0.0);
        for (var f = bitcast<u32>(n5.z); f < bitcast<u32>(n5.w); f++) {
            sum += feature(f, wx, wy);
        }
        h += dmul(windows[depth + 1u], sum) * weight;
        i += 1u;
    }
    return h;
}

// Sums a table's cells around a point (relative to the table's origin).
fn eval_table(p: vec2<f32>, size: f32, dims: vec2<u32>, offset: u32, reach: i32, stage: f32) -> vec3<f32> {
    var h = c(0.0);
    let center = vec2<i32>(floor(p / size));
    for (var dy = -reach; dy <= reach; dy++) {
        for (var dx = -reach; dx <= reach; dx++) {
            let cell = center + vec2(dx, dy);
            if any(cell < vec2(0)) || any(cell >= vec2<i32>(dims)) {
                continue;
            }
            let range = cells[offset + u32(cell.y) * dims.x + u32(cell.x)];
            if range.x < range.y {
                h += eval_cell(range.x, range.y, p - vec2<f32>(cell) * size, stage);
            }
        }
    }
    return h;
}

// Height and gradient at `offset` meters from the view center.
fn terrain(offset: vec2<f32>) -> vec3<f32> {
    let stage = params.scale.w;
    let chunks = params.tables.x * params.tables.y;
    return eval_table(params.view.zw + offset, params.scale.y, params.tables.zw, chunks, 2, stage)
        + eval_table(params.view.xy + offset, params.scale.x, params.tables.xy, 0u, 1, stage);
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4(terrain(in.world_position.xy * params.scale.z), 1.0);
}
