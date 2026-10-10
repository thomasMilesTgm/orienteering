// Pass 2 of the GPU map: styles the terrain evaluated by pass 1 (terrain_field.wgsl), which it
// reads texel for texel: biome fills (from slope averaged over neighboring texels), relief shading,
// contours and the shoreline. Cheap enough to run every frame.

// Matches bevy_sprite_render's mesh2d vertex output.
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) world_position: vec4<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(5) @interpolate(flat) instance_index: u32,
}

struct Style {
    // contour interval, slope averaging offset in texels, number of biome rules, fallback biome.
    style: vec4<f32>,
    // Per rule: altitude min, max, slope min, max.
    rules: array<vec4<f32>, 8>,
    // Each rule's biome, four per vec4.
    rule_biomes: array<vec4<f32>, 2>,
    // Linear RGB fill per biome: water, marsh, meadow, forest, rock, alpine.
    colors: array<vec4<f32>, 6>,
}

@group(2) @binding(0) var<uniform> style: Style;
// (height, dh/dx, dh/dy) per physical pixel, from pass 1.
@group(2) @binding(1) var field: texture_2d<f32>;

const MARSH: u32 = 1u;

fn sample(p: vec2<i32>) -> vec3<f32> {
    let size = vec2<i32>(textureDimensions(field));
    return textureLoad(field, clamp(p, vec2(0), size - 1), 0).xyz;
}

fn biome(altitude: f32, slope: f32) -> u32 {
    for (var r = 0u; r < u32(style.style.z); r++) {
        let band = style.rules[r];
        if altitude >= band.x && altitude < band.y && slope >= band.z && slope < band.w {
            return u32(style.rule_biomes[r / 4u][r % 4u]);
        }
    }
    return u32(style.style.w);
}

fn line(distance_px: f32, half_width: f32) -> f32 {
    return 1.0 - smoothstep(half_width - 0.5, half_width + 0.5, distance_px);
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let p = vec2<i32>(floor(in.position.xy));
    let t = sample(p);
    let h = t.x;
    let g = t.yz;

    // Contours, from screen-space derivatives (which need uniform control flow, so first).
    let interval = style.style.x;
    let v = h / interval;
    let dv = max(fwidth(v), 1e-6);
    let dh = max(fwidth(h), 1e-6);

    // Slope averaged over the classification scale, as the CPU does, so summits and saddles
    // don't speckle as flat ground.
    let s = i32(round(style.style.y));
    var slope = length(g);
    slope += length(sample(p + vec2(s, s)).yz);
    slope += length(sample(p + vec2(-s, s)).yz);
    slope += length(sample(p + vec2(s, -s)).yz);
    slope += length(sample(p + vec2(-s, -s)).yz);
    slope /= 5.0;

    let b = biome(h, slope);
    var color = style.colors[b].rgb;
    if b == MARSH && u32(in.position.y) % 4u == 0u {
        color = vec3(0.0, 0.3, 0.7);
    }

    // Soft relief shading lit from the north-west.
    let normal = normalize(vec3(-g, 1.0));
    let light = normalize(vec3(-1.0, 1.0, 1.5));
    color *= 0.88 + 0.12 * dot(normal, light);

    let brown = vec3(0.49, 0.16, 0.02);
    let blue = vec3(0.0, 0.3, 0.7);
    if h >= 0.0 {
        let nearest = round(v);
        let index = abs(nearest - 5.0 * round(nearest / 5.0)) < 0.5;
        let distance = abs(v - nearest) / dv;
        color = mix(color, brown, line(distance, select(0.7, 1.4, index)));
    }
    // The shoreline.
    color = mix(color, blue, line(abs(h) / dh, 1.2));

    return vec4(color, 1.0);
}
