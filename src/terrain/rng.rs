//! A tiny, fully specified PRNG so worlds are reproducible across platforms and dependency
//! versions (`rand`'s `SmallRng`/`StdRng` make no such guarantee).

/// SplitMix64.
#[derive(Debug, Clone)]
pub struct Rng64(u64);

impl Rng64 {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        mix(self.0)
    }

    /// Uniform in [0, 1).
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1. / (1u64 << 53) as f64)
    }

    /// Uniform in [lo, hi).
    pub fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }

    /// Uniform integer in [lo, hi].
    pub fn int(&mut self, lo: u32, hi: u32) -> u32 {
        lo + (self.next_u64() % u64::from(hi - lo + 1)) as u32
    }

    pub fn chance(&mut self, p: f64) -> bool {
        self.unit() < p
    }

    /// Either 1 or -1.
    pub fn sign(&mut self) -> f64 {
        if self.chance(0.5) { 1. } else { -1. }
    }
}

/// The SplitMix64 finalizer: a good 64-bit bit mixer.
pub fn mix(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Combines values into a single well-mixed seed.
pub fn hash(values: &[u64]) -> u64 {
    values
        .iter()
        .fold(0x6A09_E667_F3BC_C908, |h, &v| mix(h ^ mix(v)))
}

/// Turns a human-readable phrase into a seed (FNV-1a, then mixed).
pub fn seed_from_phrase(phrase: &str) -> u64 {
    let fnv = phrase.bytes().fold(0xCBF2_9CE4_8422_2325_u64, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01B3)
    });
    mix(fnv)
}
