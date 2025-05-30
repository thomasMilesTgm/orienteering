//! Procedural generation utilities for the game.

use rand::{
    Rng, SeedableRng,
    rngs::{SmallRng, StdRng},
};

pub trait ProceduralValue {
    fn from_rng<T: Rng>(rng: &mut T) -> Self;
}

#[derive(Clone, Default, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MapSeed {
    bytes: [u8; 32],
}

impl MapSeed {
    pub fn from_string(string: String) -> Self {
        string
            .bytes()
            .enumerate()
            .fold(Self::default(), |mut new, (i, b)| {
                // Wrap around if there are > 32 chars in the string
                let ix = i % 32;
                // XOR the byte into the seed
                new.bytes[ix] ^= b;
                new
            })
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn into_small_rng(self) -> SmallRng {
        SmallRng::from_seed(self.bytes)
    }
    pub fn into_sdt_rng(self) -> StdRng {
        StdRng::from_seed(self.bytes)
    }
}

impl From<[u8; 32]> for MapSeed {
    fn from(value: [u8; 32]) -> Self {
        Self { bytes: value }
    }
}

impl AsRef<[u8]> for MapSeed {
    fn as_ref(&self) -> &[u8] {
        &self.bytes
    }
}

impl AsMut<[u8]> for MapSeed {
    fn as_mut(&mut self) -> &mut [u8] {
        &mut self.bytes
    }
}
