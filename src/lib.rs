//! # Orienteering
//!
//! Rust library for procedural world generation.

pub mod topography;

pub mod seed {
    #[derive(Clone)]
    pub struct MapSeed {
        bytes: Vec<u8>,
    }

    impl Default for MapSeed {
        fn default() -> Self {
            Self { bytes: vec![0; 32] }
        }
    }

    impl MapSeed {
        pub fn from_string(string: String) -> Self {
            Self::from(string.as_bytes())
        }
    }

    impl From<&[u8]> for MapSeed {
        fn from(value: &[u8]) -> Self {
            Self {
                bytes: value.to_vec(),
            }
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
}
