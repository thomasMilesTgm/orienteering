//! Math shared by the orienteering crates: rectangles, numbers that carry their own gradient, and a
//! reproducible random number generator.

mod aabb;
pub mod rng;
mod scalar;

pub use aabb::Aabb;
pub use scalar::{Dual, Scalar};
