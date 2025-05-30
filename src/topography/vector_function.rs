//! # Vector Field Functions
//!
//! A vector field is defined by:
//!
//! ```text
//! Vx(x, y) = f(x, y);
//! Vy(x, y) = g(x, y);
//! ```
//!
//! Where `Vx` and `Vy` are the x and y components of the vector at the point `(x, y)`.
//!
//! This module defines various functions that can be used as `Vx`/`Vy`, which are all
//! [`SeedableRng`] so they can be randomized for use in procedural generation, while their general
//! 'shape' remains known so we can apply logic to how these function pairs can be composed to
//! create known vector field shapes.

use nalgebra::Point2;
use rand::Rng;

const CONSTANT_K_RANGE: std::ops::Range<f32> = -10.0..10.0;

const SIN_X2_Y2_K_RANGE: std::ops::Range<f32> = 1.0..5.0;
const SIN_X2_Y2_A_RANGE: std::ops::Range<f32> = 0.0..0.01;
const SIN_X2_Y2_B_RANGE: std::ops::Range<f32> = 0.0..0.01;

const K_XY_RANGE: std::ops::Range<f32> = -10.0..10.0;

const JXKY_J_RANGE: std::ops::Range<f32> = -10.0..10.0;
const JXKY_K_RANGE: std::ops::Range<f32> = -10.0..10.0;

/// A vector field function, used to define the x or y component of a vector field at a given
/// point in 2D space.
pub trait VectorFn {
    fn v_xy(&self, xy: Point2<f32>) -> f32;
    fn from_rng<T: Rng>(rng: &mut T) -> Self;
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ConstantDir {
    pub k: f32,
}

impl VectorFn for ConstantDir {
    fn v_xy(&self, _xy: Point2<f32>) -> f32 {
        self.k
    }

    fn from_rng<T: Rng>(rng: &mut T) -> Self {
        let k = rng.random_range(CONSTANT_K_RANGE);
        Self { k }
    }
}

/// V = k * sin(a * x^2 + b * y^2)
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SinX2Y2 {
    pub k: f32,
    pub a: f32,
    pub b: f32,
}

impl VectorFn for SinX2Y2 {
    fn v_xy(&self, xy: Point2<f32>) -> f32 {
        self.k * ((self.a * xy.x).powi(2) + (self.b * xy.y).powi(2)).sin()
    }
    fn from_rng<T: Rng>(rng: &mut T) -> Self {
        let k = rng.random_range(SIN_X2_Y2_K_RANGE);
        let a = rng.random_range(SIN_X2_Y2_A_RANGE);
        let b = rng.random_range(SIN_X2_Y2_B_RANGE);
        Self { k, a, b }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct KxMinus {
    pub k: f32,
}

impl VectorFn for KxMinus {
    fn v_xy(&self, xy: Point2<f32>) -> f32 {
        self.k * xy.x
    }
    fn from_rng<T: Rng>(rng: &mut T) -> Self {
        let k = -rng.random_range(K_XY_RANGE).abs();
        Self { k }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct KxPlus {
    pub k: f32,
}

impl VectorFn for KxPlus {
    fn v_xy(&self, xy: Point2<f32>) -> f32 {
        self.k * xy.x
    }
    fn from_rng<T: Rng>(rng: &mut T) -> Self {
        let k = rng.random_range(K_XY_RANGE).abs();
        Self { k }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct KyMinus {
    pub k: f32,
}

impl VectorFn for KyMinus {
    fn v_xy(&self, xy: Point2<f32>) -> f32 {
        self.k * xy.y
    }
    fn from_rng<T: Rng>(rng: &mut T) -> Self {
        let k = -rng.random_range(K_XY_RANGE).abs();
        Self { k }
    }
}
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct KyPlus {
    pub k: f32,
}

impl VectorFn for KyPlus {
    fn v_xy(&self, xy: Point2<f32>) -> f32 {
        self.k * xy.y
    }
    fn from_rng<T: Rng>(rng: &mut T) -> Self {
        let k = rng.random_range(K_XY_RANGE).abs();
        Self { k }
    }
}

/// V = j * x + k * y
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct JxKy {
    pub j: f32,
    pub k: f32,
}

impl VectorFn for JxKy {
    fn v_xy(&self, xy: Point2<f32>) -> f32 {
        self.j * xy.x + self.k * xy.y
    }
    fn from_rng<T: Rng>(rng: &mut T) -> Self {
        let j = rng.random_range(JXKY_J_RANGE);
        let k = rng.random_range(JXKY_K_RANGE);
        Self { j, k }
    }
}
