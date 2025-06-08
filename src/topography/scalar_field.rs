//! Using scalar fields guarantees the resulting vector field is conservative, guaranteeing that
//! the line integral of a point to itself is zero, which is a requirement for closed contours.

use crate::calculus::*;
use nalgebra::{Point2, Vector2};

pub trait ScalarField {
    /// Returns the value of the scalar field at the given point.
    fn phi(&self, xy: Point2<f32>) -> f32;

    /// v = ∇φ = [∂φ/∂x, ∂φ/∂y]
    fn v_xy(&self, xy: Point2<f32>) -> Vector2<f32>;
}

pub struct Fn2D {
    pub x: FnType,
    pub y: FnType,
}

impl Fn2D {
    pub fn f(&self, xy: Point2<f32>) -> f32 {
        self.x.f(xy.x) + self.y.f(xy.y)
    }

    pub fn df_dx(&self, xy: Point2<f32>) -> f32 {
        self.x.df_dt(xy.x) + self.y.df_dt(xy.y)
    }
}
