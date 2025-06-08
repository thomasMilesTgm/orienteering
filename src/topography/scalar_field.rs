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

#[derive(Debug, Clone)]
pub struct Fn2D {
    /// x(t)
    pub x_t: FnType,

    /// x(t)
    pub y_t: FnType,
}

impl<T> ScalarField for T
where
    T: AsRef<Fn2D>,
{
    fn phi(&self, xy: Point2<f32>) -> f32 {
        let fn2d = self.as_ref();
        let t = fn2d.x_t.t_at(xy.x).or(fn2d.y_t.t_at(xy.y)).unwrap();
        fn2d.x_t.f(t) + fn2d.y_t.f(t)
    }

    fn v_xy(&self, xy: Point2<f32>) -> Vector2<f32> {
        let fn2d = self.as_ref();
        let t = fn2d.x_t.t_at(xy.x).or(fn2d.y_t.t_at(xy.y)).unwrap();

        let dx_dt = fn2d.x_t.df_dt(t);
        let dy_dt = fn2d.y_t.df_dt(t);
        todo!()
    }
}

#[derive(Debug, Clone, derive_more::AsRef)]
pub struct CircularField(Fn2D);

impl Default for CircularField {
    fn default() -> Self {
        Self(Fn2D {
            x_t: Cos.into(),
            y_t: Sin.into(),
        })
    }
}
