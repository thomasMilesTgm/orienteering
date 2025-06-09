//! Using scalar fields guarantees the resulting vector field is conservative, guaranteeing that
//! the line integral of a point to itself is zero, which is a requirement for closed contours.

use crate::calculus::*;
use enum_dispatch::enum_dispatch;
use nalgebra::{Point2, Vector2};

#[enum_dispatch]
pub trait ScalarField {
    /// Returns the value of the scalar field at the given point.
    fn phi(&self, xy: Point2<f32>) -> f32;

    /// v = ∇φ = [∂φ/∂x, ∂φ/∂y]
    fn gradient_at(&self, xy: Point2<f32>) -> Vector2<f32>;
}

#[derive(Debug, Clone, derive_more::Deref)]
pub struct CircularPotential(FnXY);

impl Default for CircularPotential {
    fn default() -> Self {
        let fx = FnType::power(2.) * FnType::constant(-1.);
        let fy = FnType::power(2.) * FnType::constant(-1.);
        let f = SumOfFnType::new(fx, fy).into();
        Self(f)
    }
}

#[derive(Debug, Clone, derive_more::Deref)]
pub struct SaddlePotential(FnXY);

impl Default for SaddlePotential {
    fn default() -> Self {
        let fx = FnType::power(2.) * FnType::constant(1.);
        let fy = FnType::power(2.) * FnType::constant(-1.);
        let f = SumOfFnType::new(fx, fy).into();
        Self(f)
    }
}

#[enum_dispatch(ScalarField)]
#[derive(Debug, Clone)]
pub enum FnXY {
    ProuctOfFnType(ProductOfFnType),
    SumOfFnType(SumOfFnType),
    ProductOfFnXY(ProductOfFnXY),
    SumOfFnXY(SumOfFnXY),
}

#[derive(Debug, Clone)]
pub struct SumOfFnXY {
    lhs: Box<FnXY>,
    rhs: Box<FnXY>,
}

impl ScalarField for SumOfFnXY {
    fn phi(&self, xy: Point2<f32>) -> f32 {
        self.lhs.phi(xy) + self.rhs.phi(xy)
    }
    fn gradient_at(&self, xy: Point2<f32>) -> Vector2<f32> {
        self.lhs.gradient_at(xy) + self.rhs.gradient_at(xy)
    }
}

impl SumOfFnXY {
    pub fn new(lhs: FnXY, rhs: FnXY) -> Self {
        Self {
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ProductOfFnXY {
    lhs: Box<FnXY>,
    rhs: Box<FnXY>,
}

impl ScalarField for ProductOfFnXY {
    fn phi(&self, xy: Point2<f32>) -> f32 {
        self.lhs.phi(xy) * self.rhs.phi(xy)
    }
    fn gradient_at(&self, xy: Point2<f32>) -> Vector2<f32> {
        Vector2::new(self.df_dx(xy), self.df_dy(xy))
    }
}

impl ProductOfFnXY {
    pub fn new(lhs: FnXY, rhs: FnXY) -> Self {
        Self {
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        }
    }
    pub fn df_dx(&self, xy: Point2<f32>) -> f32 {
        let dx = self.lhs.gradient_at(xy).x;
        let dy = self.rhs.phi(xy);
        dx * dy
    }
    pub fn df_dy(&self, xy: Point2<f32>) -> f32 {
        let dx = self.lhs.phi(xy);
        let dy = self.rhs.gradient_at(xy).y;
        dx * dy
    }
}

/// The product of a function of x and a function of y
#[derive(Debug, Clone)]
pub struct ProductOfFnType {
    fx: FnType,
    fy: FnType,
}

impl ScalarField for ProductOfFnType {
    fn phi(&self, xy: Point2<f32>) -> f32 {
        self.fx.f(xy.x) * self.fy.f(xy.y)
    }
    fn gradient_at(&self, xy: Point2<f32>) -> Vector2<f32> {
        Vector2::new(self.df_dx(xy), self.df_dy(xy))
    }
}

impl ProductOfFnType {
    pub fn new(fx: FnType, fy: FnType) -> Self {
        Self { fx, fy }
    }

    pub fn df_dx(&self, xy: Point2<f32>) -> f32 {
        let dx = self.fx.df_dt(xy.x);
        let dy = self.fy.f(xy.y);
        dx * dy
    }

    pub fn df_dy(&self, xy: Point2<f32>) -> f32 {
        let dx = self.fx.f(xy.x);
        let dy = self.fy.df_dt(xy.y);
        dx * dy
    }
}

#[derive(Debug, Clone)]
pub struct SumOfFnType {
    fx: FnType,
    fy: FnType,
}

impl ScalarField for SumOfFnType {
    fn phi(&self, xy: Point2<f32>) -> f32 {
        self.fx.f(xy.x) + self.fy.f(xy.y)
    }
    fn gradient_at(&self, xy: Point2<f32>) -> Vector2<f32> {
        Vector2::new(self.df_dx(xy), self.df_dy(xy))
    }
}

impl SumOfFnType {
    pub fn new(fx: FnType, fy: FnType) -> Self {
        Self { fx, fy }
    }
    pub fn df_dx(&self, xy: Point2<f32>) -> f32 {
        self.fx.df_dt(xy.x)
    }
    pub fn df_dy(&self, xy: Point2<f32>) -> f32 {
        self.fy.df_dt(xy.y)
    }
}
