//! Using scalar fields guarantees the resulting vector field is conservative, guaranteeing that
//! the line integral of a point to itself is zero, which is a requirement for closed contours.

use crate::{calculus::*, chain};
use enum_dispatch::enum_dispatch;
use nalgebra::{Point2, Vector2};

#[enum_dispatch]
pub trait ScalarField {
    /// Returns the value of the scalar field at the given point.
    fn phi(&self, xy: Point2<f32>) -> f32;

    /// v = ∇φ = [∂φ/∂x, ∂φ/∂y]
    fn gradient_at(&self, xy: Point2<f32>) -> Vector2<f32>;
}

impl<T> ScalarField for T
where
    T: std::ops::Deref<Target = FnXY>,
{
    fn phi(&self, xy: Point2<f32>) -> f32 {
        self.deref().phi(xy)
    }

    fn gradient_at(&self, xy: Point2<f32>) -> Vector2<f32> {
        self.deref().gradient_at(xy)
    }
}

#[derive(Debug, Clone, Default)]
pub struct SummedPotential {
    pub potentials: Vec<WeightedPotential>,
}

impl SummedPotential {
    pub fn new(potentials: Vec<WeightedPotential>) -> Self {
        Self { potentials }
    }
}

impl ScalarField for SummedPotential {
    fn phi(&self, xy: Point2<f32>) -> f32 {
        self.potentials.iter().map(|p| p.phi(xy)).sum()
    }

    fn gradient_at(&self, xy: Point2<f32>) -> Vector2<f32> {
        self.potentials.iter().map(|p| p.gradient_at(xy)).sum()
    }
}

#[derive(Debug, Clone)]
pub struct WeightedPotential {
    pub weight: f32,
    pub potential: Potential,
}

impl WeightedPotential {
    pub fn new(potential: Potential, weight: f32) -> Self {
        Self { weight, potential }
    }
}

impl ScalarField for WeightedPotential {
    fn phi(&self, xy: Point2<f32>) -> f32 {
        self.weight * self.potential.phi(xy)
    }
    fn gradient_at(&self, xy: Point2<f32>) -> Vector2<f32> {
        self.weight * self.potential.gradient_at(xy)
    }
}

#[derive(Debug, Clone)]
#[enum_dispatch(ScalarField)]
pub enum Potential {
    Circular(CircularPotential),
    Constant(ConstantPotential),
    Saddle(SaddlePotential),
    Oscillating(OscillatingPotential),
    Custom(FnXY),
}

impl Potential {
    pub fn constant(value: f32) -> Self {
        ConstantPotential(value).into()
    }
    pub fn circular() -> Self {
        CircularPotential::default().into()
    }
    pub fn saddle() -> Self {
        SaddlePotential::default().into()
    }
    pub fn oscillating(omega_x: f32, omega_y: f32) -> Self {
        OscillatingPotential::new(omega_x, omega_y).into()
    }
    pub fn custom(f: FnXY) -> Self {
        f.into()
    }
}

#[derive(Debug, Clone)]
pub struct LocalizedPotential {
    pub min: Point2<f32>,
    pub max: Point2<f32>,
    f: FnXY,
}

impl std::ops::Deref for LocalizedPotential {
    type Target = FnXY;
    fn deref(&self) -> &Self::Target {
        &self.f
    }
}

impl LocalizedPotential {
    pub fn new(min: Point2<f32>, max: Point2<f32>) -> Self {
        todo!()
    }
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

#[derive(Debug, Clone)]
pub struct ConstantPotential(pub f32);

impl ScalarField for ConstantPotential {
    fn phi(&self, _xy: Point2<f32>) -> f32 {
        self.0
    }
    fn gradient_at(&self, _xy: Point2<f32>) -> Vector2<f32> {
        Vector2::new(0., 0.)
    }
}

#[derive(Debug, Clone, derive_more::Deref)]
pub struct OscillatingPotential(FnXY);

impl Default for OscillatingPotential {
    fn default() -> Self {
        let fx = FnType::sin();
        let fy = FnType::cos();
        let f = SumOfFnType::new(fx, fy).into();
        Self(f)
    }
}

impl OscillatingPotential {
    pub fn new(omega_x: f32, omega_y: f32) -> Self {
        let ix = FnType::constant(omega_x) * FnType::linear();
        let fx = chain!(ix => FnType::sin());

        let iy = FnType::constant(omega_y) * FnType::linear();
        let fy = chain!(iy => FnType::cos());

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
