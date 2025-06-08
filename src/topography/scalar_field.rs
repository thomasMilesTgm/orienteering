//! Using scalar fields guarantees the resulting vector field is conservative, guaranteeing that
//! the line integral of a point to itself is zero, which is a requirement for closed contours.

use enum_dispatch::enum_dispatch;
use nalgebra::{Point2, Vector2};

pub trait ScalarField {
    /// Returns the value of the scalar field at the given point.
    fn phi(&self, xy: Point2<f32>) -> f32;

    /// v = ∇φ = [∂φ/∂x, ∂φ/∂y]
    fn v_xy(&self, xy: Point2<f32>) -> Vector2<f32>;
}

#[enum_dispatch]
pub trait DifferentiableFn {
    /// Calculates the value of the function at time `t`, i.e., `f(t)`
    fn f(&self, t: f32) -> f32;

    /// Calculates the derivative of `x(t)` with respect to `t`, i.e., `dx/dt`
    fn df_dt(&self, t: f32) -> f32;

    /// Apply the chain rule, calculating `d/dt[f(g(t))]` where `f` is `self` and `g` is `inner`
    fn chain<T: DifferentiableFn>(&self, t: f32, inner: &T) -> f32 {
        self.df_dt(inner.f(t)) * inner.df_dt(t)
    }

    /// Apply the product rule, calculating `d/dt[f(t) * g(t)]`
    fn product<T: DifferentiableFn>(&self, t: f32, g: &T) -> f32 {
        self.f(t) * g.df_dt(t) + self.df_dt(t) * g.f(t)
    }

    /// Apply the quotient rule, calculating `d/dt[f(t) / g(t)]`, where `f` is `self` and `g` is
    /// `inner`
    fn quotient<T: DifferentiableFn>(&self, t: f32, g: &T) -> f32 {
        (self.df_dt(t) * g.f(t) - self.f(t) * g.df_dt(t)) / g.f(t).powi(2)
    }
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

/// A product of [`FnType`]
#[derive(Debug, Clone)]
pub struct FnChain {
    inner: Box<FnType>,
    outer: Box<FnType>,
}

impl FnChain {
    pub fn new(inner: FnType, outer: FnType) -> Self {
        Self {
            inner: Box::new(inner),
            outer: Box::new(outer),
        }
    }
}

impl DifferentiableFn for FnChain {
    fn f(&self, t: f32) -> f32 {
        self.outer.f(self.inner.f(t))
    }

    fn df_dt(&self, t: f32) -> f32 {
        let outer = self.outer.as_ref();
        let inner = self.inner.as_ref();
        outer.chain(t, inner)
    }
}

/// A quotient of two [`FnType`]
#[derive(Debug, Clone)]
pub struct FnQuotient {
    /// The numerator function
    lhs: Box<FnType>,
    /// The denominator function
    rhs: Box<FnType>,
}

impl FnQuotient {
    pub fn new(lhs: FnType, rhs: FnType) -> Self {
        Self {
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        }
    }
}

impl DifferentiableFn for FnQuotient {
    fn f(&self, t: f32) -> f32 {
        self.lhs.f(t) / self.rhs.f(t)
    }
    fn df_dt(&self, t: f32) -> f32 {
        let lhs = self.lhs.as_ref();
        let rhs = self.rhs.as_ref();
        lhs.quotient(t, rhs)
    }
}

/// A product of two [`FnType`]
#[derive(Debug, Clone)]
pub struct FnProduct {
    lhs: Box<FnType>,
    rhs: Box<FnType>,
}

impl FnProduct {
    pub fn new(lhs: FnType, rhs: FnType) -> Self {
        Self {
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        }
    }
}

impl DifferentiableFn for FnProduct {
    fn f(&self, t: f32) -> f32 {
        self.lhs.f(t) * self.rhs.f(t)
    }

    fn df_dt(&self, t: f32) -> f32 {
        let lhs = self.lhs.as_ref();
        let rhs = self.rhs.as_ref();
        rhs.product(t, lhs)
    }
}

#[enum_dispatch(DifferentiableFn)]
#[derive(Debug, Clone)]
pub enum FnType {
    Constant(Constant),
    Exp(Exp),
    Linear(Linear),
    Log(Log),
    Power(Power),
    Sin(Sin),
    Chain(FnChain),
    Product(FnProduct),
    Quotient(FnQuotient),
}

impl FnType {
    pub fn constant(c: f32) -> Self {
        FnType::Constant(Constant(c))
    }

    pub fn exp() -> Self {
        FnType::Exp(Exp)
    }

    pub fn linear() -> Self {
        FnType::Linear(Linear)
    }

    pub fn log() -> Self {
        FnType::Log(Log)
    }

    pub fn power(k: f32) -> Self {
        FnType::Power(Power { k })
    }

    pub fn sin() -> Self {
        FnType::Sin(Sin)
    }

    pub fn product_of(lhs: FnType, rhs: FnType) -> Self {
        FnType::Product(FnProduct::new(lhs, rhs))
    }

    pub fn chain(outer: FnType, inner: FnType) -> Self {
        FnType::Chain(FnChain::new(inner, outer))
    }

    pub fn quotient_of(lhs: FnType, rhs: FnType) -> Self {
        FnType::Quotient(FnQuotient::new(lhs, rhs))
    }
}

impl std::ops::Mul for FnType {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self::product_of(self, rhs)
    }
}

impl std::ops::Div for FnType {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        Self::quotient_of(self, rhs)
    }
}

/// `f(t) = c`
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Constant(pub f32);

impl DifferentiableFn for Constant {
    fn f(&self, _: f32) -> f32 {
        self.0
    }

    fn df_dt(&self, _: f32) -> f32 {
        0.
    }
}

/// `f(t) = e^t`
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Exp;

impl DifferentiableFn for Exp {
    fn f(&self, t: f32) -> f32 {
        t.exp()
    }
    fn df_dt(&self, t: f32) -> f32 {
        t.exp()
    }
}

/// `f(t) = c * t`
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Linear;

impl DifferentiableFn for Linear {
    fn f(&self, t: f32) -> f32 {
        t
    }
    fn df_dt(&self, _: f32) -> f32 {
        1.
    }
}

/// `f(t) = ln(t)` (natural logarithm)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Log;

impl DifferentiableFn for Log {
    fn f(&self, t: f32) -> f32 {
        t.ln()
    }

    fn df_dt(&self, t: f32) -> f32 {
        1. / t
    }
}

/// Power `f(t) = t^k`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Power {
    pub k: f32,
}

impl DifferentiableFn for Power {
    fn f(&self, t: f32) -> f32 {
        t.powf(self.k)
    }

    fn df_dt(&self, t: f32) -> f32 {
        self.k * t.powf(self.k - 1.)
    }
}

/// `f(t) = sin(t)`
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sin;

impl DifferentiableFn for Sin {
    fn f(&self, r: f32) -> f32 {
        (r).sin()
    }

    fn df_dt(&self, t: f32) -> f32 {
        -t.cos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::TAU;

    /* --- Basic Calculus --- */

    /// f  = sin(x) * x^2
    /// f' = 2x * sin(x) - x^2 * cos(x)
    #[test]
    fn test_fn_product() {
        let f1 = FnType::sin();
        let f2 = FnType::power(2.);

        let product = FnProduct::new(f1, f2);

        let f_expect = |x: f32| x.powi(2) * x.sin();
        let df_expect = |x: f32| 2. * x * x.sin() - x.powf(2.) * x.cos();

        assert_eq!(product.f(1.), f_expect(1.));
        assert_eq!(product.df_dt(1.), df_expect(1.));
    }

    /// f  = sin(x^2)
    /// f' = -2x * cos(x^2)
    #[test]
    fn test_fn_chain() {
        let outer = FnType::sin();

        let inner = FnType::power(2.0);

        let chain = FnChain::new(inner, outer);

        let f_expect = |x: f32| x.powi(2).sin();
        let df_expect = |x: f32| -2. * x * x.powi(2).cos();

        assert_eq!(chain.f(1.), f_expect(1.));
        assert_eq!(chain.df_dt(1.), df_expect(1.));
    }

    /// f = x^2 / sin(x)
    /// f' = (2x * sin(x) - x^2 * cos(x)) / sin(x)^2
    #[test]
    fn test_fn_quotient() {
        let lhs = FnType::power(2.0);
        let rhs = FnType::sin();
        let quotient = lhs / rhs;

        let f_expect = |x: f32| x.powi(2) / x.sin();
        let df_expect = |x: f32| (2. * x * x.sin() + x.powf(2.) * x.cos()) / x.sin().powi(2);

        assert_eq!(
            quotient.f(1.),
            f_expect(1.),
            "Quotient Function Incorrect: q(t) = f(t) / g(t)"
        );
        assert_eq!(
            quotient.df_dt(1.),
            df_expect(1.),
            "Quotient Rule Derivative Failed"
        );
    }

    /* --- Primitive Functions --- */
    #[test]
    fn test_constant() {
        for i in 0..12 {
            let c = i as f32 / TAU;
            let constant = FnType::constant(c);
            assert_eq!(constant.f(1.0), c);
            assert_eq!(constant.df_dt(1.0), 0.0);
        }
    }
    #[test]
    fn test_exp() {
        let exp = FnType::exp();
        let f_expect = |x: f32| x.exp();
        let df_expect = |x: f32| x.exp();
        for i in 0..10 {
            let t = i as f32;
            assert_eq!(exp.f(t), f_expect(t));
            assert_eq!(exp.df_dt(t), df_expect(t));
        }
    }

    #[test]
    fn test_linear() {
        let linear = FnType::linear();
        let f_expect = |x: f32| x;
        let df_expect = |_: f32| 1.0;
        for i in 0..10 {
            let t = i as f32;
            assert_eq!(linear.f(t), f_expect(t));
            assert_eq!(linear.df_dt(t), df_expect(t));
        }
    }
    #[test]
    fn test_log() {
        let log = FnType::log();
        let f_expect = |x: f32| x.ln();
        let df_expect = |x: f32| 1. / x;
        for i in 1..10 {
            let t = i as f32;
            assert_eq!(log.f(t), f_expect(t));
            assert_eq!(log.df_dt(t), df_expect(t));
        }
    }
    #[test]
    fn test_power() {
        let power = FnType::power(2.0);
        let f_expect = |x: f32| x.powi(2);
        let df_expect = |x: f32| 2. * x;
        for i in 0..10 {
            let t = i as f32;
            assert_eq!(power.f(t), f_expect(t));
            assert_eq!(power.df_dt(t), df_expect(t));
        }
    }
    #[test]
    fn test_sin() {
        let sin = FnType::sin();
        let f_expect = |x: f32| x.sin();
        let df_expect = |x: f32| -x.cos();

        for i in 0..12 {
            let t = i as f32 / TAU;
            assert_eq!(sin.f(t), f_expect(t));
            assert_eq!(sin.df_dt(t), df_expect(t));
        }
    }
}
