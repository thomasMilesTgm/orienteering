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
    fn chain<T: DifferentiableFn>(&self, t: f32, inner: T) -> f32 {
        self.df_dt(inner.f(t)) * inner.df_dt(t)
    }

    /// Apply the product rule, calculating `d/dt[f(t) * g(t)]`
    fn product<T: DifferentiableFn>(&self, t: f32, g: T) -> f32 {
        dbg!(self.f(t) * g.df_dt(t)) + dbg!(self.df_dt(t) * g.f(t))
    }

    /// Apply the quotient rule, calculating `d/dt[f(t) / g(t)]`, where `f` is `self` and `g` is
    /// `inner`
    fn quotient<T: DifferentiableFn>(&self, t: f32, g: T) -> f32 {
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
        self.outer.chain(t, *self.inner.clone())
    }
}

/// A product of [`FnType`]
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
        self.rhs.product(t, *self.lhs.clone())
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
    Product(FnProduct),
    Chain(FnChain),
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
    #[test]
    fn test_fn_product() {
        let f1 = FnType::Constant(Constant(2.0));

        assert_eq!(f1.f(1.0), 2.0);
        assert_eq!(f1.df_dt(1.0), 0.0);

        let f2 = FnType::Linear(Linear);

        assert_eq!(f2.f(1.0), 1.0);
        assert_eq!(f2.df_dt(1.0), 1.0);

        let product = FnProduct::new(f1, f2);

        assert_eq!(product.f(1.0), 2.0);
        assert_eq!(product.df_dt(1.0), 2.0);
    }
}
