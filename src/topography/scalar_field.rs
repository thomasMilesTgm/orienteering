//! Using scalar fields guarantees the resulting vector field is conservative, guaranteeing that
//! the line integral of a point to itself is zero, which is a requirement for closed contours.

use nalgebra::{Point2, Vector2};

pub trait ScalarField {
    /// Returns the value of the scalar field at the given point.
    fn phi(&self, xy: Point2<f32>) -> f32;

    /// v = ∇φ = [∂φ/∂x, ∂φ/∂y]
    fn v_xy(&self, xy: Point2<f32>) -> Vector2<f32>;
}

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
        self.f(t) * g.df_dt(t) + self.df_dt(t) * g.f(t)
    }

    /// Apply the quotient rule, calculating `d/dt[f(t) / g(t)]`, where `f` is `self` and `g` is
    /// `inner`
    fn quotient<T: DifferentiableFn>(&self, t: f32, g: T) -> f32 {
        (self.df_dt(t) * g.f(t) - self.f(t) * g.df_dt(t)) / g.f(t).powi(2)
    }
}

/// `f(t) = c`
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
pub struct Linear {
    pub k: f32,
}

impl DifferentiableFn for Linear {
    fn f(&self, t: f32) -> f32 {
        self.k * t
    }
    fn df_dt(&self, _: f32) -> f32 {
        self.k
    }
}

/// `f(t) = ln(t)` (natural logarithm)
pub struct Ln;

impl DifferentiableFn for Ln {
    fn f(&self, t: f32) -> f32 {
        t.ln()
    }

    fn df_dt(&self, t: f32) -> f32 {
        1. / t
    }
}

/// Power `f(t) = t^k`.
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
pub struct Sin;

impl DifferentiableFn for Sin {
    fn f(&self, r: f32) -> f32 {
        (r).sin()
    }

    fn df_dt(&self, t: f32) -> f32 {
        -t.cos()
    }
}
