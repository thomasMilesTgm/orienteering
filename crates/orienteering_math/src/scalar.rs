//! Number types height fields can be evaluated with.
//!
//! Fields are written once, generically over [`Scalar`]. Evaluating with `f64` gives the height;
//! evaluating with [`Dual`] gives the height *and* its exact gradient (forward-mode automatic
//! differentiation), so no derivative is ever written by hand.

use nalgebra::Vector2;
use std::ops::{Add, Div, Mul, Neg, Sub};

pub trait Scalar:
    Copy
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
{
    fn constant(c: f64) -> Self;

    /// The plain value, discarding any derivative information.
    fn value(self) -> f64;

    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn exp(self) -> Self;
    fn tanh(self) -> Self;

    #[inline]

    fn scale(self, k: f64) -> Self {
        self * Self::constant(k)
    }

    #[inline]

    fn offset(self, c: f64) -> Self {
        self + Self::constant(c)
    }
}

impl Scalar for f64 {
    #[inline]
    fn constant(c: f64) -> Self {
        c
    }
    #[inline]
    fn value(self) -> f64 {
        self
    }
    #[inline]
    fn sin(self) -> Self {
        f64::sin(self)
    }
    #[inline]
    fn cos(self) -> Self {
        f64::cos(self)
    }
    #[inline]
    fn exp(self) -> Self {
        f64::exp(self)
    }
    #[inline]
    fn tanh(self) -> Self {
        f64::tanh(self)
    }
}

/// A value together with its gradient with respect to (x, y).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Dual {
    pub v: f64,
    pub d: Vector2<f64>,
}

impl Dual {
    #[inline]
    pub fn new(v: f64, d: Vector2<f64>) -> Self {
        Self { v, d }
    }

    /// Seeds for evaluating a field at `(x, y)`: `x` varies along x, `y` along y.
    #[inline]
    pub fn variables(x: f64, y: f64) -> (Self, Self) {
        (
            Self::new(x, Vector2::new(1., 0.)),
            Self::new(y, Vector2::new(0., 1.)),
        )
    }

    /// Applies a unary function `f` whose derivative at `self.v` is `df`.
    #[inline]
    fn map(self, f: f64, df: f64) -> Self {
        Self::new(f, self.d * df)
    }
}

impl Add for Dual {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self::new(self.v + rhs.v, self.d + rhs.d)
    }
}

impl Sub for Dual {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.v - rhs.v, self.d - rhs.d)
    }
}

impl Mul for Dual {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self::new(self.v * rhs.v, self.d * rhs.v + rhs.d * self.v)
    }
}

impl Div for Dual {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        let v = self.v / rhs.v;
        Self::new(v, (self.d - rhs.d * v) / rhs.v)
    }
}

impl Neg for Dual {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self::new(-self.v, -self.d)
    }
}

impl Scalar for Dual {
    #[inline]
    fn constant(c: f64) -> Self {
        Self::new(c, Vector2::zeros())
    }
    #[inline]
    fn value(self) -> f64 {
        self.v
    }
    #[inline]
    fn sin(self) -> Self {
        self.map(self.v.sin(), self.v.cos())
    }
    #[inline]
    fn cos(self) -> Self {
        self.map(self.v.cos(), -self.v.sin())
    }
    #[inline]
    fn exp(self) -> Self {
        let e = self.v.exp();
        self.map(e, e)
    }
    #[inline]
    fn tanh(self) -> Self {
        let t = self.v.tanh();
        self.map(t, 1. - t * t)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn product_and_chain_rules() {
        // f = sin(x * y) / exp(x), at (0.7, -1.3)
        let (x, y) = Dual::variables(0.7, -1.3);
        let f = (x * y).sin() / x.exp();

        let (xv, yv) = (0.7_f64, -1.3_f64);
        let dfdx = (yv * (xv * yv).cos() - (xv * yv).sin()) / xv.exp();
        let dfdy = xv * (xv * yv).cos() / xv.exp();

        assert_relative_eq!(f.v, (xv * yv).sin() / xv.exp());
        assert_relative_eq!(f.d.x, dfdx, epsilon = 1e-12);
        assert_relative_eq!(f.d.y, dfdy, epsilon = 1e-12);
    }
}
