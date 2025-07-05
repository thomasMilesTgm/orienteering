//! Building blocks of equations.

use enum_dispatch::enum_dispatch;

pub type Number = f64;

pub mod arithmetic;
pub mod format;
mod helpers;
pub mod hyperbolic;
pub mod operator;
pub mod trigonometric;

pub mod prelude {
    pub use super::helpers::*;
    pub use super::{Differentiate, FnOfT, FunctionT, Integrate};
}

pub mod consts {
    pub use std::f64::consts::E;
}

#[enum_dispatch]
pub trait Integrate {
    fn integral(&self) -> FunctionT;
}

#[enum_dispatch]
pub trait FnOfT {
    fn f(&self, t: Number) -> Number;
}

#[enum_dispatch]
pub trait Differentiate {
    fn derivative(&self) -> FunctionT;
}

/// A function of a single variable `t`.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[enum_dispatch(FnOfT, Differentiate, Integrate)]
pub enum FunctionT {
    Undefined(arithmetic::Undefined),

    /* Ops */
    Chain(operator::Chain),
    Plus(operator::Plus),
    Minus(operator::Minus),
    Times(operator::Times),
    Divide(operator::Divide),

    /* Math */
    Constant(arithmetic::Constant),
    Exponential(arithmetic::Exponential),
    Linear(arithmetic::Linear),
    Logarithm(arithmetic::Logarithm),
    Power(arithmetic::Power),

    /* Trig */
    ArcCos(trigonometric::ArcCos),
    ArcSin(trigonometric::ArcSin),
    ArcTan(trigonometric::ArcTan),
    Cos(trigonometric::Cos),
    Sin(trigonometric::Sin),
    Tan(trigonometric::Tan),

    /* Hyperbolic */
    ArcCosh(hyperbolic::ArcCosh),
    ArcSinh(hyperbolic::ArcSinh),
    ArcTanh(hyperbolic::ArcTanh),
    Cosh(hyperbolic::Cosh),
    Sinh(hyperbolic::Sinh),
    Tanh(hyperbolic::Tanh),
}

impl std::ops::Neg for FunctionT {
    type Output = Self;
    fn neg(self) -> Self::Output {
        self * helpers::constant(-1.)
    }
}

impl<T: Into<FunctionT>> std::ops::Add<T> for FunctionT {
    type Output = Self;

    fn add(self, rhs: T) -> Self::Output {
        operator::Plus::new(self.clone(), rhs).into()
    }
}

impl<T: Into<FunctionT>> std::ops::Sub<T> for FunctionT {
    type Output = Self;
    fn sub(self, rhs: T) -> Self::Output {
        operator::Minus::new(self.clone(), rhs).into()
    }
}

impl<T: Into<FunctionT>> std::ops::Mul<T> for FunctionT {
    type Output = Self;
    fn mul(self, rhs: T) -> Self::Output {
        operator::Times::new(self.clone(), rhs).into()
    }
}

impl<T: Into<FunctionT>> std::ops::Div<T> for FunctionT {
    type Output = Self;
    fn div(self, rhs: T) -> Self::Output {
        operator::Divide::new(self.clone(), rhs).into()
    }
}

#[cfg(test)]
mod test {
    use super::prelude::*;

    #[test]
    fn trig() {
        assert_eq!(sin().derivative(), cos());
        assert_eq!(sin().integral(), -cos());
        assert_eq!(cos().derivative(), -sin());
        assert_eq!(cos().integral(), sin());
    }
}
