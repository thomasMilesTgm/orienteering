//! Building blocks of equations.

use enum_dispatch::enum_dispatch;

pub type Number = f64;

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
    fn df_dt(&self) -> FunctionT;
}

/// A function of a single variable `t`.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[enum_dispatch(FnOfT, Differentiate, Integrate)]
pub enum FunctionT {
    Undefined(math::Undefined),
    /* Ops */
    Chain(operator::Chain),
    Plus(operator::Plus),
    Minus(operator::Minus),
    Times(operator::Times),
    Divide(operator::Divide),

    /* Math */
    Constant(math::Constant),
    Linear(math::Linear),
    Power(math::Power),
    Exponential(math::Exponential),
    Logarithm(math::Logarithm),

    /* Trig */
    Sin(trigonometric::Sin),
    Cos(trigonometric::Cos),
    Tan(trigonometric::Tan),
    ArcSin(trigonometric::ArcSin),
    ArcCos(trigonometric::ArcCos),
    ArcTan(trigonometric::ArcTan),

    /* Hyperbolic */
    Sinh(hyperbolic::Sinh),
    Cosh(hyperbolic::Cosh),
    Tanh(hyperbolic::Tanh),
    ArcSinh(hyperbolic::ArcSinh),
    ArcCosh(hyperbolic::ArcCosh),
    ArcTanh(hyperbolic::ArcTanh),
}

impl FunctionT {
    pub fn sqrt(self) -> Self {
        use helpers::*;
        chain(self, power(0.5))
    }

    pub fn abs(self) -> Self {
        self.pow(2.).sqrt() // sqrt(f(t)^2)
    }
    pub fn ln(self) -> Self {
        use helpers::*;
        chain(self, ln())
    }
    pub fn exp(self) -> Self {
        use helpers::*;
        chain(self, exp())
    }
    pub fn pow(self, exponent: Number) -> Self {
        use helpers::*;
        chain(self, power(exponent))
    }
}

mod helpers {
    use super::*;

    /// f(t) = outer(inner(t))
    pub fn chain<L: Into<FunctionT>, R: Into<FunctionT>>(inner: L, outer: R) -> FunctionT {
        FunctionT::Chain(operator::Chain::new(outer, inner))
    }

    /// f(t) = |f(t)|
    pub fn abs<F: Into<FunctionT>>(f: F) -> FunctionT {
        let f: FunctionT = f.into();
        f.abs()
    }

    /// f(t) = c
    pub const fn constant(value: Number) -> FunctionT {
        FunctionT::Constant(math::Constant(value))
    }
    /// f(t) = t
    pub const fn linear() -> FunctionT {
        FunctionT::Linear(math::Linear)
    }
    /// f(t) = t^exponent
    pub const fn power(exponent: Number) -> FunctionT {
        FunctionT::Power(math::Power { exponent })
    }
    /// f(t) = base^t
    pub const fn exponential(base: Number) -> FunctionT {
        FunctionT::Exponential(math::Exponential { base })
    }
    /// f(t) = e^t
    pub const fn exp() -> FunctionT {
        FunctionT::Exponential(math::Exponential { base: consts::E })
    }
    /// f(t) = log_base(t)
    pub const fn logarithm(base: Number) -> FunctionT {
        FunctionT::Logarithm(math::Logarithm { base })
    }
    /// f(t) = ln(t)
    pub const fn ln() -> FunctionT {
        FunctionT::Logarithm(math::Logarithm { base: consts::E })
    }
    /// f(t) = sin(t)
    pub const fn sin() -> FunctionT {
        FunctionT::Sin(trigonometric::Sin)
    }
    /// f(t) = cos(t)
    pub const fn cos() -> FunctionT {
        FunctionT::Cos(trigonometric::Cos)
    }
    /// f(t) = tan(t)
    pub const fn tan() -> FunctionT {
        FunctionT::Tan(trigonometric::Tan)
    }
    /// f(t) = arcsin(t)
    pub const fn arcsin() -> FunctionT {
        FunctionT::ArcSin(trigonometric::ArcSin)
    }
    /// f(t) = arccos(t)
    pub const fn arccos() -> FunctionT {
        FunctionT::ArcCos(trigonometric::ArcCos)
    }
    /// f(t) = arctan(t)
    pub const fn arctan() -> FunctionT {
        FunctionT::ArcTan(trigonometric::ArcTan)
    }
    /// f(t) = sinh(t)
    pub const fn sinh() -> FunctionT {
        FunctionT::Sinh(hyperbolic::Sinh)
    }
    /// f(t) = cosh(t)
    pub const fn cosh() -> FunctionT {
        FunctionT::Cosh(hyperbolic::Cosh)
    }
    /// f(t) = tanh(t)
    pub const fn tanh() -> FunctionT {
        FunctionT::Tanh(hyperbolic::Tanh)
    }
    /// f(t) = arcsinh(t)
    pub const fn arcsinh() -> FunctionT {
        FunctionT::ArcSinh(hyperbolic::ArcSinh)
    }
    /// f(t) = arccosh(t)
    pub const fn arccosh() -> FunctionT {
        FunctionT::ArcCosh(hyperbolic::ArcCosh)
    }
    /// f(t) = arctanh(t)
    pub const fn arctanh() -> FunctionT {
        FunctionT::ArcTanh(hyperbolic::ArcTanh)
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

pub mod operator {
    use super::math::Undefined;
    use super::prelude::*;
    use super::{FunctionT, Number};

    /// Use the
    #[derive(Debug, Clone)]
    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    pub struct Chain {
        outer: Box<FunctionT>,
        inner: Box<FunctionT>,
    }
    impl Chain {
        pub fn new<L: Into<FunctionT>, R: Into<FunctionT>>(inner: L, outer: R) -> Self {
            Self {
                outer: Box::new(outer.into()),
                inner: Box::new(inner.into()),
            }
        }
    }
    impl FnOfT for Chain {
        fn f(&self, t: Number) -> Number {
            self.outer.f(self.inner.f(t))
        }
    }
    impl Integrate for Chain {
        fn integral(&self) -> FunctionT {
            // TODO
            Undefined.into()
        }
    }
    impl Differentiate for Chain {
        fn df_dt(&self) -> FunctionT {
            chain((*self.inner).clone(), self.outer.df_dt()) * self.inner.df_dt()
        }
    }

    #[derive(Debug, Clone)]
    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    pub struct Plus {
        lhs: Box<FunctionT>,
        rhs: Box<FunctionT>,
    }
    impl FnOfT for Plus {
        fn f(&self, t: Number) -> Number {
            self.lhs.f(t) + self.rhs.f(t)
        }
    }
    impl Differentiate for Plus {
        fn df_dt(&self) -> FunctionT {
            self.lhs.df_dt() + self.rhs.df_dt()
        }
    }
    impl Integrate for Plus {
        fn integral(&self) -> FunctionT {
            self.lhs.integral() + self.rhs.integral()
        }
    }
    impl Plus {
        pub fn new<L: Into<FunctionT>, R: Into<FunctionT>>(lhs: L, rhs: R) -> Self {
            Self {
                lhs: Box::new(lhs.into()),
                rhs: Box::new(rhs.into()),
            }
        }
    }

    #[derive(Debug, Clone)]
    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    pub struct Minus {
        lhs: Box<FunctionT>,
        rhs: Box<FunctionT>,
    }
    impl FnOfT for Minus {
        fn f(&self, t: Number) -> Number {
            self.lhs.f(t) - self.rhs.f(t)
        }
    }
    impl Differentiate for Minus {
        fn df_dt(&self) -> FunctionT {
            self.lhs.df_dt() - self.rhs.df_dt()
        }
    }
    impl Integrate for Minus {
        fn integral(&self) -> FunctionT {
            self.lhs.integral() - self.rhs.integral()
        }
    }
    impl Minus {
        pub fn new<L: Into<FunctionT>, R: Into<FunctionT>>(lhs: L, rhs: R) -> Self {
            Self {
                lhs: Box::new(lhs.into()),
                rhs: Box::new(rhs.into()),
            }
        }
    }

    #[derive(Debug, Clone)]
    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    pub struct Times {
        lhs: Box<FunctionT>,
        rhs: Box<FunctionT>,
    }
    impl FnOfT for Times {
        fn f(&self, t: Number) -> Number {
            self.lhs.f(t) * self.rhs.f(t)
        }
    }
    impl Integrate for Times {
        fn integral(&self) -> FunctionT {
            // TODO
            Undefined.into()
        }
    }
    impl Differentiate for Times {
        fn df_dt(&self) -> FunctionT {
            self.lhs.df_dt() * (*self.rhs).clone() + self.rhs.df_dt() * (*self.lhs).clone()
        }
    }
    impl Times {
        pub fn new<L: Into<FunctionT>, R: Into<FunctionT>>(lhs: L, rhs: R) -> Self {
            Self {
                lhs: Box::new(lhs.into()),
                rhs: Box::new(rhs.into()),
            }
        }
    }

    #[derive(Debug, Clone)]
    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    pub struct Divide {
        lhs: Box<FunctionT>,
        rhs: Box<FunctionT>,
    }
    impl FnOfT for Divide {
        fn f(&self, t: Number) -> Number {
            self.lhs.f(t) / self.rhs.f(t)
        }
    }
    impl Integrate for Divide {
        fn integral(&self) -> FunctionT {
            // Integration of division is not straightforward and typically requires special techniques.
            // This is a placeholder; actual implementation would depend on the specific functions involved.
            Undefined.into()
        }
    }
    impl Differentiate for Divide {
        fn df_dt(&self) -> FunctionT {
            let f = (*self.lhs).clone();
            let g = (*self.rhs).clone();
            let df_dt = self.lhs.df_dt();
            let dg_dt = self.rhs.df_dt();

            // quotient rule: (f/g)' = (f' * g - f * g') / g^2
            (df_dt * g.clone() - f * dg_dt) / g.pow(2.)
        }
    }
    impl Divide {
        pub fn new<L: Into<FunctionT>, R: Into<FunctionT>>(lhs: L, rhs: R) -> Self {
            Self {
                lhs: Box::new(lhs.into()),
                rhs: Box::new(rhs.into()),
            }
        }
    }
}

pub mod math {
    use super::prelude::*;
    use super::{FunctionT, Number};

    #[derive(Debug, Clone, Copy)]
    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    pub struct Undefined;

    impl FnOfT for Undefined {
        fn f(&self, _: Number) -> Number {
            f64::NAN
        }
    }
    impl Integrate for Undefined {
        fn integral(&self) -> FunctionT {
            FunctionT::Undefined(Undefined)
        }
    }
    impl Differentiate for Undefined {
        fn df_dt(&self) -> FunctionT {
            FunctionT::Undefined(Undefined)
        }
    }

    /// f(t) = [`Constant::0`]
    #[derive(Debug, Clone, Copy)]
    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    pub struct Constant(pub Number);

    impl FnOfT for Constant {
        fn f(&self, _: Number) -> Number {
            self.0
        }
    }

    impl Integrate for Constant {
        fn integral(&self) -> FunctionT {
            linear() * *self
        }
    }

    impl Differentiate for Constant {
        fn df_dt(&self) -> FunctionT {
            constant(0.)
        }
    }

    /// f(t) = t
    #[derive(Debug, Clone, Copy)]
    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    pub struct Linear;

    impl FnOfT for Linear {
        fn f(&self, t: Number) -> Number {
            t
        }
    }

    impl Integrate for Linear {
        fn integral(&self) -> FunctionT {
            constant(0.5) * power(2.)
        }
    }

    impl Differentiate for Linear {
        fn df_dt(&self) -> FunctionT {
            constant(1.)
        }
    }

    /// f(t) = t^[`Power::exponent`]
    #[derive(Debug, Clone, Copy)]
    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    pub struct Power {
        pub exponent: Number,
    }

    impl FnOfT for Power {
        fn f(&self, t: Number) -> Number {
            t.powf(self.exponent)
        }
    }

    impl Integrate for Power {
        fn integral(&self) -> FunctionT {
            if self.exponent == -1.0 {
                ln() * linear() - linear()
            } else {
                constant(1. / (self.exponent + 1.)) * power(self.exponent + 1.0)
            }
        }
    }

    impl Differentiate for Power {
        fn df_dt(&self) -> FunctionT {
            if self.exponent == 0.0 {
                constant(0.)
            } else {
                constant(self.exponent) * power(self.exponent - 1.0)
            }
        }
    }

    /// f(t) = [`Exponential`]^t
    #[derive(Debug, Clone, Copy)]
    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    pub struct Exponential {
        pub base: Number,
    }

    impl FnOfT for Exponential {
        fn f(&self, t: Number) -> Number {
            self.base.powf(t)
        }
    }

    impl Integrate for Exponential {
        fn integral(&self) -> FunctionT {
            if self.base == super::consts::E {
                exp()
            } else {
                exponential(self.base) / constant(self.base.ln())
            }
        }
    }
    impl Differentiate for Exponential {
        fn df_dt(&self) -> FunctionT {
            if self.base == super::consts::E {
                exp()
            } else {
                exponential(self.base) * constant(self.base.ln())
            }
        }
    }

    /// f(t) = log_[`base`](Logarithm::base)(t)
    #[derive(Debug, Clone)]
    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    pub struct Logarithm {
        pub base: Number,
    }

    impl FnOfT for Logarithm {
        fn f(&self, t: Number) -> Number {
            t.log(self.base)
        }
    }

    impl Integrate for Logarithm {
        fn integral(&self) -> FunctionT {
            if self.base == super::consts::E {
                ln() * linear() - linear()
            } else {
                logarithm(self.base) * linear() - linear() / ln()
            }
        }
    }
    impl Differentiate for Logarithm {
        fn df_dt(&self) -> FunctionT {
            if self.base == super::consts::E {
                constant(1.) / linear()
            } else {
                constant(1.) / (constant(self.base.ln()) * linear())
            }
        }
    }
}

pub mod trigonometric {
    use super::FunctionT;
    use super::Number;
    use super::prelude::*;

    /// f(t) = sin(t)
    #[derive(Debug, Clone)]
    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    pub struct Sin;

    impl FnOfT for Sin {
        fn f(&self, t: Number) -> Number {
            t.sin()
        }
    }

    impl Integrate for Sin {
        fn integral(&self) -> FunctionT {
            cos() * constant(-1.)
        }
    }

    impl Differentiate for Sin {
        fn df_dt(&self) -> FunctionT {
            cos()
        }
    }

    /// f(t) = cos(t)
    #[derive(Debug, Clone)]
    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    pub struct Cos;

    impl FnOfT for Cos {
        fn f(&self, t: Number) -> Number {
            t.cos()
        }
    }

    impl Integrate for Cos {
        fn integral(&self) -> FunctionT {
            sin()
        }
    }

    impl Differentiate for Cos {
        fn df_dt(&self) -> FunctionT {
            sin() * constant(-1.)
        }
    }

    /// f(t) = tan(t)
    #[derive(Debug, Clone)]
    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    pub struct Tan;

    impl FnOfT for Tan {
        fn f(&self, t: Number) -> Number {
            t.tan()
        }
    }

    impl Integrate for Tan {
        fn integral(&self) -> FunctionT {
            constant(-1.) * chain(cos().abs(), ln()) // -ln(|cos(t)|)
        }
    }
    impl Differentiate for Tan {
        fn df_dt(&self) -> FunctionT {
            constant(1.) / (cos() * cos()) // 1 / sec^2(t)
        }
    }

    /// f(t) = arcsin(t)
    #[derive(Debug, Clone)]
    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    pub struct ArcSin;

    impl FnOfT for ArcSin {
        fn f(&self, t: Number) -> Number {
            t.asin()
        }
    }

    impl Integrate for ArcSin {
        fn integral(&self) -> FunctionT {
            arcsin() * linear() + (constant(1.) - power(2.)).sqrt()
        }
    }

    impl Differentiate for ArcSin {
        fn df_dt(&self) -> FunctionT {
            constant(1.) / (constant(1.) - linear().pow(2.)).sqrt() // 1 / sqrt(1 - t^2)
        }
    }

    /// f(t) = arccos(t)
    #[derive(Debug, Clone)]
    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    pub struct ArcCos;

    impl FnOfT for ArcCos {
        fn f(&self, t: Number) -> Number {
            t.acos()
        }
    }

    impl Integrate for ArcCos {
        fn integral(&self) -> FunctionT {
            arccos() * linear() - (constant(1.) - power(2.)).sqrt()
        }
    }
    impl Differentiate for ArcCos {
        fn df_dt(&self) -> FunctionT {
            constant(-1.) / (constant(1.) - linear().pow(2.)).sqrt() // -1 / sqrt(1 - t^2)
        }
    }

    /// f(t) = arcsin(t)
    #[derive(Debug, Clone)]
    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    pub struct ArcTan;

    impl FnOfT for ArcTan {
        fn f(&self, t: Number) -> Number {
            t.atan()
        }
    }
    impl Integrate for ArcTan {
        fn integral(&self) -> FunctionT {
            // atan(t) * x - 0.5 * ln(1 + t^2)
            arctan() * linear() - constant(0.5) * chain(constant(1.) + power(2.), ln())
        }
    }
    impl Differentiate for ArcTan {
        fn df_dt(&self) -> FunctionT {
            constant(1.) / (constant(1.) + linear().pow(2.)) // 1 / (1 + t^2)
        }
    }
}

pub mod hyperbolic {
    use super::prelude::*;

    /// f(t) = sinh(t)
    #[derive(Debug, Clone)]
    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    pub struct Sinh;
    impl FnOfT for Sinh {
        fn f(&self, t: f64) -> f64 {
            t.sinh()
        }
    }

    impl Integrate for Sinh {
        fn integral(&self) -> FunctionT {
            cosh()
        }
    }

    impl Differentiate for Sinh {
        fn df_dt(&self) -> FunctionT {
            cosh()
        }
    }

    /// f(t) = cosh(t)
    #[derive(Debug, Clone)]
    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    pub struct Cosh;

    impl FnOfT for Cosh {
        fn f(&self, t: f64) -> f64 {
            t.cosh()
        }
    }
    impl Integrate for Cosh {
        fn integral(&self) -> FunctionT {
            sinh()
        }
    }
    impl Differentiate for Cosh {
        fn df_dt(&self) -> FunctionT {
            sinh()
        }
    }

    /// f(t) = tanh(t)
    #[derive(Debug, Clone)]
    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    pub struct Tanh;

    impl FnOfT for Tanh {
        fn f(&self, t: f64) -> f64 {
            t.tanh()
        }
    }

    impl Integrate for Tanh {
        fn integral(&self) -> FunctionT {
            chain(cosh(), ln())
        }
    }
    impl Differentiate for Tanh {
        fn df_dt(&self) -> FunctionT {
            constant(1.) / (cosh() * cosh()) // 1 / cosh^2(t) = sech^2(t)
        }
    }

    /// f(t) = arcsinh(t)
    #[derive(Debug, Clone)]
    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    pub struct ArcSinh;

    impl FnOfT for ArcSinh {
        fn f(&self, t: f64) -> f64 {
            t.asinh()
        }
    }
    impl Integrate for ArcSinh {
        fn integral(&self) -> FunctionT {
            arcsinh() * linear() - (constant(1.) + power(2.)).sqrt()
        }
    }
    impl Differentiate for ArcSinh {
        fn df_dt(&self) -> FunctionT {
            constant(1.) / (constant(1.) + linear().pow(2.)).sqrt() // 1 / sqrt(1 + t^2)
        }
    }

    /// f(t) = arccosh(t)
    #[derive(Debug, Clone)]
    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    pub struct ArcCosh;

    impl FnOfT for ArcCosh {
        fn f(&self, t: f64) -> f64 {
            t.acosh()
        }
    }

    impl Integrate for ArcCosh {
        fn integral(&self) -> FunctionT {
            arccosh() * linear() - (power(2.) + constant(1.)).sqrt() * (power(2.) - constant(1.))
        }
    }

    impl Differentiate for ArcCosh {
        fn df_dt(&self) -> FunctionT {
            constant(1.) / (power(2.) - constant(1.)).sqrt()
        }
    }

    /// f(t) = arctanh(t)
    #[derive(Debug, Clone)]
    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    pub struct ArcTanh;

    impl FnOfT for ArcTanh {
        fn f(&self, t: f64) -> f64 {
            t.atanh()
        }
    }

    impl Integrate for ArcTanh {
        fn integral(&self) -> FunctionT {
            arctanh() * linear() + constant(0.5) * (constant(1.) - power(2.)).ln()
        }
    }

    impl Differentiate for ArcTanh {
        fn df_dt(&self) -> FunctionT {
            constant(1.) / (constant(1.) - power(2.)) // 1 / (1 - t^2)
        }
    }
}
