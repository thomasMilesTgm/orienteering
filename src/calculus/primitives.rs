//! Building blocks of equations.

pub type Number = f64;

pub mod consts {
    use super::*;
    use std::f64::consts::E;

    /// e^t
    pub const EXP: Exponential = Exponential { base: E };

    /// Natural log
    pub const LN: Logarithm = Logarithm { base: E };
}

/// f(t) = [`Constant::k`]
pub struct Constant {
    pub k: Number,
}

/// f(t) = t
pub struct Linear;

/// f(t) = t^[`Power::exponent`]
pub struct Power {
    pub exponent: Number,
}

/// f(t) = [`Exponential`]^t
pub struct Exponential {
    pub base: Number,
}

/// f(t) = log_[`base`](Logarithm::base)(t)
pub struct Logarithm {
    pub base: Number,
}

pub mod trigonometric {
    /// f(t) = sin(t)
    pub struct Sin;

    /// f(t) = cos(t)
    pub struct Cos;

    /// f(t) = tan(t)
    pub struct Tan;

    /// f(t) = arcsin(t)
    pub struct ArcSin;

    /// f(t) = arcsin(t)
    pub struct ArcTan;

    /// f(t) = arccos(t)
    pub struct ArcCos;
}

pub mod hyperbolic {
    /// f(t) = sinh(t)
    pub struct Sinh;

    /// f(t) = cosh(t)
    pub struct Cosh;

    /// f(t) = tanh(t)
    pub struct Tanh;

    /// f(t) = arcsinh(t)
    pub struct ArcSinh;

    /// f(t) = arccosh(t)
    pub struct ArcCosh;

    /// f(t) = arctanh(t)
    pub struct ArcTanh;
}
