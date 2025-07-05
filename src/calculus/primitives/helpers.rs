use super::*;

impl FunctionT {
    /* Math */
    /// `ln(self)`
    pub fn ln(self) -> Self {
        helpers::chain(self, helpers::ln())
    }

    /// `log_base(self)`
    pub fn log(self, base: Number) -> Self {
        helpers::chain(self, helpers::logarithm(base))
    }

    /// `self^exponent`
    pub fn pow(self, exponent: Number) -> Self {
        helpers::chain(self, helpers::power(exponent))
    }

    /// `e^self`
    pub fn exp(self) -> Self {
        helpers::chain(self, helpers::exp())
    }

    /// square root of `self`
    pub fn sqrt(self) -> Self {
        helpers::chain(self, helpers::power(0.5))
    }

    /// absolute value of `self`
    pub fn abs(self) -> Self {
        self.pow(2.).sqrt() // sqrt(f(t)^2)
    }

    /* Trig */
    pub fn acos(self) -> Self {
        helpers::chain(self, helpers::arccos())
    }

    pub fn asin(self) -> Self {
        helpers::chain(self, helpers::arcsin())
    }

    pub fn atan(self) -> Self {
        helpers::chain(self, helpers::arctan())
    }

    pub fn sin(self) -> Self {
        helpers::chain(self, helpers::sin())
    }

    pub fn cos(self) -> Self {
        helpers::chain(self, helpers::cos())
    }

    pub fn tan(self) -> Self {
        helpers::chain(self, helpers::tan())
    }

    /* Hyperbolic */
    pub fn acosh(self) -> Self {
        helpers::chain(self, helpers::arccosh())
    }

    pub fn asinh(self) -> Self {
        helpers::chain(self, helpers::arcsinh())
    }

    pub fn atanh(self) -> Self {
        helpers::chain(self, helpers::arctanh())
    }

    pub fn cosh(self) -> Self {
        helpers::chain(self, helpers::cosh())
    }

    pub fn sinh(self) -> Self {
        helpers::chain(self, helpers::sinh())
    }

    pub fn tanh(self) -> Self {
        helpers::chain(self, helpers::tanh())
    }
}

/// f(t) = outer(inner(t))
pub fn chain<L: Into<FunctionT>, R: Into<FunctionT>>(inner: L, outer: R) -> FunctionT {
    FunctionT::Chain(operator::Chain::new(inner, outer))
}

/// f(t) = |f(t)|
pub fn abs<F: Into<FunctionT>>(f: F) -> FunctionT {
    let f: FunctionT = f.into();
    f.abs()
}

/// f(t) = c
pub const fn constant(value: Number) -> FunctionT {
    FunctionT::Constant(arithmetic::Constant(value))
}
/// f(t) = t
pub const fn linear() -> FunctionT {
    FunctionT::Linear(arithmetic::Linear)
}
/// f(t) = t^exponent
pub const fn power(exponent: Number) -> FunctionT {
    FunctionT::Power(arithmetic::Power { exponent })
}
/// f(t) = base^t
pub const fn exponential(base: Number) -> FunctionT {
    FunctionT::Exponential(arithmetic::Exponential { base })
}
/// f(t) = e^t
pub const fn exp() -> FunctionT {
    FunctionT::Exponential(arithmetic::Exponential { base: consts::E })
}
/// f(t) = log_base(t)
pub const fn logarithm(base: Number) -> FunctionT {
    FunctionT::Logarithm(arithmetic::Logarithm { base })
}
/// f(t) = ln(t)
pub const fn ln() -> FunctionT {
    FunctionT::Logarithm(arithmetic::Logarithm { base: consts::E })
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

pub const fn undefined() -> FunctionT {
    FunctionT::Undefined(arithmetic::Undefined)
}

pub fn sec() -> FunctionT {
    constant(1.) / cos()
}
