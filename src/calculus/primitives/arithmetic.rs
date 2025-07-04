use super::prelude::*;
use super::{FunctionT, Number};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
    fn derivative(&self) -> FunctionT {
        FunctionT::Undefined(Undefined)
    }
}

/// f(t) = [`Constant::0`]
#[derive(Debug, Clone, Copy, PartialEq)]
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
    fn derivative(&self) -> FunctionT {
        constant(0.)
    }
}

/// f(t) = t
#[derive(Debug, Clone, Copy, PartialEq)]
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
    fn derivative(&self) -> FunctionT {
        constant(1.)
    }
}

/// f(t) = t^[`Power::exponent`]
#[derive(Debug, Clone, Copy, PartialEq)]
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
    fn derivative(&self) -> FunctionT {
        if self.exponent == 0.0 {
            constant(0.)
        } else {
            constant(self.exponent) * power(self.exponent - 1.0)
        }
    }
}

/// f(t) = [`Exponential`]^t
#[derive(Debug, Clone, Copy, PartialEq)]
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
    fn derivative(&self) -> FunctionT {
        if self.base == super::consts::E {
            exp()
        } else {
            exponential(self.base) * constant(self.base.ln())
        }
    }
}

/// f(t) = log_[`base`](Logarithm::base)(t)
#[derive(Debug, Clone, Copy, PartialEq)]
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
    fn derivative(&self) -> FunctionT {
        if self.base == super::consts::E {
            constant(1.) / linear()
        } else {
            constant(1.) / (constant(self.base.ln()) * linear())
        }
    }
}
