use super::FunctionT;
use super::Number;
use super::prelude::*;

/// f(t) = sin(t)
#[derive(Debug, Clone, Copy, PartialEq)]
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
    fn derivative(&self) -> FunctionT {
        cos()
    }
}

/// f(t) = cos(t)
#[derive(Debug, Clone, Copy, PartialEq)]
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
    fn derivative(&self) -> FunctionT {
        sin() * constant(-1.)
    }
}

/// f(t) = tan(t)
#[derive(Debug, Clone, Copy, PartialEq)]
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
    fn derivative(&self) -> FunctionT {
        constant(1.) / (cos() * cos()) // 1 / sec^2(t)
    }
}

/// f(t) = arcsin(t)
#[derive(Debug, Clone, Copy, PartialEq)]
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
    fn derivative(&self) -> FunctionT {
        constant(1.) / (constant(1.) - linear().pow(2.)).sqrt() // 1 / sqrt(1 - t^2)
    }
}

/// f(t) = arccos(t)
#[derive(Debug, Clone, Copy, PartialEq)]
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
    fn derivative(&self) -> FunctionT {
        constant(-1.) / (constant(1.) - linear().pow(2.)).sqrt() // -1 / sqrt(1 - t^2)
    }
}

/// f(t) = arcsin(t)
#[derive(Debug, Clone, Copy, PartialEq)]
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
    fn derivative(&self) -> FunctionT {
        constant(1.) / (constant(1.) + linear().pow(2.)) // 1 / (1 + t^2)
    }
}
