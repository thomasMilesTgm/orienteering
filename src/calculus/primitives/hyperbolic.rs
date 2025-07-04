use super::prelude::*;

/// f(t) = sinh(t)
#[derive(Debug, Clone, Copy, PartialEq)]
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
    fn derivative(&self) -> FunctionT {
        cosh()
    }
}

/// f(t) = cosh(t)
#[derive(Debug, Clone, Copy, PartialEq)]
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
    fn derivative(&self) -> FunctionT {
        sinh()
    }
}

/// f(t) = tanh(t)
#[derive(Debug, Clone, Copy, PartialEq)]
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
    fn derivative(&self) -> FunctionT {
        constant(1.) / (cosh() * cosh()) // 1 / cosh^2(t) = sech^2(t)
    }
}

/// f(t) = arcsinh(t)
#[derive(Debug, Clone, Copy, PartialEq)]
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
    fn derivative(&self) -> FunctionT {
        constant(1.) / (constant(1.) + linear().pow(2.)).sqrt() // 1 / sqrt(1 + t^2)
    }
}

/// f(t) = arccosh(t)
#[derive(Debug, Clone, Copy, PartialEq)]
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
    fn derivative(&self) -> FunctionT {
        constant(1.) / (power(2.) - constant(1.)).sqrt()
    }
}

/// f(t) = arctanh(t)
#[derive(Debug, Clone, Copy, PartialEq)]
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
    fn derivative(&self) -> FunctionT {
        constant(1.) / (constant(1.) - power(2.)) // 1 / (1 - t^2)
    }
}
