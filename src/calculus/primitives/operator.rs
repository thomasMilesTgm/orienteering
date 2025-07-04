use super::arithmetic::Undefined;
use super::prelude::*;
use super::{FunctionT, Number};

/// Use the
#[derive(Debug, Clone, PartialEq)]
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
    fn derivative(&self) -> FunctionT {
        chain((*self.inner).clone(), self.outer.derivative()) * self.inner.derivative()
    }
}

#[derive(Debug, Clone, PartialEq)]
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
    fn derivative(&self) -> FunctionT {
        self.lhs.derivative() + self.rhs.derivative()
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

#[derive(Debug, Clone, PartialEq)]
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
    fn derivative(&self) -> FunctionT {
        self.lhs.derivative() - self.rhs.derivative()
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

#[derive(Debug, Clone, PartialEq)]
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
    fn derivative(&self) -> FunctionT {
        self.lhs.derivative() * (*self.rhs).clone() + self.rhs.derivative() * (*self.lhs).clone()
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

#[derive(Debug, Clone, PartialEq)]
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
    fn derivative(&self) -> FunctionT {
        let f = (*self.lhs).clone();
        let g = (*self.rhs).clone();
        let df_dt = self.lhs.derivative();
        let dg_dt = self.rhs.derivative();

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
