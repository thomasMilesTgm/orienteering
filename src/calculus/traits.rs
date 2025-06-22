use enum_dispatch::enum_dispatch;

use super::primitives::{FunctionT, Number};

pub trait Integrate {
    fn f(&self) -> FunctionT;
}

#[enum_dispatch]
pub trait FnOfT {
    fn f(&self, t: Number) -> Number;
}

#[enum_dispatch]
pub trait Differentiate {
    fn df_dt(&self) -> FunctionT;
}
