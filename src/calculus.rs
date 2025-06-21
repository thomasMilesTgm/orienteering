//! Basic calculus utilities for composing differential functions

use std::f32::consts::E;

use enum_dispatch::enum_dispatch;

#[enum_dispatch]
pub trait DifferentiableFn {
    /// Calculates the value of the function at time `t`, i.e., `f(t)`
    fn f(&self, t: f32) -> f32;

    /// Calculates the derivative of `x(t)` with respect to `t`, i.e., `dx/dt`
    fn df_dt(&self, t: f32) -> f32;

    /// Apply the chain rule, calculating `d/dt[f(g(t))]` where `f` is `self` and `g` is `inner`
    fn chain<T: DifferentiableFn>(&self, t: f32, inner: &T) -> f32 {
        self.df_dt(inner.f(t)) * inner.df_dt(t)
    }

    /// Apply the product rule, calculating `d/dt[f(t) * g(t)]`
    fn product<T: DifferentiableFn>(&self, t: f32, g: &T) -> f32 {
        self.f(t) * g.df_dt(t) + self.df_dt(t) * g.f(t)
    }

    /// Apply the quotient rule, calculating `d/dt[f(t) / g(t)]`, where `f` is `self` and `g` is
    /// `inner`
    fn quotient<T: DifferentiableFn>(&self, t: f32, g: &T) -> f32 {
        (self.df_dt(t) * g.f(t) - self.f(t) * g.df_dt(t)) / g.f(t).powi(2)
    }
}

#[enum_dispatch(DifferentiableFn)]
#[derive(Debug, Clone)]
pub enum FnType {
    Tanh(Tanh),
    StepRegion(StepRegion),
    Constant(Constant),
    Exp(Exp),
    Linear(Linear),
    Log(Log),
    Power(Power),
    Sin(Sin),
    Cos(Cos),
    Chain(FnChain),
    Product(FnProduct),
    Quotient(FnQuotient),
    Sum(FnSum),
    Sub(FnSub),
}
impl std::fmt::Display for FnType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FnType::Tanh(fn_step) => write!(f, "{}", fn_step),
            FnType::StepRegion(fn_region) => write!(f, "{}", fn_region),
            FnType::Constant(c) => write!(f, "{}", c),
            FnType::Exp(exp) => write!(f, "{}", exp),
            FnType::Linear(linear) => write!(f, "{}", linear),
            FnType::Log(log) => write!(f, "{}", log),
            FnType::Power(power) => write!(f, "{}", power),
            FnType::Sin(sin) => write!(f, "{}", sin),
            FnType::Cos(cos) => write!(f, "{}", cos),
            FnType::Chain(chain) => write!(f, "{}", chain),
            FnType::Product(product) => write!(f, "{}", product),
            FnType::Quotient(quotient) => write!(f, "{}", quotient),
            FnType::Sum(sum) => write!(f, "{}", sum),
            FnType::Sub(sub) => write!(f, "{}", sub),
        }
    }
}

impl FnType {
    pub fn constant(c: f32) -> Self {
        FnType::Constant(Constant(c))
    }

    pub fn exp() -> Self {
        FnType::Exp(Exp)
    }

    pub fn linear() -> Self {
        FnType::Linear(Linear)
    }

    pub fn log() -> Self {
        FnType::Log(Log)
    }

    pub fn power(k: f32) -> Self {
        FnType::Power(Power(k))
    }

    pub fn sin() -> Self {
        FnType::Sin(Sin)
    }

    pub fn cos() -> Self {
        FnType::Cos(Cos)
    }

    pub fn product_of<L: Into<FnType>, R: Into<FnType>>(lhs: L, rhs: R) -> Self {
        FnType::Product(FnProduct::new(lhs, rhs))
    }

    pub fn chain<L: Into<FnType>, R: Into<FnType>>(outer: L, inner: R) -> Self {
        FnType::Chain(FnChain::new(inner, outer))
    }

    pub fn quotient_of<L: Into<FnType>, R: Into<FnType>>(lhs: L, rhs: R) -> Self {
        FnType::Quotient(FnQuotient::new(lhs, rhs))
    }

    pub fn sum_of<L: Into<FnType>, R: Into<FnType>>(lhs: L, rhs: R) -> Self {
        FnType::Sum(FnSum::new(lhs, rhs))
    }
    pub fn subtract<L: Into<FnType>, R: Into<FnType>>(lhs: L, rhs: R) -> Self {
        FnType::Sub(FnSub::new(lhs, rhs))
    }

    pub fn tanh() -> Self {
        FnType::Tanh(Tanh)
    }

    pub fn step_region(start: f32, end: f32, k: f32) -> Self {
        FnType::StepRegion(StepRegion::new(start, end, k))
    }
}

impl std::ops::Mul<f32> for FnType {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        Self::product_of(self, Constant(rhs))
    }
}

impl<T: Into<Self>> std::ops::Mul<T> for FnType {
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        Self::product_of(self, rhs)
    }
}

impl std::ops::Div<f32> for FnType {
    type Output = Self;
    fn div(self, rhs: f32) -> Self::Output {
        Self::quotient_of(self, Constant(rhs))
    }
}

impl<T: Into<Self>> std::ops::Div<T> for FnType {
    type Output = Self;
    fn div(self, rhs: T) -> Self::Output {
        Self::quotient_of(self, rhs)
    }
}

impl std::ops::Add<f32> for FnType {
    type Output = Self;
    fn add(self, rhs: f32) -> Self::Output {
        FnType::Sum(FnSum::new(self, Constant(rhs)))
    }
}

impl<T: Into<Self>> std::ops::Add<T> for FnType {
    type Output = Self;
    fn add(self, rhs: T) -> Self::Output {
        FnType::Sum(FnSum::new(self, rhs))
    }
}

impl std::ops::Sub<f32> for FnType {
    type Output = Self;
    fn sub(self, rhs: f32) -> Self::Output {
        FnType::subtract(self, Constant(rhs))
    }
}

impl<T: Into<Self>> std::ops::Sub<T> for FnType {
    type Output = Self;
    fn sub(self, rhs: T) -> Self::Output {
        FnType::subtract(self, rhs)
    }
}

/// A product of [`FnType`]
#[derive(Debug, Clone)]
pub struct FnChain {
    inner: Box<FnType>,
    outer: Box<FnType>,
}

impl std::fmt::Display for FnChain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let outer = format!("{}", self.outer);
        let inner = format!("{}", self.inner);
        let fo_in = outer.replace("%", &inner);

        write!(f, "{}", fo_in)
    }
}

impl FnChain {
    pub fn new<L: Into<FnType>, R: Into<FnType>>(inner: L, outer: R) -> Self {
        Self {
            inner: Box::new(inner.into()),
            outer: Box::new(outer.into()),
        }
    }
}

impl DifferentiableFn for FnChain {
    fn f(&self, t: f32) -> f32 {
        self.outer.f(self.inner.f(t))
    }

    fn df_dt(&self, t: f32) -> f32 {
        let outer = self.outer.as_ref();
        let inner = self.inner.as_ref();
        outer.chain(t, inner)
    }
}

/// A quotient of two [`FnType`]
#[derive(Debug, Clone)]
pub struct FnQuotient {
    /// The numerator function
    lhs: Box<FnType>,
    /// The denominator function
    rhs: Box<FnType>,
}

impl std::fmt::Display for FnQuotient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}/{}", self.lhs, self.rhs)
    }
}

impl FnQuotient {
    pub fn new<L: Into<FnType>, R: Into<FnType>>(lhs: L, rhs: R) -> Self {
        Self {
            lhs: Box::new(lhs.into()),
            rhs: Box::new(rhs.into()),
        }
    }
}

impl DifferentiableFn for FnQuotient {
    fn f(&self, t: f32) -> f32 {
        self.lhs.f(t) / self.rhs.f(t)
    }
    fn df_dt(&self, t: f32) -> f32 {
        let lhs = self.lhs.as_ref();
        let rhs = self.rhs.as_ref();
        lhs.quotient(t, rhs)
    }
}

#[derive(Debug, Clone)]
pub struct FnSub {
    lhs: Box<FnType>,
    rhs: Box<FnType>,
}
impl std::fmt::Display for FnSub {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({} - {})", self.lhs, self.rhs)
    }
}

impl DifferentiableFn for FnSub {
    fn f(&self, t: f32) -> f32 {
        self.lhs.f(t) - self.rhs.f(t)
    }
    fn df_dt(&self, t: f32) -> f32 {
        self.lhs.df_dt(t) - self.rhs.df_dt(t)
    }
}

impl FnSub {
    pub fn new<L: Into<FnType>, R: Into<FnType>>(lhs: L, rhs: R) -> Self {
        Self {
            lhs: Box::new(lhs.into()),
            rhs: Box::new(rhs.into()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct FnSum {
    lhs: Box<FnType>,
    rhs: Box<FnType>,
}

impl std::fmt::Display for FnSum {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({} + {})", self.lhs, self.rhs)
    }
}

impl DifferentiableFn for FnSum {
    fn f(&self, t: f32) -> f32 {
        self.lhs.f(t) + self.rhs.f(t)
    }
    fn df_dt(&self, t: f32) -> f32 {
        self.lhs.df_dt(t) + self.rhs.df_dt(t)
    }
}

impl FnSum {
    pub fn new<L: Into<FnType>, R: Into<FnType>>(lhs: L, rhs: R) -> Self {
        Self {
            lhs: Box::new(lhs.into()),
            rhs: Box::new(rhs.into()),
        }
    }
}

/// A product of two [`FnType`]
#[derive(Debug, Clone)]
pub struct FnProduct {
    lhs: Box<FnType>,
    rhs: Box<FnType>,
}

impl std::fmt::Display for FnProduct {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({} * {})", self.lhs, self.rhs)
    }
}

impl FnProduct {
    pub fn new<L: Into<FnType>, R: Into<FnType>>(lhs: L, rhs: R) -> Self {
        Self {
            lhs: Box::new(lhs.into()),
            rhs: Box::new(rhs.into()),
        }
    }
}

impl DifferentiableFn for FnProduct {
    fn f(&self, t: f32) -> f32 {
        self.lhs.f(t) * self.rhs.f(t)
    }

    fn df_dt(&self, t: f32) -> f32 {
        let lhs = self.lhs.as_ref();
        let rhs = self.rhs.as_ref();
        rhs.product(t, lhs)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tanh;

impl std::fmt::Display for Tanh {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "tanh(%)")
    }
}

impl DifferentiableFn for Tanh {
    fn f(&self, t: f32) -> f32 {
        t.tanh()
    }

    fn df_dt(&self, t: f32) -> f32 {
        1. / t.cosh().powi(2)
    }
}

#[derive(Debug, Clone)]
pub struct StepRegion {
    pub start: f32,
    pub end: f32,
    pub k: f32,
    func: Box<FnType>,
}

impl std::fmt::Display for StepRegion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.func)
    }
}

impl StepRegion {
    pub fn new(start: f32, end: f32, k: f32) -> Self {
        // if k == 1, at start/end the value will < 1%, greater values will increase the drop off
        // rate, which may be useful for very small regions.
        let k = k * E / (end - start);
        let t_minus_start = FnSub::new(Constant(start), Linear); // (start - t)
        let k_x_t_minus_start = FnProduct::new(Constant(k), t_minus_start); // k * (start - t)
        let step_up = FnChain::new(k_x_t_minus_start, Tanh); // tanh(k(start - t))

        let t_minus_end = FnSub::new(Constant(end), Linear); // (end - t)
        let k_x_t_minus_end = FnProduct::new(Constant(k), t_minus_end); // k * (end - t)
        let step_down = FnChain::new(k_x_t_minus_end, Tanh); // tanh(k(end - t))

        let f = FnType::subtract(step_down, step_up) * 0.5; // 0.5 * (tanh(k(end - t)) - tanh(k(start - t)))

        Self {
            start,
            end,
            k,
            func: Box::new(f),
        }
    }
}

impl DifferentiableFn for StepRegion {
    fn f(&self, t: f32) -> f32 {
        self.func.f(t)
    }
    fn df_dt(&self, t: f32) -> f32 {
        self.func.df_dt(t)
    }
}

/// `f(t) = c`
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Constant(pub f32);

impl std::fmt::Display for Constant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl DifferentiableFn for Constant {
    fn f(&self, _: f32) -> f32 {
        self.0
    }

    fn df_dt(&self, _: f32) -> f32 {
        0.
    }
}

/// `f(t) = e^t`
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Exp;

impl std::fmt::Display for Exp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "e^%")
    }
}

impl DifferentiableFn for Exp {
    fn f(&self, t: f32) -> f32 {
        t.exp()
    }
    fn df_dt(&self, t: f32) -> f32 {
        t.exp()
    }
}

/// `f(t) = t`
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Linear;

impl std::fmt::Display for Linear {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "%")
    }
}

impl DifferentiableFn for Linear {
    fn f(&self, t: f32) -> f32 {
        t
    }
    fn df_dt(&self, _: f32) -> f32 {
        1.
    }
}

/// `f(t) = ln(t)` (natural logarithm)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Log;

impl std::fmt::Display for Log {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ln(%))")
    }
}

impl DifferentiableFn for Log {
    fn f(&self, t: f32) -> f32 {
        t.ln()
    }

    fn df_dt(&self, t: f32) -> f32 {
        1. / t
    }
}

/// Power `f(t) = t^k`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Power(pub f32);

impl std::fmt::Display for Power {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "%^{}", self.0)
    }
}

impl DifferentiableFn for Power {
    fn f(&self, t: f32) -> f32 {
        t.powf(self.0)
    }

    fn df_dt(&self, t: f32) -> f32 {
        self.0 * t.powf(self.0 - 1.)
    }
}

/// `f(t) = sin(t)`
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sin;

impl std::fmt::Display for Sin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "sin(%))")
    }
}

impl DifferentiableFn for Sin {
    fn f(&self, r: f32) -> f32 {
        r.sin()
    }

    fn df_dt(&self, t: f32) -> f32 {
        -t.cos()
    }
}

/// `f(t) = cos(t)`
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cos;

impl std::fmt::Display for Cos {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "cos(%))")
    }
}

impl DifferentiableFn for Cos {
    fn f(&self, r: f32) -> f32 {
        r.cos()
    }

    fn df_dt(&self, t: f32) -> f32 {
        t.sin()
    }
}

/// Create a [`FnChain`]
///
/// ```rust
/// use orienteering::chain;
/// use orienteering::calculus::*;
///
/// // sin(x^2)
/// let sin_x_squared = chain!(Power(2.) => Sin);
/// ```
#[macro_export]
macro_rules! chain {
    ($inner:expr => $outer:expr) => {
        FnType::chain($outer, $inner)
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::TAU;

    /* --- Basic Calculus --- */

    /// f  = sin(x) * x^2
    /// f' = 2x * sin(x) - x^2 * cos(x)
    #[test]
    fn test_fn_product() {
        let f1 = FnType::sin();
        let f2 = FnType::power(2.);

        let product = FnProduct::new(f1, f2);

        let f_expect = |x: f32| x.powi(2) * x.sin();
        let df_expect = |x: f32| 2. * x * x.sin() - x.powf(2.) * x.cos();

        assert_eq!(product.f(1.), f_expect(1.));
        assert_eq!(product.df_dt(1.), df_expect(1.));
    }

    /// f  = sin(x^2)
    /// f' = -2x * cos(x^2)
    #[test]
    fn test_fn_chain() {
        let chain = chain!(Power(2.) => Sin);

        let f_expect = |x: f32| x.powi(2).sin();
        let df_expect = |x: f32| -2. * x * x.powi(2).cos();

        assert_eq!(chain.f(1.), f_expect(1.));
        assert_eq!(chain.df_dt(1.), df_expect(1.));
    }

    /// f = x^2 / sin(x)
    /// f' = (2x * sin(x) - x^2 * cos(x)) / sin(x)^2
    #[test]
    fn test_fn_quotient() {
        let lhs = FnType::power(2.0);
        let rhs = FnType::sin();
        let quotient = lhs / rhs;

        let f_expect = |x: f32| x.powi(2) / x.sin();
        let df_expect = |x: f32| (2. * x * x.sin() + x.powf(2.) * x.cos()) / x.sin().powi(2);

        assert_eq!(
            quotient.f(1.),
            f_expect(1.),
            "Quotient Function Incorrect: q(t) = f(t) / g(t)"
        );
        assert_eq!(
            quotient.df_dt(1.),
            df_expect(1.),
            "Quotient Rule Derivative Failed"
        );
    }

    /* --- Primitive Functions --- */
    #[test]
    fn test_constant() {
        for i in 0..12 {
            let c = i as f32 / TAU;
            let constant = FnType::constant(c);
            assert_eq!(constant.f(1.0), c);
            assert_eq!(constant.df_dt(1.0), 0.0);
        }
    }
    #[test]
    fn test_exp() {
        let exp = FnType::exp();
        let f_expect = |x: f32| x.exp();
        let df_expect = |x: f32| x.exp();
        for i in 0..10 {
            let t = i as f32;
            assert_eq!(exp.f(t), f_expect(t));
            assert_eq!(exp.df_dt(t), df_expect(t));
        }
    }

    #[test]
    fn test_linear() {
        let linear = FnType::linear();
        let f_expect = |x: f32| x;
        let df_expect = |_: f32| 1.0;
        for i in 0..10 {
            let t = i as f32;
            assert_eq!(linear.f(t), f_expect(t));
            assert_eq!(linear.df_dt(t), df_expect(t));
        }
    }
    #[test]
    fn test_log() {
        let log = FnType::log();
        let f_expect = |x: f32| x.ln();
        let df_expect = |x: f32| 1. / x;
        for i in 1..10 {
            let t = i as f32;
            assert_eq!(log.f(t), f_expect(t));
            assert_eq!(log.df_dt(t), df_expect(t));
        }
    }
    #[test]
    fn test_power() {
        let power = FnType::power(2.0);
        let f_expect = |x: f32| x.powi(2);
        let df_expect = |x: f32| 2. * x;
        for i in 0..10 {
            let t = i as f32;
            assert_eq!(power.f(t), f_expect(t));
            assert_eq!(power.df_dt(t), df_expect(t));
        }
    }
    #[test]
    fn test_sin() {
        let sin = FnType::sin();
        let f_expect = |x: f32| x.sin();
        let df_expect = |x: f32| -x.cos();

        for i in 0..12 {
            let t = i as f32 / TAU;
            assert_eq!(sin.f(t), f_expect(t));
            assert_eq!(sin.df_dt(t), df_expect(t));
        }
    }

    #[test]
    fn test_step() {
        let step = FnType::tanh() * FnType::constant(0.5) + 0.5;

        approx::assert_relative_eq!(step.f(-1000.), 0.);
        approx::assert_relative_eq!(step.f(-100.), 0.);
        approx::assert_relative_eq!(step.f(-10.), 0.);
        approx::assert_relative_eq!(step.f(0.), 0.5);
        approx::assert_relative_eq!(step.f(10.), 1.);
        approx::assert_relative_eq!(step.f(100.), 1.);
        approx::assert_relative_eq!(step.f(1000.), 1.);
    }

    #[test]
    fn test_step_region() {
        const START: f32 = -10.;
        const END: f32 = 20.;
        const K: f32 = 40.;
        let step = FnType::step_region(START, END, K);
        println!("Step Region: {step}");

        const N: usize = 100;
        let ext = END - START;
        let t0 = START - ext / 2.;
        let t1 = END + ext / 2.;
        let dt = (t1 - t0) / N as f32;
        for i in 0..N {
            let t = t0 + dt * i as f32;
            let f = step.f(t);
            println!("f({t}) = {f}");
        }

        approx::assert_relative_eq!(0., step.f(START - 1.), epsilon = 1e-2);
        approx::assert_relative_eq!(1., step.f(START + 1.), epsilon = 1e-2);
        approx::assert_relative_eq!(1., step.f((END - START) / 2.), epsilon = 1e-2);
        approx::assert_relative_eq!(1., step.f(END - 1.), epsilon = 1e-2);
        approx::assert_relative_eq!(0., step.f(END + 1.), epsilon = 1e-2);
    }
}
