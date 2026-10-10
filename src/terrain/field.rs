//! Composable height fields.

use super::scalar::{Dual, Scalar};
use nalgebra::{Point2, Vector2};
use std::f64::consts::TAU;

/// Anything with a height at every point.
pub trait Surface {
    /// Height in meters at `p`.
    fn height(&self, p: Point2<f64>) -> f64;

    /// Height at `p` together with its gradient `∇h = [∂h/∂x, ∂h/∂y]`.
    fn height_and_gradient(&self, p: Point2<f64>) -> (f64, Vector2<f64>);

    fn gradient(&self, p: Point2<f64>) -> Vector2<f64> {
        self.height_and_gradient(p).1
    }
}

/// An axis-aligned rectangle.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Aabb {
    pub min: Point2<f64>,
    pub max: Point2<f64>,
}

impl Aabb {
    pub fn new(min: Point2<f64>, max: Point2<f64>) -> Self {
        Self { min, max }
    }

    pub fn centered(center: Point2<f64>, size: f64) -> Self {
        let half = Vector2::repeat(size / 2.);
        Self::new(center - half, center + half)
    }

    pub fn width(&self) -> f64 {
        self.max.x - self.min.x
    }

    pub fn height(&self) -> f64 {
        self.max.y - self.min.y
    }

    pub fn center(&self) -> Point2<f64> {
        nalgebra::center(&self.min, &self.max)
    }

    pub fn contains(&self, p: Point2<f64>) -> bool {
        (self.min.x..=self.max.x).contains(&p.x) && (self.min.y..=self.max.y).contains(&p.y)
    }

    /// Grows the box by `margin` on every side.
    pub fn grow(&self, margin: f64) -> Self {
        let m = Vector2::repeat(margin);
        Self::new(self.min - m, self.max + m)
    }

    /// The four quadrants, sharing edges exactly so their [`Window`]s partition this one.
    pub fn quadrants(&self) -> [Self; 4] {
        let (min, c, max) = (self.min, self.center(), self.max);
        [
            Self::new(min, c),
            Self::new(Point2::new(c.x, min.y), Point2::new(max.x, c.y)),
            Self::new(Point2::new(min.x, c.y), Point2::new(c.x, max.y)),
            Self::new(c, max),
        ]
    }

    /// A point at fractional position `(u, v)` ∈ [0, 1]² within the box.
    pub fn lerp(&self, u: f64, v: f64) -> Point2<f64> {
        Point2::new(
            self.min.x + u * self.width(),
            self.min.y + v * self.height(),
        )
    }
}

/// Beyond this many blend widths outside its bounds a [`Window`] is treated as exactly zero
/// (its true value there is ~1e-5 or smaller).
pub const WINDOW_CUTOFF: f64 = 6.;

/// A smooth rectangular mask: ≈1 inside `bounds`, ≈0 outside, with tanh transitions of width
/// `blend` centered on the edges.
///
/// Along one axis the window over `[a, b]` is `½(tanh((t - a)/w) - tanh((t - b)/w))`. Windows of
/// intervals that share edges telescope when summed, so any tiling of the plane by rectangles with
/// a common `blend` sums to exactly 1 everywhere. Fields blended with such windows never dip or
/// bulge at the seams.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Window {
    pub bounds: Aabb,
    pub blend: f64,
}

impl Window {
    pub fn new(bounds: Aabb, blend: f64) -> Self {
        Self { bounds, blend }
    }

    /// The region outside which the window is treated as zero.
    pub fn support(&self) -> Aabb {
        self.bounds.grow(WINDOW_CUTOFF * self.blend)
    }

    pub fn eval<S: Scalar>(&self, x: S, y: S) -> S {
        let Aabb { min, max } = self.bounds;
        self.axis(x, min.x, max.x) * self.axis(y, min.y, max.y)
    }

    fn axis<S: Scalar>(&self, t: S, a: f64, b: f64) -> S {
        let k = 1. / self.blend;
        (t.offset(-a).scale(k).tanh() - t.offset(-b).scale(k).tanh()).scale(0.5)
    }
}

/// A height field built from terrain primitives and combinators.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Field {
    Constant(f64),

    /// An inclined plane through `origin` (where it has height 0).
    Plane {
        origin: Point2<f64>,
        slope: Vector2<f64>,
    },

    /// A Gaussian hill (or hollow, if `height` < 0).
    Bump {
        center: Point2<f64>,
        radius: f64,
        height: f64,
    },

    /// A Gaussian ridge (or valley) along the infinite line through `origin` in `direction`.
    Ridge {
        origin: Point2<f64>,
        direction: Vector2<f64>,
        width: f64,
        height: f64,
    },

    /// A sinusoidal plane wave travelling along `direction`.
    Wave {
        direction: Vector2<f64>,
        wavelength: f64,
        phase: f64,
        amplitude: f64,
    },

    /// `field` multiplied by `window`, and skipped entirely outside the window's support.
    Masked { window: Window, field: Box<Field> },

    Scaled { factor: f64, field: Box<Field> },
    Sum(Vec<Field>),
    Product(Vec<Field>),
}

impl Field {
    pub fn plane(origin: Point2<f64>, slope: Vector2<f64>) -> Self {
        Self::Plane { origin, slope }
    }

    pub fn bump(center: Point2<f64>, radius: f64, height: f64) -> Self {
        Self::Bump {
            center,
            radius,
            height,
        }
    }

    pub fn ridge(origin: Point2<f64>, direction: Vector2<f64>, width: f64, height: f64) -> Self {
        Self::Ridge {
            origin,
            direction: direction.normalize(),
            width,
            height,
        }
    }

    pub fn wave(direction: Vector2<f64>, wavelength: f64, phase: f64, amplitude: f64) -> Self {
        Self::Wave {
            direction: direction.normalize(),
            wavelength,
            phase,
            amplitude,
        }
    }

    pub fn masked(self, window: Window) -> Self {
        Self::Masked {
            window,
            field: Box::new(self),
        }
    }

    pub fn scaled(self, factor: f64) -> Self {
        Self::Scaled {
            factor,
            field: Box::new(self),
        }
    }

    /// Evaluates the field at `(x, y)` with any [`Scalar`] type.
    pub fn eval<S: Scalar>(&self, x: S, y: S) -> S {
        match self {
            Self::Constant(c) => S::constant(*c),
            Self::Plane { origin, slope } => {
                x.offset(-origin.x).scale(slope.x) + y.offset(-origin.y).scale(slope.y)
            }
            Self::Bump {
                center,
                radius,
                height,
            } => {
                let (dx, dy) = (x.offset(-center.x), y.offset(-center.y));
                let r2 = dx * dx + dy * dy;
                r2.scale(-0.5 / (radius * radius)).exp().scale(*height)
            }
            Self::Ridge {
                origin,
                direction,
                width,
                height,
            } => {
                // Signed distance from the ridge line, along its normal.
                let d = x.offset(-origin.x).scale(-direction.y)
                    + y.offset(-origin.y).scale(direction.x);
                (d * d).scale(-0.5 / (width * width)).exp().scale(*height)
            }
            Self::Wave {
                direction,
                wavelength,
                phase,
                amplitude,
            } => {
                let along = x.scale(direction.x) + y.scale(direction.y);
                along
                    .scale(TAU / wavelength)
                    .offset(*phase)
                    .sin()
                    .scale(*amplitude)
            }
            Self::Masked { window, field } => {
                if !window.support().contains(Point2::new(x.value(), y.value())) {
                    return S::constant(0.);
                }
                window.eval(x, y) * field.eval(x, y)
            }
            Self::Scaled { factor, field } => field.eval(x, y).scale(*factor),
            Self::Sum(fields) => fields
                .iter()
                .fold(S::constant(0.), |acc, f| acc + f.eval(x, y)),
            Self::Product(fields) => fields
                .iter()
                .fold(S::constant(1.), |acc, f| acc * f.eval(x, y)),
        }
    }
}

impl Surface for Field {
    fn height(&self, p: Point2<f64>) -> f64 {
        self.eval(p.x, p.y)
    }

    fn height_and_gradient(&self, p: Point2<f64>) -> (f64, Vector2<f64>) {
        let (x, y) = Dual::variables(p.x, p.y);
        let h = self.eval(x, y);
        (h.v, h.d)
    }
}

impl std::ops::Add for Field {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        match self {
            Self::Sum(mut fields) => {
                fields.push(rhs);
                Self::Sum(fields)
            }
            lhs => Self::Sum(vec![lhs, rhs]),
        }
    }
}

impl std::ops::Mul for Field {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        match self {
            Self::Product(mut fields) => {
                fields.push(rhs);
                Self::Product(fields)
            }
            lhs => Self::Product(vec![lhs, rhs]),
        }
    }
}

impl std::ops::Mul<f64> for Field {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self {
        self.scaled(rhs)
    }
}

/// Central finite-difference gradient, for testing analytic gradients against.
#[cfg(test)]
pub(crate) fn numeric_gradient<S: Surface>(s: &S, p: Point2<f64>, h: f64) -> Vector2<f64> {
    let dx = Vector2::new(h, 0.);
    let dy = Vector2::new(0., h);
    Vector2::new(
        (s.height(p + dx) - s.height(p - dx)) / (2. * h),
        (s.height(p + dy) - s.height(p - dy)) / (2. * h),
    )
}

#[cfg(test)]
mod test {
    use super::*;
    use approx::assert_relative_eq;

    fn window(min: (f64, f64), max: (f64, f64), blend: f64) -> Window {
        Window::new(
            Aabb::new(Point2::new(min.0, min.1), Point2::new(max.0, max.1)),
            blend,
        )
    }

    #[test]
    fn gradient_matches_finite_difference() {
        let w = window((-200., -150.), (250., 300.), 40.);
        let field = (Field::bump(Point2::new(30., -20.), 80., 25.)
            + Field::ridge(Point2::new(0., 50.), Vector2::new(1., 2.), 30., -12.)
            + Field::wave(Vector2::new(3., -1.), 170., 0.4, 6.))
            * Field::wave(Vector2::new(-1., 1.), 400., 1.1, 1.)
            + Field::plane(Point2::origin(), Vector2::new(0.02, -0.01)).masked(w) * 3.;

        for &(x, y) in &[(0., 0.), (35., -12.), (-190., 260.), (240., -140.), (90., 90.)] {
            let p = Point2::new(x, y);
            let (h, g) = field.height_and_gradient(p);
            assert_relative_eq!(h, field.height(p), epsilon = 1e-12);
            let n = numeric_gradient(&field, p, 1e-3);
            assert_relative_eq!(g, n, epsilon = 1e-6);
        }
    }

    #[test]
    fn windows_partition_unity() {
        let blend = 25.;
        let tiles: Vec<Window> = Aabb::new(Point2::new(-300., -300.), Point2::new(300., 300.))
            .quadrants()
            .into_iter()
            .flat_map(|q| q.quadrants())
            .map(|q| Window::new(q, blend))
            .collect();
        let whole = window((-300., -300.), (300., 300.), blend);

        for &(x, y) in &[(0., 0.), (150., 1.), (-299., 37.), (12., -150.5), (-75., 75.)] {
            let sum: f64 = tiles.iter().map(|w| w.eval(x, y)).sum();
            assert_relative_eq!(sum, whole.eval(x, y), epsilon = 1e-12);
        }
    }

    #[test]
    fn masked_field_vanishes_outside_support() {
        let w = window((0., 0.), (100., 100.), 10.);
        let f = Field::Constant(50.).masked(w);
        assert_relative_eq!(f.height(Point2::new(50., 50.)), 50., epsilon = 0.01);
        assert_eq!(f.height(Point2::new(200., 50.)), 0.);
    }
}
