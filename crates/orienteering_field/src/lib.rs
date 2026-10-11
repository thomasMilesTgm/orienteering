//! Composable height fields.
//!
//! A [`Field`] is a serializable expression tree of terrain primitives (bumps, ridges, scarps,
//! waves, planes) and combinators (sums, products, scaling, domain warping, and smooth rectangular
//! [`Window`]s). Fields are evaluated generically over [`Scalar`]: with `f64` for heights, or with
//! [`Dual`] numbers for exact gradients (see [`Surface`]).

use nalgebra::{Point2, Vector2};
use orienteering_math::{Aabb, Dual, Scalar};
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

/// The largest error, in meters, accepted from truncating a [`Window`]'s tails to zero.
pub const WINDOW_TOLERANCE: f64 = 1e-3;

/// A [`Window`]'s support never extends more than this many blend widths past its bounds.
pub const WINDOW_CUTOFF: f64 = 6.;

/// How many widths either side of a [`Field::Scarp`] its offset fades over.
pub const SCARP_REACH: f64 = 4.;

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

    /// The region outside which `window × f` is below [`WINDOW_TOLERANCE`] for any `|f| ≤ bound`,
    /// capped at [`WINDOW_CUTOFF`] blend widths.
    ///
    /// A distance `d` past an edge, each axis factor is `½(1 - tanh(d/w)) < e^(-2d/w)` (and the
    /// other factor is at most 1), so the tail is negligible once `d ≥ ½ w ln(bound / tolerance)`.
    pub fn support(&self, bound: f64) -> Aabb {
        let blends = (0.5 * (bound / WINDOW_TOLERANCE).ln()).clamp(0., WINDOW_CUTOFF);
        self.bounds.grow(blends * self.blend)
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

    /// A cliff band or escarpment: a smooth step of total `height` across the line through
    /// `origin` perpendicular to `normal`, rising towards `normal` over about `width` meters. The
    /// step's offset fades away from the cliff (over [`SCARP_REACH`] widths), so it stays a local
    /// cliff rather than shifting everything on either side.
    Scarp {
        origin: Point2<f64>,
        normal: Vector2<f64>,
        width: f64,
        height: f64,
    },

    /// A mountain range (or, if `height` < 0, a trough): a Gaussian ridge of the given `width`
    /// along the segment of length `2 · half_length` through `center` in `direction`, tapering
    /// off past each end over about `width`.
    Range {
        center: Point2<f64>,
        direction: Vector2<f64>,
        half_length: f64,
        width: f64,
        height: f64,
    },

    /// `field`, skipped entirely (as exactly 0) outside `support`. The caller guarantees the field
    /// is negligible there.
    Bounded {
        support: Aabb,
        field: Box<Field>,
    },

    /// `field` multiplied by `window`, and skipped entirely outside `support`, where the product
    /// is negligible. Build with [`Field::masked`], which computes the support.
    Masked {
        window: Window,
        support: Aabb,
        field: Box<Field>,
    },

    /// `field` evaluated at displaced coordinates `p + (dx(p), dy(p))` (domain warping), which
    /// bends its shapes into irregular, meandering ones.
    Warped {
        dx: Box<Field>,
        dy: Box<Field>,
        field: Box<Field>,
    },

    Scaled {
        factor: f64,
        field: Box<Field>,
    },
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

    pub fn scarp(origin: Point2<f64>, normal: Vector2<f64>, width: f64, height: f64) -> Self {
        Self::Scarp {
            origin,
            normal: normal.normalize(),
            width,
            height,
        }
    }

    /// A [`Field::Range`] along the segment from `a` to `b`.
    pub fn range(a: Point2<f64>, b: Point2<f64>, width: f64, height: f64) -> Self {
        let along = b - a;
        Self::Range {
            center: nalgebra::center(&a, &b),
            direction: along.normalize(),
            half_length: along.norm() / 2.,
            width,
            height,
        }
    }

    /// The region outside which a [`Field::Range`] from `a` to `b` is below [`WINDOW_TOLERANCE`]:
    /// the segment's bounding box grown by enough widths for both the Gaussian flanks and the
    /// tanh tapers at the ends to decay.
    pub fn range_support(a: Point2<f64>, b: Point2<f64>, width: f64, height: f64) -> Aabb {
        let decay = (height.abs() / WINDOW_TOLERANCE).ln().max(0.);
        let widths = (2. * decay).sqrt().max(decay / 2.);
        Aabb::new(
            Point2::new(a.x.min(b.x), a.y.min(b.y)),
            Point2::new(a.x.max(b.x), a.y.max(b.y)),
        )
        .grow(widths * width)
    }

    pub fn bounded(self, support: Aabb) -> Self {
        Self::Bounded {
            support,
            field: Box::new(self),
        }
    }

    pub fn masked(self, window: Window) -> Self {
        Self::Masked {
            window,
            support: window.support(self.bound()),
            field: Box::new(self),
        }
    }

    /// An upper bound on `|h|` anywhere (possibly infinite).
    pub fn bound(&self) -> f64 {
        match self {
            Self::Constant(c) => c.abs(),
            Self::Plane { slope, .. } if *slope == Vector2::zeros() => 0.,
            Self::Plane { .. } => f64::INFINITY,
            Self::Bump { height, .. } | Self::Ridge { height, .. } | Self::Range { height, .. } => {
                height.abs()
            }
            Self::Wave { amplitude, .. } => amplitude.abs(),
            Self::Scarp { height, .. } => height.abs() / 2.,
            Self::Masked { field, .. }
            | Self::Warped { field, .. }
            | Self::Bounded { field, .. } => field.bound(),
            Self::Scaled { factor, .. } if *factor == 0. => 0.,
            Self::Scaled { factor, field } => factor.abs() * field.bound(),
            Self::Sum(fields) => fields.iter().map(Self::bound).sum(),
            Self::Product(fields) => {
                let bounds: Vec<f64> = fields.iter().map(Self::bound).collect();
                if bounds.contains(&0.) {
                    0.
                } else {
                    bounds.iter().product()
                }
            }
        }
    }

    pub fn warped(self, dx: Field, dy: Field) -> Self {
        Self::Warped {
            dx: Box::new(dx),
            dy: Box::new(dy),
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
            Self::Scarp {
                origin,
                normal,
                width,
                height,
            } => {
                let d = x.offset(-origin.x).scale(normal.x) + y.offset(-origin.y).scale(normal.y);
                let reach = SCARP_REACH * width;
                let fade = (d * d).scale(-0.5 / (reach * reach)).exp();
                d.scale(1. / width).tanh() * fade.scale(height / 2.)
            }
            Self::Range {
                center,
                direction,
                half_length,
                width,
                height,
            } => {
                let (rx, ry) = (x.offset(-center.x), y.offset(-center.y));
                let along = rx.scale(direction.x) + ry.scale(direction.y);
                let across = rx.scale(-direction.y) + ry.scale(direction.x);
                let k = 1. / width;
                let taper = (along.offset(*half_length).scale(k).tanh()
                    - along.offset(-half_length).scale(k).tanh())
                .scale(0.5);
                (across * across).scale(-0.5 * k * k).exp() * taper.scale(*height)
            }
            Self::Bounded { support, field } => {
                if !support.contains(Point2::new(x.value(), y.value())) {
                    return S::constant(0.);
                }
                field.eval(x, y)
            }
            Self::Masked {
                window,
                support,
                field,
            } => {
                if !support.contains(Point2::new(x.value(), y.value())) {
                    return S::constant(0.);
                }
                window.eval(x, y) * field.eval(x, y)
            }
            Self::Warped { dx, dy, field } => field.eval(x + dx.eval(x, y), y + dy.eval(x, y)),
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
pub fn numeric_gradient<S: Surface>(s: &S, p: Point2<f64>, h: f64) -> Vector2<f64> {
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
            + Field::plane(Point2::origin(), Vector2::new(0.02, -0.01)).masked(w) * 3.
            + Field::scarp(Point2::new(-20., 40.), Vector2::new(-2., 1.), 12., 9.)
            + Field::range(Point2::new(-150., -60.), Point2::new(120., 90.), 35., 14.)
                .bounded(Aabb::centered(Point2::origin(), 1000.))
            + Field::bump(Point2::new(-30., 10.), 50., 8.).warped(
                Field::wave(Vector2::new(1., 2.), 120., 0.3, 15.),
                Field::wave(Vector2::new(-2., 1.), 150., 1.9, 15.),
            );

        for &(x, y) in &[
            (0., 0.),
            (35., -12.),
            (-190., 260.),
            (240., -140.),
            (90., 90.),
        ] {
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

        for &(x, y) in &[
            (0., 0.),
            (150., 1.),
            (-299., 37.),
            (12., -150.5),
            (-75., 75.),
        ] {
            let sum: f64 = tiles.iter().map(|w| w.eval(x, y)).sum();
            assert_relative_eq!(sum, whole.eval(x, y), epsilon = 1e-12);
        }
    }

    #[test]
    fn truncated_tails_are_within_tolerance() {
        let w = window((0., 0.), (100., 100.), 10.);
        for height in [0.01, 1., 30., 500.] {
            let f = Field::bump(Point2::new(50., 50.), 1e6, height).masked(w);
            let Field::Masked { support, .. } = &f else {
                unreachable!()
            };
            // Just inside the support, the untruncated value is what gets dropped beyond it.
            let edge = Point2::new(support.max.x - 1e-9, 50.);
            assert!(support.contains(edge));
            // Large bounds hit the cutoff cap, which allows a proportionally larger error.
            let allowed = WINDOW_TOLERANCE.max(height * (-2. * WINDOW_CUTOFF).exp());
            assert!(
                f.height(edge).abs() <= allowed,
                "{height}: {}",
                f.height(edge)
            );
            assert_eq!(f.height(edge + Vector2::new(1e-6, 0.)), 0.);
        }
    }

    #[test]
    fn ranges_are_bounded_by_their_support() {
        let (a, b) = (Point2::new(-300., 100.), Point2::new(400., -200.));
        let range = Field::range(a, b, 50., 250.);
        let support = Field::range_support(a, b, 50., 250.);
        // Full height in the middle of the segment, negligible just outside the support.
        assert_relative_eq!(range.height(nalgebra::center(&a, &b)), 250., epsilon = 1e-3);
        for p in [
            Point2::new(support.min.x, support.min.y),
            Point2::new(support.max.x, (support.min.y + support.max.y) / 2.),
            Point2::new((support.min.x + support.max.x) / 2., support.max.y),
        ] {
            assert!(
                range.height(p).abs() < WINDOW_TOLERANCE,
                "{p}: {}",
                range.height(p)
            );
        }
        let bounded = range.bounded(support);
        assert_eq!(bounded.height(support.max + Vector2::new(1., 1.)), 0.);
    }

    #[test]
    fn bound_is_an_upper_bound() {
        let w = window((-100., -100.), (100., 100.), 20.);
        let f = (Field::bump(Point2::origin(), 30., 12.)
            + Field::wave(Vector2::new(1., 1.), 90., 0.3, 4.)
            + Field::ridge(Point2::new(10., 0.), Vector2::new(0., 1.), 15., -7.))
        .masked(w)
            * Field::Constant(-2.);
        let bound = f.bound();
        assert_relative_eq!(bound, 46.);
        for i in -20..=20 {
            for j in -20..=20 {
                let p = Point2::new(i as f64 * 7., j as f64 * 7.);
                assert!(f.height(p).abs() <= bound);
            }
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
