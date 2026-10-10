use nalgebra::{Point2, Vector2};

/// An axis-aligned rectangle.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Aabb {
    pub min: Point2<f64>,
    pub max: Point2<f64>,
}

impl Aabb {
    #[inline]
    pub fn new(min: Point2<f64>, max: Point2<f64>) -> Self {
        Self { min, max }
    }

    #[inline]

    pub fn centered(center: Point2<f64>, size: f64) -> Self {
        let half = Vector2::repeat(size / 2.);
        Self::new(center - half, center + half)
    }

    #[inline]

    pub fn width(&self) -> f64 {
        self.max.x - self.min.x
    }

    #[inline]

    pub fn height(&self) -> f64 {
        self.max.y - self.min.y
    }

    #[inline]

    pub fn center(&self) -> Point2<f64> {
        nalgebra::center(&self.min, &self.max)
    }

    #[inline]

    pub fn contains(&self, p: Point2<f64>) -> bool {
        (self.min.x..=self.max.x).contains(&p.x) && (self.min.y..=self.max.y).contains(&p.y)
    }

    /// Grows the box by `margin` on every side.
    #[inline]
    pub fn grow(&self, margin: f64) -> Self {
        let m = Vector2::repeat(margin);
        Self::new(self.min - m, self.max + m)
    }

    /// The four quadrants, sharing edges exactly so windows on them partition this one.
    #[inline]
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
    #[inline]
    pub fn lerp(&self, u: f64, v: f64) -> Point2<f64> {
        Point2::new(
            self.min.x + u * self.width(),
            self.min.y + v * self.height(),
        )
    }
}
