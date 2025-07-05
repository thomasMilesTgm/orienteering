//! # Gravitational Potential Energy (GPE)
//!
//! The topology of an orienteering world is defined by a procedurally generated scalar field
//! Representing the GPE at the terrain height at any xy point in space (which is proportional to
//! the elevation at that point).
//!
//! To produce locally distinct/natural-looking areas across an arbitrarily large world, the [`Gpe`]
//! up of a linear combination of [`GpeChunk`]s, which are localized using a pair of phase shifted
//! hyperbolic tan functions to ensure the summed field remains continuously differential, while
//! the effect of the field functions can be ignored once you're more than one chunk away.

use crate::calculus::primitives::prelude::*;
use derive_more::{Deref, DerefMut, From};
use nalgebra::Point2;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ChunkIndex(usize, usize);

pub struct Gpe {
    pub chunks: HashMap<ChunkIndex, GpeChunk>,
}

pub struct GpeChunk {
    pub mask: ChunkMask,
    pub generated: SumOfProductOfFnXY,
}

impl GpeChunk {
    pub fn f(&self, xy: Point2<f64>) -> f64 {
        self.mask.f(xy) * self.generated.f(xy)
    }
}

pub struct Aabb2D {
    pub min: Point2<f64>,
    pub max: Point2<f64>,
}

pub struct SumOfProductOfFnXY {
    pub sum: Vec<ProductOfFnXY>,
}

impl SumOfProductOfFnXY {
    pub fn new(sum: Vec<ProductOfFnXY>) -> Self {
        SumOfProductOfFnXY { sum }
    }
    pub fn f(&self, xy: Point2<f64>) -> f64 {
        self.sum.iter().map(|p| p.f(xy)).sum()
    }
}

/// f(x, y) = f_x(x) * f_y(y)
pub struct ProductOfFnXY {
    pub fx: FunctionX,
    pub fy: FunctionY,
}

#[derive(Debug, Clone, Deref, DerefMut, From)]
pub struct FunctionX(pub FunctionT);

#[derive(Debug, Clone, Deref, DerefMut, From)]
pub struct FunctionY(pub FunctionT);

impl ProductOfFnXY {
    pub fn new(fx: FunctionT, fy: FunctionT) -> Self {
        ProductOfFnXY {
            fx: fx.into(),
            fy: fy.into(),
        }
    }

    pub fn f(&self, xy: Point2<f64>) -> f64 {
        self.fx.f(xy.x) * self.fy.f(xy.y)
    }
}

impl Aabb2D {
    pub fn new(min: Point2<f64>, max: Point2<f64>) -> Self {
        Aabb2D { min, max }
    }
    pub fn width(&self) -> f64 {
        self.max.x - self.min.x
    }
    pub fn height(&self) -> f64 {
        self.max.y - self.min.y
    }
}

pub struct ChunkMask {
    pub f: ProductOfFnXY,
    pub aabb: Aabb2D,
    pub weight: f64,
    pub slope: f64,
}

impl ChunkMask {
    pub fn new(weight: f64, slope: f64, aabb: Aabb2D) -> Self {
        let (x0, x1) = (aabb.min.x, aabb.max.x);
        let (y0, y1) = (aabb.min.y, aabb.max.y);

        let fx = constant(weight / 2.) * (constant(slope) * (linear() - constant(x0))).tanh()
            - constant(weight / 2.) * (constant(slope) * (linear() - constant(x1))).tanh();

        let fy = constant(weight / 2.) * (constant(slope) * (linear() - constant(y0))).tanh()
            - constant(weight / 2.) * (constant(slope) * (linear() - constant(y1))).tanh();

        let f = ProductOfFnXY::new(fx, fy);

        ChunkMask {
            f,
            aabb,
            weight,
            slope,
        }
    }

    pub fn f(&self, xy: Point2<f64>) -> f64 {
        self.f.f(xy)
    }
}

#[cfg(test)]
mod test {
    use std::io::Cursor;

    use image::{ImageFormat, RgbImage};

    use super::*;

    #[test]
    fn mask() {
        let aabb = Aabb2D::new(Point2::new(-250., -250.), Point2::new(250., 250.));
        let mask = ChunkMask::new(1., 0.05, aabb);

        const RES: u32 = 1000;

        let mut img = RgbImage::new(RES, RES);

        for i in 0..RES {
            let x = i as f64 - RES as f64 / 2.;
            for j in 0..RES {
                let y = j as f64 - RES as f64 / 2.;
                let value = mask.f(Point2::new(x, y));
                let pix = img.get_pixel_mut(i, j);
                let color = (value * 255.) as u8;
                pix.0 = [color; 3];
            }
        }

        let mut buf = vec![];
        img.write_to(&mut Cursor::new(&mut buf), ImageFormat::Png)
            .unwrap();

        // std::fs::write("mask.png", buf).unwrap();
    }

    #[test]
    fn masked_chunk() {
        let aabb = Aabb2D::new(Point2::new(-500., -500.), Point2::new(500., 500.));
        let mask = ChunkMask::new(1., 0.01, aabb);

        let f0 = ProductOfFnXY::new(
            constant(-4.0) * ((linear() - constant(121.)) * constant(0.01)).sin(),
            constant(-1.0) * ((linear() - constant(91.)) * constant(0.008)).cos().pow(2.),
        );

        let f1 = ProductOfFnXY::new(
            constant(0.9) * ((linear() - constant(29.)) * constant(0.006)).cosh(),
            constant(0.8) * ((linear() + constant(19.)) * constant(0.005)).sin(),
        );

        let f2 = ProductOfFnXY::new(
            constant(-2.4) * ((linear() - constant(12.)) * constant(0.05)),
            linear() * constant(0.000002),
        );

        let f3 = ProductOfFnXY::new(
            constant(14.0) * ((linear() - constant(96.)) * constant(0.0015)).sin(),
            constant(1.0)
                * ((linear() - constant(75.)) * constant(0.0085))
                    .cos()
                    .pow(2.),
        );

        let generated = SumOfProductOfFnXY::new(vec![f0, f1, f2, f3]);

        let chunk = GpeChunk { mask, generated };

        const RES: u32 = 2000;

        let mut img = RgbImage::new(RES, RES);

        for i in 0..RES {
            let x = i as f64 - RES as f64 / 2.;
            for j in 0..RES {
                let y = j as f64 - RES as f64 / 2.;
                let xy = Point2::new(x, y);
                let value = chunk.f(xy);
                let pix = img.get_pixel_mut(i, j);
                let color = (value.abs() * 10.).round() as u8;

                if color % 7 != 0 || chunk.mask.f(xy) < 0.05 {
                    continue;
                }
                if color == 0 {
                    pix.0 = [140, 134, 119];
                } else if value > 0. {
                    pix.0 = [color, color.saturating_mul(4).saturating_add(50), color];
                } else {
                    pix.0 = [0, 0, color + (100 - color)];
                }
            }
        }

        let mut buf = vec![];
        img.write_to(&mut Cursor::new(&mut buf), ImageFormat::Png)
            .unwrap();

        std::fs::write("mask.png", buf).unwrap();
    }
}
