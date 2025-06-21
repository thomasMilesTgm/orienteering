//! The world

use super::scalar_field::*;
use nalgebra::{Point2, Vector2};

pub struct World {
    pub chunks: Vec<Vec<Chunk>>,
}

pub struct Chunk {
    localized: LocalizedPotential,
    inner_fields: Vec<Potential>,
}

impl ScalarField for Chunk {
    fn phi(&self, xy: Point2<f32>) -> f32 {
        let inner: f32 = self.inner_fields.iter().map(|f| f.phi(xy)).sum();
        inner * self.localized.phi(xy)
    }

    fn gradient_at(&self, xy: Point2<f32>) -> Vector2<f32> {
        let inner: Vector2<_> = self.inner_fields.iter().map(|f| f.gradient_at(xy)).sum();
        inner * self.localized.phi(xy)
    }
}

impl Chunk {
    pub fn new(size: f32, center: Point2<f32>) -> Self {
        const RAMP: f32 = 40.;
        let min = center - Vector2::new(size / 2.0, size / 2.0);
        let max = center + Vector2::new(size / 2.0, size / 2.0);
        let localized = LocalizedPotential::new(min, max, RAMP);

        // TODO: Randomly generate inner fields..
        let inner_fields = vec![
            CircularPotential::default().into(),
            OscillatingPotential::default().into(),
        ];

        Self {
            localized,
            inner_fields,
        }
    }
}
