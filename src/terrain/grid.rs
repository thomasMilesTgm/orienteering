//! Sampling a surface on a regular grid.

use super::field::{Aabb, Surface};
use nalgebra::{Point2, Vector2};
use rayon::prelude::*;

/// Heights, and optionally exact gradients, sampled on a regular grid.
#[derive(Debug, Clone)]
pub struct HeightGrid {
    pub origin: Point2<f64>,
    pub spacing: f64,
    pub nx: usize,
    pub ny: usize,
    values: Vec<f64>,
    gradients: Option<Vec<Vector2<f64>>>,
}

impl HeightGrid {
    /// Samples heights of `surface` over `area`, one row per task in parallel.
    #[tracing::instrument(level = "info", skip(surface), fields(nx, ny))]
    pub fn sample<S: Surface + Sync + ?Sized>(surface: &S, area: Aabb, spacing: f64) -> Self {
        let mut grid = Self::empty(area, spacing);
        let (nx, origin) = (grid.nx, grid.origin);
        grid.values
            .par_chunks_mut(nx)
            .enumerate()
            .for_each(|(j, row)| {
                for (i, h) in row.iter_mut().enumerate() {
                    *h = surface.height(point(origin, spacing, i, j));
                }
            });
        grid
    }

    /// Like [`HeightGrid::sample`], but also stores the exact gradient at every sample (about 20%
    /// slower).
    #[tracing::instrument(level = "info", skip(surface), fields(nx, ny))]
    pub fn sample_with_gradients<S: Surface + Sync + ?Sized>(
        surface: &S,
        area: Aabb,
        spacing: f64,
    ) -> Self {
        let mut grid = Self::empty(area, spacing);
        let (nx, origin) = (grid.nx, grid.origin);
        let mut gradients = vec![Vector2::zeros(); grid.values.len()];
        grid.values
            .par_chunks_mut(nx)
            .zip(gradients.par_chunks_mut(nx))
            .enumerate()
            .for_each(|(j, (heights, grads))| {
                for (i, (h, g)) in heights.iter_mut().zip(grads).enumerate() {
                    (*h, *g) = surface.height_and_gradient(point(origin, spacing, i, j));
                }
            });
        grid.gradients = Some(gradients);
        grid
    }

    fn empty(area: Aabb, spacing: f64) -> Self {
        let nx = (area.width() / spacing).ceil() as usize + 1;
        let ny = (area.height() / spacing).ceil() as usize + 1;
        tracing::Span::current().record("nx", nx).record("ny", ny);
        Self {
            origin: area.min,
            spacing,
            nx,
            ny,
            values: vec![0.; nx * ny],
            gradients: None,
        }
    }

    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.values[j * self.nx + i]
    }

    /// The gradient at sample `(i, j)`: exact if the grid was sampled with gradients, otherwise
    /// estimated with finite differences (one-sided at the edges).
    pub fn gradient(&self, i: usize, j: usize) -> Vector2<f64> {
        if let Some(gradients) = &self.gradients {
            return gradients[j * self.nx + i];
        }
        let (i0, i1) = (i.saturating_sub(1), (i + 1).min(self.nx - 1));
        let (j0, j1) = (j.saturating_sub(1), (j + 1).min(self.ny - 1));
        Vector2::new(
            (self.get(i1, j) - self.get(i0, j)) / ((i1 - i0) as f64 * self.spacing),
            (self.get(i, j1) - self.get(i, j0)) / ((j1 - j0) as f64 * self.spacing),
        )
    }

    pub fn has_gradients(&self) -> bool {
        self.gradients.is_some()
    }

    pub fn point(&self, i: usize, j: usize) -> Point2<f64> {
        point(self.origin, self.spacing, i, j)
    }

    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// The (min, max) sampled height.
    pub fn range(&self) -> (f64, f64) {
        self.values
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| {
                (lo.min(v), hi.max(v))
            })
    }
}

fn point(origin: Point2<f64>, spacing: f64, i: usize, j: usize) -> Point2<f64> {
    origin + Vector2::new(i as f64, j as f64) * spacing
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::terrain::Field;
    use approx::assert_relative_eq;

    #[test]
    fn exact_and_estimated_gradients_agree() {
        let hill = Field::bump(Point2::new(10., 20.), 60., 40.);
        let area = Aabb::centered(Point2::origin(), 200.);
        let exact = HeightGrid::sample_with_gradients(&hill, area, 0.5);
        let estimated = HeightGrid::sample(&hill, area, 0.5);
        assert_eq!(exact.values(), estimated.values());

        for (i, j) in [(1, 1), (100, 150), (250, 37), (399, 399)] {
            assert_relative_eq!(exact.gradient(i, j), hill.gradient(exact.point(i, j)));
            assert_relative_eq!(
                exact.gradient(i, j),
                estimated.gradient(i, j),
                epsilon = 2e-3
            );
        }
    }
}
