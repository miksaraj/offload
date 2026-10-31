//! Constant-velocity Kalman filter over `(cx, cy, a, h)` box measurements —
//! a port of the filter used by the reference ByteTrack/DeepSORT
//! implementations.
//!
//! State is 8-dimensional: centre x/y, aspect ratio `a = w / h`, height
//! `h`, and their velocities. Process and measurement noise scale with
//! the box height, so large (near) and small (far) players are tracked
//! with proportionate uncertainty. One `predict` step = one sampled
//! frame.

use nalgebra::{SMatrix, SVector};

pub type StateVec = SVector<f64, 8>;
pub type StateCov = SMatrix<f64, 8, 8>;
pub type MeasVec = SVector<f64, 4>;
type MeasCov = SMatrix<f64, 4, 4>;
type ObsMat = SMatrix<f64, 4, 8>;

/// Position noise as a fraction of box height.
const STD_WEIGHT_POSITION: f64 = 1.0 / 20.0;
/// Velocity noise as a fraction of box height.
const STD_WEIGHT_VELOCITY: f64 = 1.0 / 160.0;

#[derive(Debug, Clone)]
pub struct KalmanFilter {
    motion: StateCov,
    observation: ObsMat,
}

impl Default for KalmanFilter {
    fn default() -> Self {
        let mut motion = StateCov::identity();
        for i in 0..4 {
            motion[(i, i + 4)] = 1.0; // dt = 1 frame
        }
        let mut observation = ObsMat::zeros();
        for i in 0..4 {
            observation[(i, i)] = 1.0;
        }
        Self {
            motion,
            observation,
        }
    }
}

impl KalmanFilter {
    /// New track state from a first measurement: zero velocity, wide
    /// velocity uncertainty.
    pub fn initiate(&self, z: &MeasVec) -> (StateVec, StateCov) {
        let mut mean = StateVec::zeros();
        mean.fixed_rows_mut::<4>(0).copy_from(z);
        let h = z[3];
        let std = [
            2.0 * STD_WEIGHT_POSITION * h,
            2.0 * STD_WEIGHT_POSITION * h,
            1e-2,
            2.0 * STD_WEIGHT_POSITION * h,
            10.0 * STD_WEIGHT_VELOCITY * h,
            10.0 * STD_WEIGHT_VELOCITY * h,
            1e-5,
            10.0 * STD_WEIGHT_VELOCITY * h,
        ];
        (mean, diag_sq(&std))
    }

    /// Advances the state one frame.
    pub fn predict(&self, mean: &StateVec, cov: &StateCov) -> (StateVec, StateCov) {
        let h = mean[3];
        let std = [
            STD_WEIGHT_POSITION * h,
            STD_WEIGHT_POSITION * h,
            1e-2,
            STD_WEIGHT_POSITION * h,
            STD_WEIGHT_VELOCITY * h,
            STD_WEIGHT_VELOCITY * h,
            1e-5,
            STD_WEIGHT_VELOCITY * h,
        ];
        let mean = self.motion * mean;
        let cov = self.motion * cov * self.motion.transpose() + diag_sq(&std);
        (mean, cov)
    }

    /// Folds a measurement into the state.
    pub fn update(&self, mean: &StateVec, cov: &StateCov, z: &MeasVec) -> (StateVec, StateCov) {
        let h = mean[3];
        let r = diag_sq::<4>(&[
            STD_WEIGHT_POSITION * h,
            STD_WEIGHT_POSITION * h,
            1e-1,
            STD_WEIGHT_POSITION * h,
        ]);
        let projected_mean = self.observation * mean;
        let s: MeasCov = self.observation * cov * self.observation.transpose() + r;
        let pht = cov * self.observation.transpose(); // 8x4
                                                      // K = P Hᵀ S⁻¹. S is symmetric positive definite (P is, and R is a
                                                      // positive diagonal), so Cholesky always succeeds for finite input;
                                                      // fall back to the plain inverse just in case.
        let gain = match s.cholesky() {
            Some(chol) => (chol.solve(&pht.transpose())).transpose(),
            None => pht * s.try_inverse().unwrap_or_else(MeasCov::identity),
        };
        let innovation = z - projected_mean;
        let mean = mean + gain * innovation;
        let cov = cov - gain * s * gain.transpose();
        (mean, cov)
    }
}

fn diag_sq<const N: usize>(std: &[f64; N]) -> SMatrix<f64, N, N> {
    SMatrix::<f64, N, N>::from_diagonal(&SVector::<f64, N>::from_fn(|i, _| std[i] * std[i]))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn z(cx: f64, cy: f64, a: f64, h: f64) -> MeasVec {
        MeasVec::new(cx, cy, a, h)
    }

    #[test]
    fn initiate_has_zero_velocity_and_symmetric_covariance() {
        let kf = KalmanFilter::default();
        let (mean, cov) = kf.initiate(&z(100.0, 50.0, 0.4, 80.0));
        assert_eq!(&mean.as_slice()[..4], &[100.0, 50.0, 0.4, 80.0]);
        assert!(mean.as_slice()[4..].iter().all(|&v| v == 0.0));
        assert_eq!(cov, cov.transpose());
    }

    #[test]
    fn learns_constant_velocity_and_predicts_ahead() {
        let kf = KalmanFilter::default();
        // Target moving +5 px/frame in x, -2 in y.
        let (mut mean, mut cov) = kf.initiate(&z(100.0, 200.0, 0.4, 80.0));
        for t in 1..=30 {
            let (m, c) = kf.predict(&mean, &cov);
            let truth = z(100.0 + 5.0 * t as f64, 200.0 - 2.0 * t as f64, 0.4, 80.0);
            let (m, c) = kf.update(&m, &c, &truth);
            mean = m;
            cov = c;
        }
        assert!((mean[4] - 5.0).abs() < 0.1, "vx = {}", mean[4]);
        assert!((mean[5] + 2.0).abs() < 0.1, "vy = {}", mean[5]);

        // Coast 5 frames with no measurements (occlusion): the prediction
        // keeps moving along the learned velocity.
        for _ in 0..5 {
            let (m, c) = kf.predict(&mean, &cov);
            mean = m;
            cov = c;
        }
        assert!(
            (mean[0] - (100.0 + 5.0 * 35.0)).abs() < 2.0,
            "cx = {}",
            mean[0]
        );
        assert!(
            (mean[1] - (200.0 - 2.0 * 35.0)).abs() < 2.0,
            "cy = {}",
            mean[1]
        );
    }

    #[test]
    fn smooths_noisy_measurements_of_a_still_target() {
        let kf = KalmanFilter::default();
        let (mut mean, mut cov) = kf.initiate(&z(300.0, 300.0, 0.4, 100.0));
        // Deterministic ±4 px jitter around (300, 300).
        let jitter = [4.0, -4.0, 3.0, -3.0, 4.0, -2.0, 1.0, -4.0];
        for i in 0..80 {
            let (m, c) = kf.predict(&mean, &cov);
            let j = jitter[i % jitter.len()];
            let (m, c) = kf.update(&m, &c, &z(300.0 + j, 300.0 - j, 0.4, 100.0));
            mean = m;
            cov = c;
        }
        assert!((mean[0] - 300.0).abs() < 2.5, "cx = {}", mean[0]);
        assert!(mean[4].abs() < 1.0, "vx = {}", mean[4]);
    }

    #[test]
    fn uncertainty_grows_while_coasting_and_shrinks_on_update() {
        let kf = KalmanFilter::default();
        let (mean, cov) = kf.initiate(&z(0.0, 0.0, 0.5, 60.0));
        let (pm, pc) = kf.predict(&mean, &cov);
        assert!(pc[(0, 0)] > cov[(0, 0)]);
        let (_, uc) = kf.update(&pm, &pc, &z(0.0, 0.0, 0.5, 60.0));
        assert!(uc[(0, 0)] < pc[(0, 0)]);
    }
}
