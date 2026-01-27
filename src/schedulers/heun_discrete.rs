//! Heun Discrete Scheduler
//!
//! The Heun scheduler is a second-order Runge-Kutta method for solving differential equations.
//! Based on the algorithm described in Karras et al. (2022) https://arxiv.org/abs/2206.00364.
//! Reference: https://github.com/crowsonkb/k-diffusion/blob/main/k_diffusion/sampling.py

use alloc::vec;
use alloc::vec::Vec;
use burn::tensor::{backend::Backend, Tensor};

#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::Float;

use super::{BetaSchedule, PredictionType};

/// Configuration for the Heun Discrete Scheduler.
#[derive(Debug, Clone)]
pub struct HeunDiscreteSchedulerConfig {
    /// The value of beta at the beginning of training.
    pub beta_start: f64,
    /// The value of beta at the end of training.
    pub beta_end: f64,
    /// How beta evolved during training.
    pub beta_schedule: BetaSchedule,
    /// Number of diffusion steps used to train the model.
    pub train_timesteps: usize,
    /// Prediction type of the scheduler function.
    pub prediction_type: PredictionType,
}

impl Default for HeunDiscreteSchedulerConfig {
    fn default() -> Self {
        Self {
            beta_start: 0.00085,
            beta_end: 0.012,
            beta_schedule: BetaSchedule::Linear,
            train_timesteps: 1000,
            prediction_type: PredictionType::Epsilon,
        }
    }
}

/// Heun Discrete Scheduler for diffusion models.
///
/// This scheduler implements the Heun method (a second-order Runge-Kutta method)
/// for solving the probability flow ODE in diffusion models.
#[derive(Debug, Clone)]
pub struct HeunDiscreteScheduler<B: Backend> {
    timesteps: Vec<f64>,
    sigmas: Vec<f64>,
    init_noise_sigma: f64,
    prev_derivative: Option<Tensor<B, 4>>,
    sample: Option<Tensor<B, 4>>,
    dt: Option<f64>,
    /// The scheduler configuration.
    pub config: HeunDiscreteSchedulerConfig,
}

impl<B: Backend> HeunDiscreteScheduler<B> {
    /// Create a new Heun Discrete Scheduler.
    ///
    /// # Arguments
    /// * `inference_steps` - Number of inference steps
    /// * `config` - Scheduler configuration
    pub fn new(inference_steps: usize, config: HeunDiscreteSchedulerConfig) -> Self {
        let betas = match config.beta_schedule {
            BetaSchedule::ScaledLinear => {
                let start = config.beta_start.sqrt();
                let end = config.beta_end.sqrt();
                let step = (end - start) / (config.train_timesteps - 1) as f64;
                (0..config.train_timesteps)
                    .map(|i| {
                        let v = start + step * i as f64;
                        v * v
                    })
                    .collect::<Vec<_>>()
            }
            BetaSchedule::Linear => {
                let step =
                    (config.beta_end - config.beta_start) / (config.train_timesteps - 1) as f64;
                (0..config.train_timesteps)
                    .map(|i| config.beta_start + step * i as f64)
                    .collect()
            }
            BetaSchedule::SquaredcosCapV2 => {
                unimplemented!(
                    "HeunDiscreteScheduler only implements linear and scaled_linear betas."
                )
            }
        };

        // alphas = 1 - betas
        let alphas: Vec<f64> = betas.iter().map(|b| 1.0 - b).collect();

        // alphas_cumprod = cumprod(alphas)
        let mut alphas_cumprod: Vec<f64> = Vec::with_capacity(config.train_timesteps);
        let mut cumprod = 1.0;
        for alpha in &alphas {
            cumprod *= alpha;
            alphas_cumprod.push(cumprod);
        }

        // timesteps = linspace(train_timesteps - 1, 0, inference_steps)
        let timesteps: Vec<f64> =
            linspace((config.train_timesteps - 1) as f64, 0.0, inference_steps);

        // sigmas = sqrt((1 - alphas_cumprod) / alphas_cumprod)
        let sigmas_full: Vec<f64> = alphas_cumprod
            .iter()
            .map(|&acp| ((1.0 - acp) / acp).sqrt())
            .collect();

        // Interpolate sigmas at timestep positions
        let xp: Vec<f64> = (0..sigmas_full.len()).map(|i| i as f64).collect();
        let sigmas_interp = interp(&timesteps, &xp, &sigmas_full);

        // For Heun scheduler:
        // sigmas = cat([sigmas[:1], sigmas[1:].repeat_interleave(2), [0.0]])
        let mut sigmas = vec![sigmas_interp[0]];
        for &s in &sigmas_interp[1..] {
            sigmas.push(s);
            sigmas.push(s);
        }
        sigmas.push(0.0);

        // init_noise_sigma = max(sigmas)
        let init_noise_sigma = sigmas.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        // timesteps = cat([timesteps[:1], timesteps[1:].repeat_interleave(2)])
        let mut ts = vec![timesteps[0]];
        for &t in &timesteps[1..] {
            ts.push(t);
            ts.push(t);
        }

        Self {
            timesteps: ts,
            sigmas,
            init_noise_sigma,
            prev_derivative: None,
            sample: None,
            dt: None,
            config,
        }
    }

    /// Get the timesteps for the scheduler.
    pub fn timesteps(&self) -> &[f64] {
        self.timesteps.as_slice()
    }

    /// Get the initial noise sigma value.
    pub fn init_noise_sigma(&self) -> f64 {
        self.init_noise_sigma
    }

    /// Check if the scheduler is in first-order mode.
    fn state_in_first_order(&self) -> bool {
        self.dt.is_none()
    }

    /// Find the index for a given timestep.
    fn index_for_timestep(&self, timestep: f64) -> usize {
        let indices: Vec<usize> = self
            .timesteps
            .iter()
            .enumerate()
            .filter_map(|(idx, &t)| if t == timestep { Some(idx) } else { None })
            .collect();

        if self.state_in_first_order() {
            *indices.last().unwrap()
        } else {
            indices[0]
        }
    }

    /// Scale the model input by the appropriate sigma value.
    ///
    /// # Arguments
    /// * `sample` - The input sample tensor
    /// * `timestep` - The current timestep
    pub fn scale_model_input(&self, sample: Tensor<B, 4>, timestep: f64) -> Tensor<B, 4> {
        let step_index = self.index_for_timestep(timestep);
        let sigma = self.sigmas[step_index];

        // sample / sqrt(sigma^2 + 1)
        let scale = (sigma.powi(2) + 1.0).sqrt();
        sample / scale
    }

    /// Perform one step of the Heun method.
    ///
    /// # Arguments
    /// * `model_output` - The model's predicted noise
    /// * `timestep` - The current timestep
    /// * `sample` - The current noisy sample
    pub fn step(
        &mut self,
        model_output: &Tensor<B, 4>,
        timestep: f64,
        sample: &Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        let step_index = self.index_for_timestep(timestep);

        let (sigma, sigma_next) = if self.state_in_first_order() {
            (self.sigmas[step_index], self.sigmas[step_index + 1])
        } else {
            // 2nd order / Heun's method
            (self.sigmas[step_index - 1], self.sigmas[step_index])
        };

        // Currently only gamma=0 is supported
        let gamma = 0.0;
        let sigma_hat = sigma * (gamma + 1.0); // sigma_hat == sigma for now

        // 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
        let sigma_input = if self.state_in_first_order() {
            sigma_hat
        } else {
            sigma_next
        };

        let pred_original_sample = match self.config.prediction_type {
            PredictionType::Epsilon => sample.clone() - model_output.clone() * sigma_input,
            PredictionType::VPrediction => {
                let sigma_sq_plus_1 = sigma_input.powi(2) + 1.0;
                model_output.clone() * (-sigma_input / sigma_sq_plus_1.sqrt())
                    + sample.clone() / sigma_sq_plus_1
            }
            PredictionType::Sample => {
                unimplemented!("Prediction type must be one of `epsilon` or `v_prediction`")
            }
        };

        let (derivative, dt, sample_out) = if self.state_in_first_order() {
            // 2. Convert to an ODE derivative for 1st order
            (
                (sample.clone() - pred_original_sample) / sigma_hat,
                sigma_next - sigma_hat,
                sample.clone(),
            )
        } else {
            // 2. 2nd order / Heun's method
            let derivative = (sample.clone() - pred_original_sample) / sigma_next;
            (
                (self.prev_derivative.as_ref().unwrap().clone() + derivative) / 2.0,
                self.dt.unwrap(),
                self.sample.as_ref().unwrap().clone(),
            )
        };

        if self.state_in_first_order() {
            // Store for 2nd order step
            self.prev_derivative = Some(derivative.clone());
            self.dt = Some(dt);
            self.sample = Some(sample.clone());
        } else {
            // Free dt and derivative - puts scheduler back in "first order mode"
            self.prev_derivative = None;
            self.dt = None;
            self.sample = None;
        }

        sample_out + derivative * dt
    }

    /// Add noise to original samples.
    ///
    /// # Arguments
    /// * `original_samples` - The original clean samples
    /// * `noise` - The noise to add
    /// * `timestep` - The timestep at which to add noise
    pub fn add_noise(
        &self,
        original_samples: &Tensor<B, 4>,
        noise: Tensor<B, 4>,
        timestep: f64,
    ) -> Tensor<B, 4> {
        let step_index = self.index_for_timestep(timestep);
        let sigma = self.sigmas[step_index];

        original_samples.clone() + noise * sigma
    }
}

/// Create a linearly spaced vector from start to end with n points.
fn linspace(start: f64, end: f64, n: usize) -> Vec<f64> {
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![start];
    }
    let step = (end - start) / (n - 1) as f64;
    (0..n).map(|i| start + step * i as f64).collect()
}

/// One-dimensional linear interpolation for monotonically increasing sample points.
fn interp(x: &[f64], xp: &[f64], yp: &[f64]) -> Vec<f64> {
    assert_eq!(xp.len(), yp.len());
    let sz = xp.len();

    let m: Vec<f64> = (0..sz - 1)
        .map(|i| (yp[i + 1] - yp[i]) / (xp[i + 1] - xp[i]))
        .collect();

    let b: Vec<f64> = (0..sz - 1).map(|i| yp[i] - m[i] * xp[i]).collect();

    x.iter()
        .map(|&xi| {
            let mut idx = 0;
            for (i, &xp_val) in xp.iter().enumerate() {
                if xi >= xp_val {
                    idx = i;
                }
            }
            let idx = idx.min(m.len() - 1);
            m[idx] * xi + b[idx]
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestBackend;
    use burn::tensor::Shape;

    #[test]
    fn test_heun_discrete_scheduler_creation() {
        let config = HeunDiscreteSchedulerConfig::default();
        let scheduler = HeunDiscreteScheduler::<TestBackend>::new(20, config);

        // Heun scheduler has: 1 + (inference_steps - 1) * 2 timesteps
        assert_eq!(scheduler.timesteps().len(), 39); // 1 + 19*2 = 39
                                                     // Sigmas: 1 + (inference_steps - 1) * 2 + 1 (appended 0)
        assert_eq!(scheduler.sigmas.len(), 40);
        assert!(scheduler.init_noise_sigma() > 0.0);
    }

    #[test]
    fn test_heun_discrete_timesteps() {
        let config = HeunDiscreteSchedulerConfig::default();
        let scheduler = HeunDiscreteScheduler::<TestBackend>::new(10, config);

        let timesteps = scheduler.timesteps();
        // First timestep should be close to train_timesteps - 1
        assert!((timesteps[0] - 999.0).abs() < 0.1);
    }

    #[test]
    fn test_heun_discrete_scale_model_input() {
        let device = Default::default();
        let config = HeunDiscreteSchedulerConfig::default();
        let scheduler = HeunDiscreteScheduler::<TestBackend>::new(20, config);

        let sample: Tensor<TestBackend, 4> = Tensor::ones([1, 4, 8, 8], &device);
        let timestep = scheduler.timesteps()[0];

        let scaled = scheduler.scale_model_input(sample, timestep);
        assert_eq!(scaled.shape(), Shape::from([1, 4, 8, 8]));

        // Scaled values should be less than original
        let scaled_mean: f32 = scaled.mean().into_scalar();
        assert!(scaled_mean < 1.0);
        assert!(scaled_mean > 0.0);
    }

    #[test]
    fn test_heun_discrete_step() {
        let device = Default::default();
        let config = HeunDiscreteSchedulerConfig::default();
        let mut scheduler = HeunDiscreteScheduler::<TestBackend>::new(20, config);

        let model_output: Tensor<TestBackend, 4> = Tensor::zeros([1, 4, 8, 8], &device);
        let sample: Tensor<TestBackend, 4> = Tensor::ones([1, 4, 8, 8], &device);
        let timestep = scheduler.timesteps()[0];

        let result = scheduler.step(&model_output, timestep, &sample);
        assert_eq!(result.shape(), Shape::from([1, 4, 8, 8]));
    }

    /// Test Heun Discrete scheduler values match diffusers-rs reference values
    /// Note: Using ScaledLinear beta schedule to match the reference values from Euler Discrete
    #[test]
    fn test_heun_discrete_matches_diffusers_rs() {
        let device = Default::default();
        // Use ScaledLinear to match reference values (same as EulerDiscrete)
        let config = HeunDiscreteSchedulerConfig {
            beta_schedule: super::super::BetaSchedule::ScaledLinear,
            ..Default::default()
        };
        let scheduler = HeunDiscreteScheduler::<TestBackend>::new(20, config);

        // Reference init_noise_sigma from diffusers-rs with ScaledLinear: 14.614646291831562
        let init_sigma = scheduler.init_noise_sigma();
        assert!(
            (init_sigma - 14.614646291831562).abs() < 1e-4,
            "init_noise_sigma mismatch: actual={}, expected=14.614646291831562",
            init_sigma
        );

        // Check first few sigmas match expected pattern
        // First sigma should be the maximum
        assert!(
            (scheduler.sigmas[0] - init_sigma).abs() < 1e-10,
            "First sigma should equal init_noise_sigma"
        );

        // Test scale_model_input
        let sample: Tensor<TestBackend, 4> = Tensor::ones([1, 4, 8, 8], &device);
        let timestep = scheduler.timesteps()[0];
        let scaled = scheduler.scale_model_input(sample, timestep);
        let scaled_mean: f32 = scaled.mean().into_scalar();

        // Reference from diffusers-rs with ScaledLinear: 0.06826489418745041
        assert!(
            (scaled_mean as f64 - 0.06826489418745041).abs() < 1e-4,
            "scale_model_input mean mismatch: actual={}, expected=0.06826489418745041",
            scaled_mean
        );
    }

    #[test]
    fn test_heun_discrete_two_step_cycle() {
        let device = Default::default();
        let config = HeunDiscreteSchedulerConfig::default();
        let mut scheduler = HeunDiscreteScheduler::<TestBackend>::new(20, config);

        let sample: Tensor<TestBackend, 4> = Tensor::ones([1, 4, 8, 8], &device);
        let model_output: Tensor<TestBackend, 4> = Tensor::zeros([1, 4, 8, 8], &device);

        // Heun scheduler alternates between first and second order
        assert!(scheduler.state_in_first_order());

        let timesteps = scheduler.timesteps().to_vec();

        // First step (first order)
        let _ = scheduler.step(&model_output, timesteps[0], &sample);
        assert!(!scheduler.state_in_first_order());

        // Second step (second order)
        let _ = scheduler.step(&model_output, timesteps[1], &sample);
        assert!(scheduler.state_in_first_order());
    }
}
