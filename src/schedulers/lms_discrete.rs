//! LMS Discrete Scheduler
//!
//! Linear Multi-Step (LMS) scheduler for diffusion models.
//! Uses a linear combination of previous model outputs to predict the next sample.
//!
//! Reference: https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_lms_discrete.py

use alloc::vec;
use alloc::vec::Vec;
use burn::tensor::{backend::Backend, Tensor};

#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::Float;

use super::integrate::integrate;
use super::{BetaSchedule, PredictionType};

/// Configuration for the LMS Discrete Scheduler.
#[derive(Debug, Clone)]
pub struct LMSDiscreteSchedulerConfig {
    /// The value of beta at the beginning of training.
    pub beta_start: f64,
    /// The value of beta at the end of training.
    pub beta_end: f64,
    /// How beta evolved during training.
    pub beta_schedule: BetaSchedule,
    /// Number of diffusion steps used to train the model.
    pub train_timesteps: usize,
    /// Order of the linear multi-step method.
    pub order: usize,
    /// Prediction type of the scheduler function.
    pub prediction_type: PredictionType,
}

impl Default for LMSDiscreteSchedulerConfig {
    fn default() -> Self {
        Self {
            beta_start: 0.00085,
            beta_end: 0.012,
            beta_schedule: BetaSchedule::ScaledLinear,
            train_timesteps: 1000,
            order: 4,
            prediction_type: PredictionType::Epsilon,
        }
    }
}

/// LMS Discrete Scheduler for diffusion models.
///
/// This scheduler implements the Linear Multi-Step method for solving
/// the probability flow ODE in diffusion models.
#[derive(Debug, Clone)]
pub struct LMSDiscreteScheduler<B: Backend> {
    timesteps: Vec<f64>,
    sigmas: Vec<f64>,
    init_noise_sigma: f64,
    derivatives: Vec<Tensor<B, 4>>,
    /// The scheduler configuration.
    pub config: LMSDiscreteSchedulerConfig,
}

impl<B: Backend> LMSDiscreteScheduler<B> {
    /// Create a new LMS Discrete Scheduler.
    ///
    /// # Arguments
    /// * `inference_steps` - Number of inference steps
    /// * `config` - Scheduler configuration
    pub fn new(inference_steps: usize, config: LMSDiscreteSchedulerConfig) -> Self {
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
                    "LMSDiscreteScheduler only implements linear and scaled_linear betas."
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
        let sigmas = interp(&timesteps, &xp, &sigmas_full);

        // Append 0.0 to sigmas
        let mut sigmas = sigmas;
        sigmas.push(0.0);

        // init_noise_sigma = max(sigmas)
        let init_noise_sigma = sigmas.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        Self {
            timesteps,
            sigmas,
            init_noise_sigma,
            derivatives: vec![],
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

    /// Scale the model input by the appropriate sigma value.
    ///
    /// # Arguments
    /// * `sample` - The input sample tensor
    /// * `timestep` - The current timestep
    pub fn scale_model_input(&self, sample: Tensor<B, 4>, timestep: f64) -> Tensor<B, 4> {
        let step_index = self
            .timesteps
            .iter()
            .position(|&t| t == timestep)
            .expect("Timestep not found in scheduler timesteps");
        let sigma = self.sigmas[step_index];

        // sample / sqrt(sigma^2 + 1)
        let scale = (sigma.powi(2) + 1.0).sqrt();
        sample / scale
    }

    /// Compute a linear multistep coefficient.
    fn get_lms_coefficient(&self, order: usize, t: usize, current_order: usize) -> f64 {
        let sigmas = &self.sigmas;

        let lms_derivative = |tau: f64| -> f64 {
            let mut prod = 1.0;
            for k in 0..order {
                if current_order == k {
                    continue;
                }
                prod *= (tau - sigmas[t - k]) / (sigmas[t - current_order] - sigmas[t - k]);
            }
            prod
        };

        // Integrate `lms_derivative` over two consecutive timesteps.
        let integration_out = integrate(lms_derivative, sigmas[t], sigmas[t + 1], 1.49e-8);
        integration_out.integral
    }

    /// Perform one step of the LMS method.
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
        let step_index = self
            .timesteps
            .iter()
            .position(|&t| t == timestep)
            .expect("Timestep not found in scheduler timesteps");
        let sigma = self.sigmas[step_index];

        // 1. Compute predicted original sample (x_0) from sigma-scaled predicted noise
        let pred_original_sample = match self.config.prediction_type {
            PredictionType::Epsilon => sample.clone() - model_output.clone() * sigma,
            PredictionType::VPrediction => {
                let sigma_sq_plus_1 = sigma.powi(2) + 1.0;
                model_output.clone() * (-sigma / sigma_sq_plus_1.sqrt())
                    + sample.clone() / sigma_sq_plus_1
            }
            PredictionType::Sample => {
                unimplemented!("Prediction type must be one of `epsilon` or `v_prediction`")
            }
        };

        // 2. Convert to an ODE derivative
        let derivative = (sample.clone() - pred_original_sample) / sigma;
        self.derivatives.push(derivative);
        if self.derivatives.len() > self.config.order {
            // Remove the first element
            self.derivatives.remove(0);
        }

        // 3. Compute linear multistep coefficients
        let order = self.config.order.min(step_index + 1);
        let lms_coeffs: Vec<f64> = (0..order)
            .map(|o| self.get_lms_coefficient(order, step_index, o))
            .collect();

        // 4. Compute previous sample based on the derivatives path
        let mut deriv_sum = self.derivatives.last().unwrap().clone() * lms_coeffs[0];
        for (coeff, derivative) in lms_coeffs
            .iter()
            .skip(1)
            .zip(self.derivatives.iter().rev().skip(1))
        {
            deriv_sum = deriv_sum + derivative.clone() * *coeff;
        }

        sample.clone() + deriv_sum
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
        let step_index = self
            .timesteps
            .iter()
            .position(|&t| t == timestep)
            .expect("Timestep not found in scheduler timesteps");
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
    fn test_lms_discrete_scheduler_creation() {
        let config = LMSDiscreteSchedulerConfig::default();
        let scheduler = LMSDiscreteScheduler::<TestBackend>::new(20, config);

        assert_eq!(scheduler.timesteps().len(), 20);
        assert_eq!(scheduler.sigmas.len(), 21); // 20 + 1 (appended 0)
        assert!(scheduler.init_noise_sigma() > 0.0);
    }

    #[test]
    fn test_lms_discrete_timesteps() {
        let config = LMSDiscreteSchedulerConfig::default();
        let scheduler = LMSDiscreteScheduler::<TestBackend>::new(10, config);

        let timesteps = scheduler.timesteps();
        // Should be decreasing from train_timesteps-1 to 0
        assert!((timesteps[0] - 999.0).abs() < 0.1);
        assert!((timesteps[9] - 0.0).abs() < 0.1);

        // Should be monotonically decreasing
        for i in 1..timesteps.len() {
            assert!(timesteps[i] < timesteps[i - 1]);
        }
    }

    #[test]
    fn test_lms_discrete_scale_model_input() {
        let device = Default::default();
        let config = LMSDiscreteSchedulerConfig::default();
        let scheduler = LMSDiscreteScheduler::<TestBackend>::new(20, config);

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
    fn test_lms_discrete_step() {
        let device = Default::default();
        let config = LMSDiscreteSchedulerConfig::default();
        let mut scheduler = LMSDiscreteScheduler::<TestBackend>::new(20, config);

        let model_output: Tensor<TestBackend, 4> = Tensor::zeros([1, 4, 8, 8], &device);
        let sample: Tensor<TestBackend, 4> = Tensor::ones([1, 4, 8, 8], &device);
        let timestep = scheduler.timesteps()[0];

        let result = scheduler.step(&model_output, timestep, &sample);
        assert_eq!(result.shape(), Shape::from([1, 4, 8, 8]));
    }

    /// Test LMS Discrete scheduler values match diffusers-rs reference values
    #[test]
    fn test_lms_discrete_matches_diffusers_rs() {
        let device = Default::default();
        let config = LMSDiscreteSchedulerConfig::default();
        let scheduler = LMSDiscreteScheduler::<TestBackend>::new(20, config);

        // Reference init_noise_sigma from diffusers-rs: 14.614646291831562
        let init_sigma = scheduler.init_noise_sigma();
        assert!(
            (init_sigma - 14.614646291831562).abs() < 1e-4,
            "init_noise_sigma mismatch: actual={}, expected=14.614646291831562",
            init_sigma
        );

        // Test scale_model_input
        let sample: Tensor<TestBackend, 4> = Tensor::ones([1, 4, 8, 8], &device);
        let timestep = scheduler.timesteps()[0];
        let scaled = scheduler.scale_model_input(sample, timestep);
        let scaled_mean: f32 = scaled.mean().into_scalar();

        // Reference from diffusers-rs: 0.06826489418745041
        assert!(
            (scaled_mean as f64 - 0.06826489418745041).abs() < 1e-4,
            "scale_model_input mean mismatch: actual={}, expected=0.06826489418745041",
            scaled_mean
        );
    }

    #[test]
    fn test_lms_discrete_derivatives_accumulation() {
        let device = Default::default();
        let config = LMSDiscreteSchedulerConfig {
            order: 4,
            ..Default::default()
        };
        let mut scheduler = LMSDiscreteScheduler::<TestBackend>::new(20, config);

        let sample: Tensor<TestBackend, 4> = Tensor::ones([1, 4, 8, 8], &device);
        let model_output: Tensor<TestBackend, 4> = Tensor::zeros([1, 4, 8, 8], &device);

        let timesteps = scheduler.timesteps().to_vec();

        // Run several steps and check derivatives accumulation
        for (i, &ts) in timesteps.iter().enumerate().take(6) {
            let _ = scheduler.step(&model_output, ts, &sample);
            // Derivatives should accumulate up to order, then stay at order
            let expected_len = (i + 1).min(4);
            assert_eq!(
                scheduler.derivatives.len(),
                expected_len,
                "Derivatives length mismatch at step {}: actual={}, expected={}",
                i,
                scheduler.derivatives.len(),
                expected_len
            );
        }
    }
}
