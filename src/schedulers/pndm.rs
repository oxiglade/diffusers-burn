//! PNDM Scheduler
//!
//! Pseudo numerical methods for diffusion models (PNDM) proposes using more
//! advanced ODE integration techniques, namely Runge-Kutta method and a
//! linear multi-step method.
//! Based on the paper: https://arxiv.org/abs/2202.09778

use alloc::vec;
use alloc::vec::Vec;
use burn::tensor::{backend::Backend, Tensor};

use super::{betas_for_alpha_bar, BetaSchedule, PredictionType};

#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::Float;

/// Configuration for the PNDM Scheduler.
#[derive(Debug, Clone)]
pub struct PNDMSchedulerConfig {
    /// The value of beta at the beginning of training.
    pub beta_start: f64,
    /// The value of beta at the end of training.
    pub beta_end: f64,
    /// How beta evolved during training.
    pub beta_schedule: BetaSchedule,
    /// Each diffusion step uses the value of alphas product at that step and
    /// at the previous one. For the final step there is no previous alpha.
    /// When this option is `true` the previous alpha product is fixed to `1`,
    /// otherwise it uses the value of alpha at step 0.
    pub set_alpha_to_one: bool,
    /// Prediction type of the scheduler function.
    pub prediction_type: PredictionType,
    /// An offset added to the inference steps.
    pub steps_offset: usize,
    /// Number of diffusion steps used to train the model.
    pub train_timesteps: usize,
}

impl Default for PNDMSchedulerConfig {
    fn default() -> Self {
        Self {
            beta_start: 0.00085,
            beta_end: 0.012,
            beta_schedule: BetaSchedule::ScaledLinear,
            set_alpha_to_one: false,
            prediction_type: PredictionType::Epsilon,
            steps_offset: 1,
            train_timesteps: 1000,
        }
    }
}

/// PNDM Scheduler for diffusion models.
///
/// This scheduler implements the PLMS method for fast sampling in diffusion models.
pub struct PNDMScheduler<B: Backend> {
    alphas_cumprod: Vec<f64>,
    final_alpha_cumprod: f64,
    step_ratio: usize,
    init_noise_sigma: f64,
    counter: usize,
    cur_sample: Option<Tensor<B, 4>>,
    ets: Vec<Tensor<B, 4>>,
    timesteps: Vec<usize>,
    /// The scheduler configuration.
    pub config: PNDMSchedulerConfig,
}

impl<B: Backend> PNDMScheduler<B> {
    /// Create a new PNDM Scheduler.
    ///
    /// # Arguments
    /// * `inference_steps` - Number of inference steps
    /// * `config` - Scheduler configuration
    /// * `device` - The device to create tensors on
    pub fn new(inference_steps: usize, config: PNDMSchedulerConfig, device: &B::Device) -> Self {
        let betas: Vec<f64> = match config.beta_schedule {
            BetaSchedule::ScaledLinear => {
                let start = config.beta_start.sqrt();
                let end = config.beta_end.sqrt();
                let step = (end - start) / (config.train_timesteps - 1) as f64;
                (0..config.train_timesteps)
                    .map(|i| {
                        let v = start + step * i as f64;
                        v * v
                    })
                    .collect()
            }
            BetaSchedule::Linear => {
                let step =
                    (config.beta_end - config.beta_start) / (config.train_timesteps - 1) as f64;
                (0..config.train_timesteps)
                    .map(|i| config.beta_start + step * i as f64)
                    .collect()
            }
            BetaSchedule::SquaredcosCapV2 => {
                let betas_tensor: Tensor<B, 1> =
                    betas_for_alpha_bar(config.train_timesteps, 0.999, device);
                let data = betas_tensor.into_data();
                data.to_vec::<f32>()
                    .unwrap()
                    .into_iter()
                    .map(|x| x as f64)
                    .collect()
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

        let final_alpha_cumprod = if config.set_alpha_to_one {
            1.0
        } else {
            alphas_cumprod[0]
        };

        // Create integer timesteps by multiplying by ratio
        let step_ratio = config.train_timesteps / inference_steps;
        let timesteps: Vec<usize> = (0..inference_steps)
            .map(|s| s * step_ratio + config.steps_offset)
            .collect();

        // Create PLMS timesteps
        // plms_timesteps = [timesteps[:-2], timesteps[-2], timesteps[-2:]]
        let n_ts = timesteps.len();
        let mut plms_timesteps = Vec::new();
        // timesteps[:-2]
        plms_timesteps.extend_from_slice(&timesteps[..n_ts - 2]);
        // timesteps[-2] (duplicate)
        plms_timesteps.push(timesteps[n_ts - 2]);
        // timesteps[-2:]
        plms_timesteps.extend_from_slice(&timesteps[n_ts - 2..]);
        // Reverse
        plms_timesteps.reverse();

        Self {
            alphas_cumprod,
            final_alpha_cumprod,
            step_ratio,
            init_noise_sigma: 1.0,
            counter: 0,
            cur_sample: None,
            ets: vec![],
            timesteps: plms_timesteps,
            config,
        }
    }

    /// Get the timesteps for the scheduler.
    pub fn timesteps(&self) -> &[usize] {
        self.timesteps.as_slice()
    }

    /// Get the initial noise sigma value.
    pub fn init_noise_sigma(&self) -> f64 {
        self.init_noise_sigma
    }

    /// Scale the model input (identity for PNDM).
    pub fn scale_model_input(&self, sample: Tensor<B, 4>, _timestep: usize) -> Tensor<B, 4> {
        sample
    }

    /// Perform one step of the PNDM (using PLMS method).
    pub fn step(
        &mut self,
        model_output: &Tensor<B, 4>,
        timestep: usize,
        sample: &Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        self.step_plms(model_output, timestep, sample)
    }

    /// Step function propagating the sample with the linear multi-step method.
    fn step_plms(
        &mut self,
        model_output: &Tensor<B, 4>,
        mut timestep: usize,
        sample: &Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        let mut prev_timestep = timestep as isize - self.step_ratio as isize;

        if self.counter != 1 {
            // Make sure ets has at most 4 elements (keep last 3)
            if self.ets.len() > 3 {
                self.ets.drain(0..self.ets.len() - 3);
            }
            self.ets.push(model_output.clone());
        } else {
            prev_timestep = timestep as isize;
            timestep += self.step_ratio;
        }

        let n_ets = self.ets.len();
        let (mut model_output, mut sample) = (model_output.clone(), sample.clone());

        if n_ets == 1 && self.counter == 0 {
            self.cur_sample = Some(sample.clone());
        } else if n_ets == 1 && self.counter == 1 {
            sample = self.cur_sample.take().unwrap();
            model_output = (model_output + self.ets.last().unwrap().clone()) / 2.0;
        } else if n_ets == 2 {
            let ets_last = self.ets.last().unwrap();
            model_output = (ets_last.clone() * 3.0 - self.ets[n_ets - 2].clone()) / 2.0;
        } else if n_ets == 3 {
            let ets_last = self.ets.last().unwrap();
            model_output = (ets_last.clone() * 23.0 - self.ets[n_ets - 2].clone() * 16.0
                + self.ets[n_ets - 3].clone() * 5.0)
                / 12.0;
        } else {
            let ets_last = self.ets.last().unwrap();
            model_output = (ets_last.clone() * 55.0 - self.ets[n_ets - 2].clone() * 59.0
                + self.ets[n_ets - 3].clone() * 37.0
                - self.ets[n_ets - 4].clone() * 9.0)
                * (1.0 / 24.0);
        }

        let prev_sample = self.get_prev_sample(sample, timestep, prev_timestep, model_output);
        self.counter += 1;

        prev_sample
    }

    /// Compute the previous sample using the PNDM formula.
    fn get_prev_sample(
        &self,
        sample: Tensor<B, 4>,
        timestep: usize,
        prev_timestep: isize,
        model_output: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        // See formula (9) of PNDM paper https://arxiv.org/pdf/2202.09778.pdf
        let alpha_prod_t = self.alphas_cumprod[timestep];
        let alpha_prod_t_prev = if prev_timestep >= 0 {
            self.alphas_cumprod[prev_timestep as usize]
        } else {
            self.final_alpha_cumprod
        };

        let beta_prod_t = 1.0 - alpha_prod_t;
        let beta_prod_t_prev = 1.0 - alpha_prod_t_prev;

        let model_output = match self.config.prediction_type {
            PredictionType::VPrediction => {
                model_output * alpha_prod_t.sqrt() + sample.clone() * beta_prod_t.sqrt()
            }
            PredictionType::Epsilon => model_output,
            PredictionType::Sample => {
                unimplemented!("Prediction type must be one of `epsilon` or `v_prediction`")
            }
        };

        // Corresponds to (α_(t−δ) - α_t) divided by
        // denominator of x_t in formula (9) and plus 1
        let sample_coeff = (alpha_prod_t_prev / alpha_prod_t).sqrt();

        // Corresponds to denominator of e_θ(x_t, t) in formula (9)
        let model_output_denom_coeff = alpha_prod_t * beta_prod_t_prev.sqrt()
            + (alpha_prod_t * beta_prod_t * alpha_prod_t_prev).sqrt();

        // Full formula (9)
        sample * sample_coeff
            - model_output * (alpha_prod_t_prev - alpha_prod_t) / model_output_denom_coeff
    }

    /// Add noise to original samples.
    pub fn add_noise(
        &self,
        original_samples: &Tensor<B, 4>,
        noise: Tensor<B, 4>,
        timestep: usize,
    ) -> Tensor<B, 4> {
        let timestep = if timestep >= self.alphas_cumprod.len() {
            timestep - 1
        } else {
            timestep
        };

        let sqrt_alpha_prod = self.alphas_cumprod[timestep].sqrt();
        let sqrt_one_minus_alpha_prod = (1.0 - self.alphas_cumprod[timestep]).sqrt();

        original_samples.clone() * sqrt_alpha_prod + noise * sqrt_one_minus_alpha_prod
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestBackend;
    use burn::tensor::Shape;

    #[test]
    fn test_pndm_scheduler_creation() {
        let device = Default::default();
        let config = PNDMSchedulerConfig::default();
        let scheduler = PNDMScheduler::<TestBackend>::new(20, config, &device);

        // PNDM has 21 timesteps due to the PLMS method
        assert_eq!(scheduler.timesteps().len(), 21);
        assert_eq!(scheduler.init_noise_sigma(), 1.0);
    }

    #[test]
    fn test_pndm_timesteps() {
        let device = Default::default();
        let config = PNDMSchedulerConfig::default();
        let scheduler = PNDMScheduler::<TestBackend>::new(20, config, &device);

        let timesteps = scheduler.timesteps();
        // First timestep should be high
        assert!(timesteps[0] > 900);
        // Last timestep should be small
        assert!(timesteps[timesteps.len() - 1] < 10);
    }

    #[test]
    fn test_pndm_scale_model_input() {
        let device = Default::default();
        let config = PNDMSchedulerConfig::default();
        let scheduler = PNDMScheduler::<TestBackend>::new(20, config, &device);

        let sample: Tensor<TestBackend, 4> = Tensor::ones([1, 4, 8, 8], &device);
        let timestep = scheduler.timesteps()[0];

        // PNDM doesn't scale input
        let scaled = scheduler.scale_model_input(sample.clone(), timestep);
        let diff: f32 = (scaled - sample).abs().mean().into_scalar();
        assert!(diff < 1e-6);
    }

    #[test]
    fn test_pndm_step() {
        let device = Default::default();
        let config = PNDMSchedulerConfig::default();
        let mut scheduler = PNDMScheduler::<TestBackend>::new(20, config, &device);

        let model_output: Tensor<TestBackend, 4> = Tensor::zeros([1, 4, 8, 8], &device);
        let sample: Tensor<TestBackend, 4> = Tensor::ones([1, 4, 8, 8], &device);
        let timestep = scheduler.timesteps()[0];

        let result = scheduler.step(&model_output, timestep, &sample);
        assert_eq!(result.shape(), Shape::from([1, 4, 8, 8]));

        // Result should be finite
        let result_data = result.into_data();
        let values: Vec<f32> = result_data.to_vec().unwrap();
        for v in &values {
            assert!(v.is_finite(), "Result contains non-finite values");
        }
    }

    /// Test PNDM scheduler values match diffusers-rs
    #[test]
    fn test_pndm_matches_diffusers_rs() {
        let device = Default::default();
        let config = PNDMSchedulerConfig::default();
        let mut scheduler = PNDMScheduler::<TestBackend>::new(20, config, &device);

        // Reference values from diffusers-rs
        // Timesteps: [951, 901, 901, 851, 801, 751, 701, 651, 601, 551, 501, 451, 401, 351, 301, 251, 201, 151, 101, 51, 1]
        let expected_timesteps = [
            951, 901, 901, 851, 801, 751, 701, 651, 601, 551, 501, 451, 401, 351, 301, 251, 201,
            151, 101, 51, 1,
        ];

        let timesteps = scheduler.timesteps();
        assert_eq!(timesteps.len(), expected_timesteps.len());
        for (i, (actual, expected)) in timesteps.iter().zip(expected_timesteps.iter()).enumerate() {
            assert_eq!(
                *actual, *expected,
                "Timestep mismatch at {}: actual={}, expected={}",
                i, actual, expected
            );
        }

        // Check init_noise_sigma (reference: 1.0)
        assert_eq!(scheduler.init_noise_sigma(), 1.0);

        // Check scale_model_input (reference mean: 1.0 - identity)
        let sample: Tensor<TestBackend, 4> = Tensor::ones([1, 4, 8, 8], &device);
        let timestep = scheduler.timesteps()[0];
        let scaled = scheduler.scale_model_input(sample, timestep);
        let scaled_mean: f32 = scaled.mean().into_scalar();
        assert!(
            (scaled_mean - 1.0).abs() < 1e-6,
            "scale_model_input mean mismatch: actual={}, expected=1.0",
            scaled_mean
        );

        // Check step (reference mean: 1.3104724884033203)
        let model_output: Tensor<TestBackend, 4> = Tensor::zeros([1, 4, 8, 8], &device);
        let sample: Tensor<TestBackend, 4> = Tensor::ones([1, 4, 8, 8], &device);
        let result = scheduler.step(&model_output, timestep, &sample);
        let result_mean: f32 = result.mean().into_scalar();
        assert!(
            (result_mean as f64 - 1.3104724884033203).abs() < 1e-4,
            "step mean mismatch: actual={}, expected=1.3104724884033203",
            result_mean
        );

        // Check add_noise (reference mean: 1.0862191915512085 at timestep 951)
        let original: Tensor<TestBackend, 4> = Tensor::ones([1, 4, 8, 8], &device);
        let noise: Tensor<TestBackend, 4> = Tensor::ones([1, 4, 8, 8], &device);
        let noisy = scheduler.add_noise(&original, noise, 951);
        let noisy_mean: f32 = noisy.mean().into_scalar();
        assert!(
            (noisy_mean as f64 - 1.0862191915512085).abs() < 1e-4,
            "add_noise mean mismatch: actual={}, expected=1.0862191915512085",
            noisy_mean
        );
    }
}
