//! DPM-Solver++ Multistep Scheduler
//!
//! DPM-Solver (and the improved version DPM-Solver++) is a fast dedicated high-order solver
//! for diffusion ODEs with the convergence order guarantee.
//!
//! Based on:
//! - DPM-Solver: <https://arxiv.org/abs/2206.00927>
//! - DPM-Solver++: <https://arxiv.org/abs/2211.01095>

use alloc::vec;
use alloc::vec::Vec;
use burn::tensor::{backend::Backend, Tensor};

#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::Float;

use super::{betas_for_alpha_bar, BetaSchedule, PredictionType};

/// The algorithm type for the solver.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub enum DPMSolverAlgorithmType {
    /// Implements the algorithms defined in <https://arxiv.org/abs/2211.01095>.
    #[default]
    DPMSolverPlusPlus,
    /// Implements the algorithms defined in <https://arxiv.org/abs/2206.00927>.
    DPMSolver,
}

/// The solver type for the second-order solver.
/// The solver type slightly affects the sample quality, especially for
/// small number of steps.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub enum DPMSolverType {
    #[default]
    Midpoint,
    Heun,
}

/// Configuration for the DPM-Solver++ Multistep Scheduler.
#[derive(Debug, Clone)]
pub struct DPMSolverMultistepSchedulerConfig {
    /// The value of beta at the beginning of training.
    pub beta_start: f64,
    /// The value of beta at the end of training.
    pub beta_end: f64,
    /// How beta evolved during training.
    pub beta_schedule: BetaSchedule,
    /// Number of diffusion steps used to train the model.
    pub train_timesteps: usize,
    /// The order of DPM-Solver; can be 1, 2, or 3. We recommend solver_order=2 for guided
    /// sampling, and solver_order=3 for unconditional sampling.
    pub solver_order: usize,
    /// Prediction type of the scheduler function.
    pub prediction_type: PredictionType,
    /// Whether to use the "dynamic thresholding" method (introduced by Imagen).
    /// For pixel-space diffusion models, you can set both `algorithm_type=DPMSolverPlusPlus`
    /// and `thresholding=true` to use dynamic thresholding. Note that thresholding is
    /// unsuitable for latent-space diffusion models (such as stable-diffusion).
    pub thresholding: bool,
    /// The ratio for the dynamic thresholding method. Default is 0.995, same as Imagen.
    pub dynamic_thresholding_ratio: f64,
    /// The threshold value for dynamic thresholding. Valid only when `thresholding: true`
    /// and `algorithm_type: DPMSolverPlusPlus`.
    pub sample_max_value: f64,
    /// The algorithm type for the solver.
    pub algorithm_type: DPMSolverAlgorithmType,
    /// The solver type for the second-order solver.
    pub solver_type: DPMSolverType,
    /// Whether to use lower-order solvers in the final steps. Only valid for < 15 inference
    /// steps. This can stabilize the sampling of DPM-Solver for steps < 15.
    pub lower_order_final: bool,
}

impl Default for DPMSolverMultistepSchedulerConfig {
    fn default() -> Self {
        Self {
            beta_start: 0.00085,
            beta_end: 0.012,
            beta_schedule: BetaSchedule::ScaledLinear,
            train_timesteps: 1000,
            solver_order: 2,
            prediction_type: PredictionType::Epsilon,
            thresholding: false,
            dynamic_thresholding_ratio: 0.995,
            sample_max_value: 1.0,
            algorithm_type: DPMSolverAlgorithmType::DPMSolverPlusPlus,
            solver_type: DPMSolverType::Midpoint,
            lower_order_final: true,
        }
    }
}

/// DPM-Solver++ Multistep Scheduler for diffusion models.
///
/// This scheduler implements DPM-Solver and DPM-Solver++ algorithms for fast
/// sampling in diffusion models.
pub struct DPMSolverMultistepScheduler<B: Backend> {
    alphas_cumprod: Vec<f64>,
    alpha_t: Vec<f64>,
    sigma_t: Vec<f64>,
    lambda_t: Vec<f64>,
    init_noise_sigma: f64,
    lower_order_nums: usize,
    model_outputs: Vec<Option<Tensor<B, 4>>>,
    timesteps: Vec<usize>,
    /// The scheduler configuration.
    pub config: DPMSolverMultistepSchedulerConfig,
}

impl<B: Backend> DPMSolverMultistepScheduler<B> {
    /// Create a new DPM-Solver++ Multistep Scheduler.
    ///
    /// # Arguments
    /// * `inference_steps` - Number of inference steps
    /// * `config` - Scheduler configuration
    /// * `device` - The device to create tensors on
    pub fn new(
        inference_steps: usize,
        config: DPMSolverMultistepSchedulerConfig,
        device: &B::Device,
    ) -> Self {
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

        // alpha_t = sqrt(alphas_cumprod)
        let alpha_t: Vec<f64> = alphas_cumprod.iter().map(|&acp| acp.sqrt()).collect();

        // sigma_t = sqrt(1 - alphas_cumprod)
        let sigma_t: Vec<f64> = alphas_cumprod
            .iter()
            .map(|&acp| (1.0 - acp).sqrt())
            .collect();

        // lambda_t = log(alpha_t) - log(sigma_t)
        let lambda_t: Vec<f64> = alpha_t
            .iter()
            .zip(sigma_t.iter())
            .map(|(&a, &s)| a.ln() - s.ln())
            .collect();

        // timesteps = linspace(train_timesteps - 1, 0, inference_steps + 1), skip first, reverse
        let step = (config.train_timesteps - 1) as f64 / inference_steps as f64;
        let mut timesteps: Vec<usize> = (0..=inference_steps)
            .map(|i| (i as f64 * step).round() as usize)
            .skip(1)
            .collect();
        timesteps.reverse();

        // Create a vector of solver_order None tensors for model outputs
        let model_outputs: Vec<Option<Tensor<B, 4>>> = vec![None; config.solver_order];

        Self {
            alphas_cumprod,
            alpha_t,
            sigma_t,
            lambda_t,
            init_noise_sigma: 1.0,
            lower_order_nums: 0,
            model_outputs,
            timesteps,
            config,
        }
    }

    /// Convert the model output to the corresponding type that the algorithm needs.
    ///
    /// DPM-Solver is designed to discretize an integral of the noise prediction model,
    /// and DPM-Solver++ is designed to discretize an integral of the data prediction model.
    fn convert_model_output(
        &self,
        model_output: &Tensor<B, 4>,
        timestep: usize,
        sample: &Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        match self.config.algorithm_type {
            DPMSolverAlgorithmType::DPMSolverPlusPlus => {
                let x0_pred = match self.config.prediction_type {
                    PredictionType::Epsilon => {
                        let alpha_t = self.alpha_t[timestep];
                        let sigma_t = self.sigma_t[timestep];
                        // (sample - sigma_t * model_output) / alpha_t
                        (sample.clone() - model_output.clone() * sigma_t) / alpha_t
                    }
                    PredictionType::Sample => model_output.clone(),
                    PredictionType::VPrediction => {
                        let alpha_t = self.alpha_t[timestep];
                        let sigma_t = self.sigma_t[timestep];
                        // alpha_t * sample - sigma_t * model_output
                        sample.clone() * alpha_t - model_output.clone() * sigma_t
                    }
                };

                // Note: thresholding is not implemented for burn tensors
                // as it requires quantile operations not available in burn
                if self.config.thresholding {
                    // For now, just return x0_pred without thresholding
                    // In a full implementation, you would need to add quantile support
                    x0_pred
                } else {
                    x0_pred
                }
            }
            DPMSolverAlgorithmType::DPMSolver => match self.config.prediction_type {
                PredictionType::Epsilon => model_output.clone(),
                PredictionType::Sample => {
                    let alpha_t = self.alpha_t[timestep];
                    let sigma_t = self.sigma_t[timestep];
                    // (sample - alpha_t * model_output) / sigma_t
                    (sample.clone() - model_output.clone() * alpha_t) / sigma_t
                }
                PredictionType::VPrediction => {
                    let alpha_t = self.alpha_t[timestep];
                    let sigma_t = self.sigma_t[timestep];
                    // alpha_t * model_output + sigma_t * sample
                    model_output.clone() * alpha_t + sample.clone() * sigma_t
                }
            },
        }
    }

    /// One step for the first-order DPM-Solver (equivalent to DDIM).
    fn dpm_solver_first_order_update(
        &self,
        model_output: Tensor<B, 4>,
        timestep: usize,
        prev_timestep: usize,
        sample: &Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        let (lambda_t, lambda_s) = (self.lambda_t[prev_timestep], self.lambda_t[timestep]);
        let (alpha_t, _alpha_s) = (self.alpha_t[prev_timestep], self.alpha_t[timestep]);
        let (sigma_t, sigma_s) = (self.sigma_t[prev_timestep], self.sigma_t[timestep]);
        let h = lambda_t - lambda_s;

        match self.config.algorithm_type {
            DPMSolverAlgorithmType::DPMSolverPlusPlus => {
                // (sigma_t / sigma_s) * sample - (alpha_t * (exp(-h) - 1)) * model_output
                sample.clone() * (sigma_t / sigma_s) - model_output * (alpha_t * ((-h).exp() - 1.0))
            }
            DPMSolverAlgorithmType::DPMSolver => {
                let alpha_s = self.alpha_t[timestep];
                // (alpha_t / alpha_s) * sample - (sigma_t * (exp(h) - 1)) * model_output
                sample.clone() * (alpha_t / alpha_s) - model_output * (sigma_t * (h.exp() - 1.0))
            }
        }
    }

    /// One step for the second-order multistep DPM-Solver.
    fn multistep_dpm_solver_second_order_update(
        &self,
        model_output_list: &[Option<Tensor<B, 4>>],
        timestep_list: [usize; 2],
        prev_timestep: usize,
        sample: &Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        let (t, s0, s1) = (
            prev_timestep,
            timestep_list[timestep_list.len() - 1],
            timestep_list[timestep_list.len() - 2],
        );

        let m0 = model_output_list[model_output_list.len() - 1]
            .as_ref()
            .unwrap();
        let m1 = model_output_list[model_output_list.len() - 2]
            .as_ref()
            .unwrap();

        let (lambda_t, lambda_s0, lambda_s1) =
            (self.lambda_t[t], self.lambda_t[s0], self.lambda_t[s1]);
        let (alpha_t, alpha_s0) = (self.alpha_t[t], self.alpha_t[s0]);
        let (sigma_t, sigma_s0) = (self.sigma_t[t], self.sigma_t[s0]);
        let (h, h_0) = (lambda_t - lambda_s0, lambda_s0 - lambda_s1);
        let r0 = h_0 / h;
        let d0 = m0;
        let d1 = (m0.clone() - m1.clone()) * (1.0 / r0);

        match self.config.algorithm_type {
            DPMSolverAlgorithmType::DPMSolverPlusPlus => match self.config.solver_type {
                DPMSolverType::Midpoint => {
                    // (sigma_t / sigma_s0) * sample
                    // - (alpha_t * (exp(-h) - 1)) * d0
                    // - 0.5 * (alpha_t * (exp(-h) - 1)) * d1
                    let coeff = alpha_t * ((-h).exp() - 1.0);
                    sample.clone() * (sigma_t / sigma_s0) - d0.clone() * coeff - d1 * (0.5 * coeff)
                }
                DPMSolverType::Heun => {
                    // (sigma_t / sigma_s0) * sample
                    // - (alpha_t * (exp(-h) - 1)) * d0
                    // + (alpha_t * ((exp(-h) - 1) / h + 1)) * d1
                    let exp_neg_h = (-h).exp();
                    sample.clone() * (sigma_t / sigma_s0)
                        - d0.clone() * (alpha_t * (exp_neg_h - 1.0))
                        + d1 * (alpha_t * ((exp_neg_h - 1.0) / h + 1.0))
                }
            },
            DPMSolverAlgorithmType::DPMSolver => match self.config.solver_type {
                DPMSolverType::Midpoint => {
                    let coeff = sigma_t * (h.exp() - 1.0);
                    sample.clone() * (alpha_t / alpha_s0) - d0.clone() * coeff - d1 * (0.5 * coeff)
                }
                DPMSolverType::Heun => {
                    let exp_h = h.exp();
                    sample.clone() * (alpha_t / alpha_s0)
                        - d0.clone() * (sigma_t * (exp_h - 1.0))
                        - d1 * (sigma_t * ((exp_h - 1.0) / h - 1.0))
                }
            },
        }
    }

    /// One step for the third-order multistep DPM-Solver.
    fn multistep_dpm_solver_third_order_update(
        &self,
        model_output_list: &[Option<Tensor<B, 4>>],
        timestep_list: [usize; 3],
        prev_timestep: usize,
        sample: &Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        let (t, s0, s1, s2) = (
            prev_timestep,
            timestep_list[timestep_list.len() - 1],
            timestep_list[timestep_list.len() - 2],
            timestep_list[timestep_list.len() - 3],
        );

        let m0 = model_output_list[model_output_list.len() - 1]
            .as_ref()
            .unwrap();
        let m1 = model_output_list[model_output_list.len() - 2]
            .as_ref()
            .unwrap();
        let m2 = model_output_list[model_output_list.len() - 3]
            .as_ref()
            .unwrap();

        let (lambda_t, lambda_s0, lambda_s1, lambda_s2) = (
            self.lambda_t[t],
            self.lambda_t[s0],
            self.lambda_t[s1],
            self.lambda_t[s2],
        );
        let (alpha_t, alpha_s0) = (self.alpha_t[t], self.alpha_t[s0]);
        let (sigma_t, sigma_s0) = (self.sigma_t[t], self.sigma_t[s0]);
        let (h, h_0, h_1) = (
            lambda_t - lambda_s0,
            lambda_s0 - lambda_s1,
            lambda_s1 - lambda_s2,
        );
        let (r0, r1) = (h_0 / h, h_1 / h);

        let d0 = m0;
        let d1_0 = (m0.clone() - m1.clone()) * (1.0 / r0);
        let d1_1 = (m1.clone() - m2.clone()) * (1.0 / r1);
        let d1 = d1_0.clone() + (d1_0.clone() - d1_1) * (r0 / (r0 + r1));
        let d2 = (d1_0.clone() - (m1.clone() - m2.clone()) * (1.0 / r1)) * (1.0 / (r0 + r1));

        match self.config.algorithm_type {
            DPMSolverAlgorithmType::DPMSolverPlusPlus => {
                let exp_neg_h = (-h).exp();
                sample.clone() * (sigma_t / sigma_s0) - d0.clone() * (alpha_t * (exp_neg_h - 1.0))
                    + d1 * (alpha_t * ((exp_neg_h - 1.0) / h + 1.0))
                    - d2 * (alpha_t * ((exp_neg_h - 1.0 + h) / h.powi(2) - 0.5))
            }
            DPMSolverAlgorithmType::DPMSolver => {
                let exp_h = h.exp();
                sample.clone() * (alpha_t / alpha_s0)
                    - d0.clone() * (sigma_t * (exp_h - 1.0))
                    - d1 * (sigma_t * ((exp_h - 1.0) / h - 1.0))
                    - d2 * (sigma_t * ((exp_h - 1.0 - h) / h.powi(2) - 0.5))
            }
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

    /// Scale the model input (no scaling needed for DPM-Solver).
    pub fn scale_model_input(&self, sample: Tensor<B, 4>, _timestep: usize) -> Tensor<B, 4> {
        sample
    }

    /// Perform one step of the DPM-Solver.
    pub fn step(
        &mut self,
        model_output: &Tensor<B, 4>,
        timestep: usize,
        sample: &Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        let step_index = self.timesteps.iter().position(|&t| t == timestep).unwrap();

        let prev_timestep = if step_index == self.timesteps.len() - 1 {
            0
        } else {
            self.timesteps[step_index + 1]
        };

        let lower_order_final = (step_index == self.timesteps.len() - 1)
            && self.config.lower_order_final
            && self.timesteps.len() < 15;
        let lower_order_second = (step_index == self.timesteps.len() - 2)
            && self.config.lower_order_final
            && self.timesteps.len() < 15;

        let model_output = self.convert_model_output(model_output, timestep, sample);

        // Shift model outputs
        for i in 0..self.config.solver_order - 1 {
            self.model_outputs[i] = self.model_outputs[i + 1].take();
        }
        // Store the latest model output
        let m = self.model_outputs.len();
        self.model_outputs[m - 1] = Some(model_output.clone());

        let prev_sample = if self.config.solver_order == 1
            || self.lower_order_nums < 1
            || lower_order_final
        {
            self.dpm_solver_first_order_update(model_output, timestep, prev_timestep, sample)
        } else if self.config.solver_order == 2 || self.lower_order_nums < 2 || lower_order_second {
            let timestep_list = [self.timesteps[step_index - 1], timestep];
            self.multistep_dpm_solver_second_order_update(
                &self.model_outputs,
                timestep_list,
                prev_timestep,
                sample,
            )
        } else {
            let timestep_list = [
                self.timesteps[step_index - 2],
                self.timesteps[step_index - 1],
                timestep,
            ];
            self.multistep_dpm_solver_third_order_update(
                &self.model_outputs,
                timestep_list,
                prev_timestep,
                sample,
            )
        };

        if self.lower_order_nums < self.config.solver_order {
            self.lower_order_nums += 1;
        }

        prev_sample
    }

    /// Add noise to original samples.
    pub fn add_noise(
        &self,
        original_samples: &Tensor<B, 4>,
        noise: Tensor<B, 4>,
        timestep: usize,
    ) -> Tensor<B, 4> {
        let sqrt_alpha_cumprod = self.alphas_cumprod[timestep].sqrt();
        let sqrt_one_minus_alpha_cumprod = (1.0 - self.alphas_cumprod[timestep]).sqrt();

        original_samples.clone() * sqrt_alpha_cumprod + noise * sqrt_one_minus_alpha_cumprod
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestBackend;
    use burn::tensor::Shape;

    #[test]
    fn test_dpmsolver_multistep_scheduler_creation() {
        let device = Default::default();
        let config = DPMSolverMultistepSchedulerConfig::default();
        let scheduler = DPMSolverMultistepScheduler::<TestBackend>::new(20, config, &device);

        assert_eq!(scheduler.timesteps().len(), 20);
        assert_eq!(scheduler.init_noise_sigma(), 1.0);
    }

    #[test]
    fn test_dpmsolver_timesteps() {
        let device = Default::default();
        let config = DPMSolverMultistepSchedulerConfig::default();
        let scheduler = DPMSolverMultistepScheduler::<TestBackend>::new(20, config, &device);

        let timesteps = scheduler.timesteps();
        // First timestep should be close to train_timesteps - 1
        assert!(timesteps[0] > 900);
        // Last timestep should be small
        assert!(timesteps[timesteps.len() - 1] < 100);

        // Should be monotonically decreasing
        for i in 1..timesteps.len() {
            assert!(timesteps[i] < timesteps[i - 1]);
        }
    }

    #[test]
    fn test_dpmsolver_scale_model_input() {
        let device = Default::default();
        let config = DPMSolverMultistepSchedulerConfig::default();
        let scheduler = DPMSolverMultistepScheduler::<TestBackend>::new(20, config, &device);

        let sample: Tensor<TestBackend, 4> = Tensor::ones([1, 4, 8, 8], &device);
        let timestep = scheduler.timesteps()[0];

        // DPM-Solver doesn't scale input
        let scaled = scheduler.scale_model_input(sample.clone(), timestep);
        let diff: f32 = (scaled - sample).abs().mean().into_scalar();
        assert!(diff < 1e-6);
    }

    #[test]
    fn test_dpmsolver_step() {
        let device = Default::default();
        let config = DPMSolverMultistepSchedulerConfig::default();
        let mut scheduler = DPMSolverMultistepScheduler::<TestBackend>::new(20, config, &device);

        let model_output: Tensor<TestBackend, 4> = Tensor::zeros([1, 4, 8, 8], &device);
        let sample: Tensor<TestBackend, 4> = Tensor::ones([1, 4, 8, 8], &device);
        let timestep = scheduler.timesteps()[0];

        let result = scheduler.step(&model_output, timestep, &sample);
        assert_eq!(result.shape(), Shape::from([1, 4, 8, 8]));
    }

    #[test]
    fn test_dpmsolver_add_noise() {
        let device = Default::default();
        let config = DPMSolverMultistepSchedulerConfig::default();
        let scheduler = DPMSolverMultistepScheduler::<TestBackend>::new(20, config, &device);

        let original: Tensor<TestBackend, 4> = Tensor::ones([1, 4, 8, 8], &device);
        let noise: Tensor<TestBackend, 4> = Tensor::zeros([1, 4, 8, 8], &device);
        let timestep = scheduler.timesteps()[0];

        let noisy = scheduler.add_noise(&original, noise, timestep);
        assert_eq!(noisy.shape(), Shape::from([1, 4, 8, 8]));
    }

    #[test]
    fn test_dpmsolver_multiple_steps() {
        let device = Default::default();
        let config = DPMSolverMultistepSchedulerConfig {
            solver_order: 2,
            ..Default::default()
        };
        let mut scheduler = DPMSolverMultistepScheduler::<TestBackend>::new(20, config, &device);

        let mut sample: Tensor<TestBackend, 4> = Tensor::ones([1, 4, 8, 8], &device);

        // Run multiple steps to test the multistep logic
        for i in 0..5 {
            let timestep = scheduler.timesteps()[i];
            let model_output: Tensor<TestBackend, 4> = Tensor::zeros([1, 4, 8, 8], &device);
            sample = scheduler.step(&model_output, timestep, &sample);
        }

        // After multiple steps with zero model output, sample should still be finite
        let sample_data = sample.into_data();
        let values: Vec<f32> = sample_data.to_vec().unwrap();
        for v in &values {
            assert!(v.is_finite(), "Sample contains non-finite values");
        }
    }

    /// Test DPM-Solver++ values match diffusers-rs
    #[test]
    fn test_dpmsolver_matches_diffusers_rs() {
        let device = Default::default();
        let config = DPMSolverMultistepSchedulerConfig::default();
        let scheduler = DPMSolverMultistepScheduler::<TestBackend>::new(20, config, &device);

        // Reference values from diffusers-rs
        // step = 999/20 = 49.95
        // Timesteps computed as: (i * step).round() for i in 1..=20, then reversed
        let expected_timesteps = [
            999, 949, 899, 849, 799, 749, 699, 649, 599, 549, 500, 450, 400, 350, 300, 250, 200,
            150, 100, 50,
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

        // Check init_noise_sigma
        assert_eq!(scheduler.init_noise_sigma(), 1.0);

        // Check alphas_cumprod at key positions
        // Reference: diffusers-rs alphas_cumprod[0] ≈ 0.9991
        assert!(
            (scheduler.alphas_cumprod[0] - 0.9991499800572107).abs() < 1e-4,
            "alphas_cumprod[0] mismatch: {}",
            scheduler.alphas_cumprod[0]
        );

        // Reference: diffusers-rs alphas_cumprod[999] ≈ 0.0047
        assert!(
            (scheduler.alphas_cumprod[999] - 0.004660095977824908).abs() < 1e-4,
            "alphas_cumprod[999] mismatch: {}",
            scheduler.alphas_cumprod[999]
        );
    }

    /// Test DPM-Solver++ step matches diffusers-rs reference values
    #[test]
    fn test_dpmsolver_step_matches_diffusers_rs() {
        let device = Default::default();
        let config = DPMSolverMultistepSchedulerConfig::default();
        let mut scheduler = DPMSolverMultistepScheduler::<TestBackend>::new(20, config, &device);

        // Use simple known inputs
        let sample: Tensor<TestBackend, 4> = Tensor::ones([1, 4, 8, 8], &device);
        let model_output: Tensor<TestBackend, 4> = Tensor::zeros([1, 4, 8, 8], &device);

        // First step at timestep 999
        let timestep = scheduler.timesteps()[0];
        let result = scheduler.step(&model_output, timestep, &sample);
        let result_mean: f32 = result.mean().into_scalar();

        // Reference from diffusers-rs: step(zeros, timestep=999, ones): mean=1.3377922773361206
        assert!(
            (result_mean as f64 - 1.3377922773361206).abs() < 1e-4,
            "Step mean mismatch: actual={}, expected=1.3377922773361206",
            result_mean
        );
    }

    /// Test DPM-Solver++ add_noise matches diffusers-rs reference values
    #[test]
    fn test_dpmsolver_add_noise_matches_diffusers_rs() {
        let device = Default::default();
        let config = DPMSolverMultistepSchedulerConfig::default();
        let scheduler = DPMSolverMultistepScheduler::<TestBackend>::new(20, config, &device);

        let original: Tensor<TestBackend, 4> = Tensor::ones([1, 4, 8, 8], &device);
        let noise: Tensor<TestBackend, 4> = Tensor::ones([1, 4, 8, 8], &device);
        let timestep = scheduler.timesteps()[0]; // 999

        let noisy = scheduler.add_noise(&original, noise, timestep);
        let noisy_mean: f32 = noisy.mean().into_scalar();

        // Reference from diffusers-rs: add_noise(ones, ones, timestep=999): mean=1.0659321546554565
        assert!(
            (noisy_mean as f64 - 1.0659321546554565).abs() < 1e-4,
            "add_noise mean mismatch: actual={}, expected=1.0659321546554565",
            noisy_mean
        );
    }
}
