//! Attention Based Building Blocks

use alloc::vec;
use alloc::vec::Vec;
use burn::config::Config;
use burn::module::Module;
use burn::nn::{
    self, Dropout, DropoutConfig, GroupNorm, GroupNormConfig, LayerNorm, LayerNormConfig,
    LinearConfig,
};
use burn::tensor::activation::{gelu, softmax};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::Float;

#[derive(Config, Debug)]
pub struct GeGluConfig {
    /// The size of the input features.
    d_input: usize,
    /// The size of the output features.
    d_output: usize,
}

/// A variant of the gated linear unit activation function from https://arxiv.org/abs/2002.05202.
#[derive(Module, Debug)]
struct GeGlu<B: Backend> {
    proj: nn::Linear<B>,
}

impl GeGluConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> GeGlu<B> {
        let proj = LinearConfig::new(self.d_input, 2 * self.d_output).init(device);
        GeGlu { proj }
    }
}

impl<B: Backend> GeGlu<B> {
    fn forward<const D: usize>(&self, xs: Tensor<B, D>) -> Tensor<B, D> {
        let hidden_states_and_gate = self.proj.forward(xs).chunk(2, D - 1);
        hidden_states_and_gate[0].clone() * gelu(hidden_states_and_gate[1].clone())
    }
}

#[derive(Config, Debug)]
pub struct FeedForwardConfig {
    /// The size of the input features.
    pub d_input: usize,
    /// The size of the output features. If not given defaults to `d_input`.
    d_output: Option<usize>,
    /// The multiplier to use for the hidden dimension.
    #[config(default = 4)]
    multiplier: usize,
    /// The dropout probability. Default: 0.0
    #[config(default = 0.)]
    dropout: f64,
}

#[derive(Module, Debug)]
pub struct FeedForward<B: Backend> {
    geglu: GeGlu<B>,
    dropout: Dropout,
    linear_outer: nn::Linear<B>,
}

impl FeedForwardConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> FeedForward<B> {
        let inner_dim = self.d_input * self.multiplier;
        let dim_out = self.d_output.unwrap_or(self.d_input);

        FeedForward {
            geglu: GeGluConfig {
                d_input: self.d_input,
                d_output: inner_dim,
            }
            .init(device),
            linear_outer: LinearConfig::new(inner_dim, dim_out).init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
}

impl<B: Backend> FeedForward<B> {
    pub fn forward<const D: usize>(&self, xs: Tensor<B, D>) -> Tensor<B, D> {
        let x = self.geglu.forward(xs);
        let x = self.dropout.forward(x);
        self.linear_outer.forward(x)
    }
}

#[derive(Config, Debug)]
pub struct CrossAttentionConfig {
    /// The number of channels in the query.
    d_query: usize,
    /// The number of channels in the context. If not given defaults to `query_dim`.
    d_context: Option<usize>,
    /// The number of heads to use for the multi-head attention.
    #[config(default = 8)]
    n_heads: usize,
    /// The number of channels in each head.
    #[config(default = 64)]
    d_head: usize,
    /// The size of the slices to use for the multi-head attention.
    slice_size: Option<usize>,
    #[config(default = 0.)]
    // The dropout probability.
    dropout: f64,
}

#[derive(Module, Debug)]
pub struct CrossAttention<B: Backend> {
    query: nn::Linear<B>,
    key: nn::Linear<B>,
    value: nn::Linear<B>,
    output: nn::Linear<B>,
    n_heads: usize,
    scale: f64,
    slice_size: Option<usize>,
}

impl CrossAttentionConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> CrossAttention<B> {
        let inner_dim = self.d_head * self.n_heads;
        let context_dim = self.d_context.unwrap_or(self.d_query);
        let scale = 1. / (self.d_head as f64).sqrt();

        CrossAttention {
            query: LinearConfig::new(self.d_query, inner_dim)
                .with_bias(false)
                .init(device),
            key: LinearConfig::new(context_dim, inner_dim)
                .with_bias(false)
                .init(device),
            value: LinearConfig::new(context_dim, inner_dim)
                .with_bias(false)
                .init(device),
            output: LinearConfig::new(inner_dim, self.d_query).init(device),
            n_heads: self.n_heads,
            scale,
            slice_size: self.slice_size,
        }
    }
}

impl<B: Backend> CrossAttention<B> {
    fn reshape_heads_to_batch_dim(&self, xs: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch_size, seq_len, dim] = xs.dims();
        xs.reshape([batch_size, seq_len, self.n_heads, dim / self.n_heads])
            .swap_dims(1, 2)
            .reshape([batch_size * self.n_heads, seq_len, dim / self.n_heads])
    }

    fn reshape_batch_dim_to_heads(&self, xs: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch_size, seq_len, dim] = xs.dims();
        xs.reshape([batch_size / self.n_heads, self.n_heads, seq_len, dim])
            .swap_dims(1, 2)
            .reshape([batch_size / self.n_heads, seq_len, dim * self.n_heads])
    }

    fn sliced_attention(
        &self,
        query: Tensor<B, 3>,
        key: Tensor<B, 3>,
        value: Tensor<B, 3>,
        slice_size: usize,
    ) -> Tensor<B, 3> {
        let batch_size_attention = query.clone().shape().dims[0];
        let mut hidden_states = Vec::with_capacity(batch_size_attention / slice_size);

        for i in 0..batch_size_attention / slice_size {
            let start_idx = i * slice_size;
            let end_idx = (i + 1) * slice_size;

            let xs = query
                .clone()
                .slice([start_idx..end_idx, 0..query.shape().dims[1]])
                .matmul(
                    key.clone()
                        .slice([start_idx..end_idx, 0..key.shape().dims[1]])
                        .swap_dims(3 - 1, 3 - 2)
                        * self.scale,
                );

            let xs = softmax(xs, 3 - 1).matmul(
                value
                    .clone()
                    .slice([start_idx..end_idx, 0..value.shape().dims[1]]),
            );

            hidden_states.push(xs);
        }

        let output = Tensor::cat(hidden_states, 0);
        self.reshape_batch_dim_to_heads(output)
    }

    fn attention(
        &self,
        query: Tensor<B, 3>,
        key: Tensor<B, 3>,
        value: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        let xs = query.matmul(key.swap_dims(3 - 1, 3 - 2) * self.scale);
        let xs = softmax(xs, 3 - 1).matmul(value);

        self.reshape_batch_dim_to_heads(xs)
    }

    pub fn forward(&self, xs: Tensor<B, 3>, context: Option<Tensor<B, 3>>) -> Tensor<B, 3> {
        let query = self.query.forward(xs.clone());
        let context = context.unwrap_or(xs);
        let key = self.key.forward(context.clone());
        let value = self.value.forward(context);

        let query = self.reshape_heads_to_batch_dim(query);
        let key = self.reshape_heads_to_batch_dim(key);
        let value = self.reshape_heads_to_batch_dim(value);

        let output_tensor = match self.slice_size {
            None => self.attention(query, key, value),
            Some(slice_size) if query.shape().dims[0] / slice_size <= 1 => {
                self.attention(query, key, value)
            }
            Some(slice_size) => self.sliced_attention(query, key, value, slice_size),
        };

        self.output.forward(output_tensor)
    }
}

#[derive(Config, Debug)]
pub struct BasicTransformerBlockConfig {
    d_model: usize,
    d_context: Option<usize>,
    n_heads: usize,
    d_head: usize,
    sliced_attn_size: Option<usize>,
}

/// A basic Transformer block.
#[derive(Module, Debug)]
pub struct BasicTransformerBlock<B: Backend> {
    attn1: CrossAttention<B>,
    ff: FeedForward<B>,
    attn2: CrossAttention<B>,
    norm1: LayerNorm<B>,
    norm2: LayerNorm<B>,
    norm3: LayerNorm<B>,
}

impl BasicTransformerBlockConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> BasicTransformerBlock<B> {
        let attn1 = CrossAttentionConfig::new(self.d_model)
            .with_n_heads(self.n_heads)
            .with_d_head(self.d_head)
            .with_slice_size(self.sliced_attn_size)
            .init(device);
        let ff = FeedForwardConfig::new(self.d_model).init(device);
        let attn2 = CrossAttentionConfig::new(self.d_model)
            .with_d_context(self.d_context)
            .with_n_heads(self.n_heads)
            .with_d_head(self.d_head)
            .with_slice_size(self.sliced_attn_size)
            .init(device);
        let norm1 = LayerNormConfig::new(self.d_model).init(device);
        let norm2 = LayerNormConfig::new(self.d_model).init(device);
        let norm3 = LayerNormConfig::new(self.d_model).init(device);

        BasicTransformerBlock {
            attn1,
            ff,
            attn2,
            norm1,
            norm2,
            norm3,
        }
    }
}

impl<B: Backend> BasicTransformerBlock<B> {
    pub fn forward(&self, xs: Tensor<B, 3>, context: Option<Tensor<B, 3>>) -> Tensor<B, 3> {
        let xs = self.attn1.forward(self.norm1.forward(xs.clone()), None) + xs;
        let xs = self.attn2.forward(self.norm2.forward(xs.clone()), context) + xs;
        self.ff.forward(self.norm3.forward(xs.clone())) + xs
    }
}

#[derive(Config, Debug)]
pub struct SpatialTransformerConfig {
    #[config(default = 1)]
    pub depth: usize,
    #[config(default = 32)]
    pub n_groups: usize,
    pub d_context: Option<usize>,
    pub sliced_attn_size: Option<usize>,
    #[config(default = false)]
    pub use_linear_projection: bool,
    pub in_channels: usize,
    pub n_heads: usize,
    pub d_head: usize,
}

#[derive(Module, Debug)]
enum Proj<B: Backend> {
    Conv2d(nn::conv::Conv2d<B>),
    Linear(nn::Linear<B>),
}

impl<B: Backend> Proj<B> {
    fn forward(&self, xs: Tensor<B, 4>) -> Tensor<B, 4> {
        match self {
            Proj::Conv2d(conv) => conv.forward(xs),
            Proj::Linear(linear) => {
                // For linear projection, we need to permute from [batch, channels, h, w]
                // to [batch, h, w, channels], apply linear, then permute back
                let xs = xs.swap_dims(1, 2).swap_dims(2, 3); // [batch, h, w, channels]
                let xs = linear.forward(xs);
                xs.swap_dims(2, 3).swap_dims(1, 2) // [batch, channels, h, w]
            }
        }
    }
}

/// Aka Transformer2DModel
#[derive(Module, Debug)]
pub struct SpatialTransformer<B: Backend> {
    norm: GroupNorm<B>,
    proj_in: Proj<B>,
    transformer_blocks: Vec<BasicTransformerBlock<B>>,
    proj_out: Proj<B>,
}

impl SpatialTransformerConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> SpatialTransformer<B> {
        let d_inner = self.n_heads * self.d_head;
        let norm = GroupNormConfig::new(self.n_groups, self.in_channels)
            .with_epsilon(1e-6)
            .init(device);
        let proj_in = if self.use_linear_projection {
            Proj::Linear(nn::LinearConfig::new(self.in_channels, d_inner).init(device))
        } else {
            Proj::Conv2d(
                nn::conv::Conv2dConfig::new([self.in_channels, d_inner], [1, 1]).init(device),
            )
        };

        let mut transformer_blocks = vec![];
        for _index in 0..self.depth {
            let tb = BasicTransformerBlockConfig::new(d_inner, self.n_heads, self.d_head)
                .with_d_context(self.d_context)
                .with_sliced_attn_size(self.sliced_attn_size)
                .init(device);

            transformer_blocks.push(tb)
        }

        let proj_out = if self.use_linear_projection {
            Proj::Linear(nn::LinearConfig::new(d_inner, self.in_channels).init(device))
        } else {
            Proj::Conv2d(
                nn::conv::Conv2dConfig::new([d_inner, self.in_channels], [1, 1]).init(device),
            )
        };

        SpatialTransformer {
            norm,
            proj_in,
            transformer_blocks,
            proj_out,
        }
    }
}

impl<B: Backend> SpatialTransformer<B> {
    pub fn forward(&self, xs: Tensor<B, 4>, context: Option<Tensor<B, 3>>) -> Tensor<B, 4> {
        let [n_batch, _n_channel, height, weight] = xs.dims();

        let residual = xs.clone();
        let xs = self.norm.forward(xs);
        let xs = self.proj_in.forward(xs);
        let d_inner = xs.shape().dims[1];
        let xs = xs
            .swap_dims(1, 2)
            .transpose()
            .reshape([n_batch, height * weight, d_inner]);

        let mut xs = xs;
        for block in self.transformer_blocks.iter() {
            xs = block.forward(xs, context.clone())
        }

        let xs = xs
            .reshape([n_batch, height, weight, d_inner])
            .transpose()
            .swap_dims(1, 2);

        self.proj_out.forward(xs) + residual
    }
}

#[derive(Config, Debug)]
pub struct AttentionBlockConfig {
    pub channels: usize,
    pub n_head_channels: Option<usize>,
    #[config(default = 32)]
    pub n_groups: usize,
    #[config(default = 1.)]
    pub rescale_output_factor: f64,
    #[config(default = 1e-5)]
    pub eps: f64,
}

#[derive(Module, Debug)]
pub struct AttentionBlock<B: Backend> {
    group_norm: nn::GroupNorm<B>,
    query: nn::Linear<B>,
    key: nn::Linear<B>,
    value: nn::Linear<B>,
    proj_attn: nn::Linear<B>,
    channels: usize,
    n_heads: usize,
    rescale_output_factor: f64,
}

impl AttentionBlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> AttentionBlock<B> {
        let n_head_channels = self.n_head_channels.unwrap_or(self.channels);
        let n_heads = self.channels / n_head_channels;
        let group_norm = GroupNormConfig::new(self.n_groups, self.channels)
            .with_epsilon(self.eps)
            .init(device);
        let query = LinearConfig::new(self.channels, self.channels).init(device);
        let key = LinearConfig::new(self.channels, self.channels).init(device);
        let value = LinearConfig::new(self.channels, self.channels).init(device);
        let proj_attn = LinearConfig::new(self.channels, self.channels).init(device);

        AttentionBlock {
            group_norm,
            query,
            key,
            value,
            proj_attn,
            channels: self.channels,
            n_heads,
            rescale_output_factor: self.rescale_output_factor,
        }
    }
}

impl<B: Backend> AttentionBlock<B> {
    fn transpose_for_scores(&self, xs: Tensor<B, 3>) -> Tensor<B, 4> {
        let [n_batch, t, h_times_d] = xs.dims();
        xs.reshape([n_batch, t, self.n_heads, h_times_d / self.n_heads])
            .swap_dims(1, 2)
    }

    pub fn forward(&self, xs: Tensor<B, 4>) -> Tensor<B, 4> {
        let residual = xs.clone();
        let [n_batch, channel, height, width] = xs.dims();
        let xs = self
            .group_norm
            .forward(xs)
            .reshape([n_batch, channel, height * width])
            .swap_dims(1, 2);

        let query_proj = self.query.forward(xs.clone());
        let key_proj = self.key.forward(xs.clone());
        let value_proj = self.value.forward(xs.clone());

        let query_states = self.transpose_for_scores(query_proj);
        let key_states = self.transpose_for_scores(key_proj);
        let value_states = self.transpose_for_scores(value_proj);

        // scale is applied twice, hence the -0.25 here rather than -0.5.
        // https://github.com/huggingface/diffusers/blob/d3d22ce5a894becb951eec03e663951b28d45135/src/diffusers/models/attention.py#L87
        let scale = f64::powf(self.channels as f64 / self.n_heads as f64, -0.25);
        let attention_scores = (query_states * scale).matmul(key_states.transpose() * scale);
        let attention_probs = softmax(attention_scores, 4 - 1);

        let xs = attention_probs.matmul(value_states);
        let xs = xs.swap_dims(1, 2);
        let xs: Tensor<B, 3> = xs.flatten(4 - 2, 4 - 1);
        let xs = self
            .proj_attn
            .forward(xs)
            .transpose()
            .reshape([n_batch, channel, height, width]);

        (xs + residual) / self.rescale_output_factor
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestBackend;
    use burn::module::{Param, ParamId};
    use burn::tensor::{Shape, TensorData, Tolerance};

    #[test]
    fn test_geglu_tensor_shape_3() {
        let device = Default::default();
        let weight = Tensor::from_data(
            TensorData::from([
                [
                    0.1221, 2.0378, -0.1171, 1.3004, -0.9630, -0.3108, -1.3376, -1.0593,
                ],
                [
                    0.4669, -0.8146, 0.9965, -0.4659, 2.0444, -0.0709, -0.0147, 0.2135,
                ],
            ]),
            &device,
        );
        let bias = Tensor::from_data(
            TensorData::from([
                0.2867778149426027,
                0.6646517317105776,
                0.023946332404821136,
                -0.1395737454364393,
                0.05131041098737321,
                -0.4225726694675192,
                0.036411720220954735,
                0.01829268669677364,
            ]),
            &device,
        );

        let geglu = GeGlu {
            proj: nn::Linear {
                weight: Param::initialized(ParamId::new(), weight),
                bias: Some(Param::initialized(ParamId::new(), bias)),
            },
        };

        let tensor: Tensor<TestBackend, 3> = Tensor::from_data(
            TensorData::from([
                [[1., 2.], [3., 4.], [5., 6.]],
                [[7., 8.], [9., 10.], [11., 12.]],
            ]),
            &device,
        );

        let output = geglu.forward(tensor);
        assert_eq!(output.shape(), Shape::from([2, 3, 4]));
        output.into_data().assert_approx_eq::<f32>(
            &TensorData::from([
                [
                    [4.2632e0, -1.7927e-1, -2.3216e-1, -3.7916e-2],
                    [1.3460e1, -2.9266e-1, -2.1707e-4, -4.5595e-2],
                    [2.7750e1, -1.1442e-1, -2.5335e-13, -2.5403e-4],
                ],
                [
                    [4.7135e1, -1.7708e-2, -0.0000e0, -6.7097e-9],
                    [7.1616e1, -1.0652e-3, -0.0000e0, -0.0000e0],
                    [1.0119e2, -2.1943e-5, -0.0000e0, -0.0000e0],
                ],
            ]),
            Tolerance::rel_abs(1e-2, 1e-2),
        );
    }

    #[test]
    fn test_geglu_tensor_shape_2() {
        let device = Default::default();
        let weight = Tensor::from_data(
            TensorData::from([
                [0.6054, 1.9322, 0.1445, 1.3004, -0.6853, -0.8947],
                [-0.3678, 0.4081, -1.9001, -1.5843, -0.9399, 0.1018],
            ]),
            &device,
        );
        let bias = Tensor::from_data(
            TensorData::from([
                0.3237631905393836,
                0.22052049807936902,
                -0.3196353346822061,
                -0.02244043444199162,
                -0.33600250665852865,
                0.5259391939301621,
            ]),
            &device,
        );

        let geglu = GeGlu {
            proj: nn::Linear {
                weight: Param::initialized(ParamId::new(), weight),
                bias: Some(Param::initialized(ParamId::new(), bias)),
            },
        };

        let tensor: Tensor<TestBackend, 2> =
            Tensor::from_data(TensorData::from([[1., 2.], [3., 4.], [5., 6.]]), &device);

        let output = geglu.forward(tensor);
        assert_eq!(output.shape(), Shape::from([3, 3]));
        output.into_data().assert_approx_eq::<f32>(
            &TensorData::from([
                [-2.4192e-5, -3.3057e-2, 2.8535e-1],
                [-0.0000e0, -2.0983e-7, 5.2465e-1],
                [-0.0000e0, -0.0000e0, 1.2599e-2],
            ]),
            Tolerance::rel_abs(1e-1, 1e-1),
        );
    }

    #[test]
    fn test_sliced_attention() {
        let device = Default::default();
        // create tensor of size [2, 4, 2]
        let query: Tensor<TestBackend, 3> = Tensor::from_data(
            TensorData::from([
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
                [[9.0, 10.0], [11.0, 12.0], [13.0, 14.0], [15.0, 16.0]],
                [[17.0, 18.0], [19.0, 20.0], [21.0, 22.0], [23.0, 24.0]],
                [[25.0, 26.0], [27.0, 28.0], [29.0, 30.0], [31.0, 32.0]],
            ]),
            &device,
        );
        let key: Tensor<TestBackend, 3> = Tensor::from_data(
            TensorData::from([
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
                [[9.0, 10.0], [11.0, 12.0], [13.0, 14.0], [15.0, 16.0]],
                [[17.0, 18.0], [19.0, 20.0], [21.0, 22.0], [23.0, 24.0]],
                [[25.0, 26.0], [27.0, 28.0], [29.0, 30.0], [31.0, 32.0]],
            ]),
            &device,
        );
        let value: Tensor<TestBackend, 3> = Tensor::from_data(
            TensorData::from([
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
                [[9.0, 10.0], [11.0, 12.0], [13.0, 14.0], [15.0, 16.0]],
                [[17.0, 18.0], [19.0, 20.0], [21.0, 22.0], [23.0, 24.0]],
                [[25.0, 26.0], [27.0, 28.0], [29.0, 30.0], [31.0, 32.0]],
            ]),
            &device,
        );

        let cross_attention = CrossAttentionConfig::new(320)
            .with_n_heads(2)
            .with_d_head(40)
            .with_slice_size(Some(2))
            .init(&device);

        let output = cross_attention.sliced_attention(query, key, value, 2);

        assert_eq!(output.shape(), Shape::from([2, 4, 4]));
        output.into_data().assert_approx_eq::<f32>(
            &TensorData::from([
                [
                    [5.9201, 6.9201, 14.9951, 15.9951],
                    [6.7557, 7.7557, 14.9986, 15.9986],
                    [6.9363, 7.9363, 14.9996, 15.9996],
                    [6.9824, 7.9824, 14.9999, 15.9999],
                ],
                [
                    [23.0000, 24.0000, 31.0000, 32.0000],
                    [23.0000, 24.0000, 31.0000, 32.0000],
                    [23.0000, 24.0000, 31.0000, 32.0000],
                    [23.0000, 24.0000, 31.0000, 32.0000],
                ],
            ]),
            Tolerance::rel_abs(1e-3, 1e-3),
        )
    }

    /// Test GELU activation matches diffusers-rs (tch gelu("none"))
    /// Reference values from diffusers-rs v0.3.1
    #[test]
    fn test_gelu_matches_diffusers_rs() {
        let device = Default::default();
        let xs: Tensor<TestBackend, 1> = Tensor::from_data(
            TensorData::from([-2.0f32, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]),
            &device,
        );

        let result = gelu(xs);

        // Reference values from diffusers-rs: gelu("none")
        // [-0.04550028, -0.15865526, -0.15426877, 0.0, 0.34573123, 0.8413447, 1.9544997]
        result.into_data().assert_approx_eq::<f32>(
            &TensorData::from([
                -0.04550028,
                -0.15865526,
                -0.15426877,
                0.0,
                0.34573123,
                0.8413447,
                1.9544997,
            ]),
            Tolerance::rel_abs(1e-4, 1e-4),
        );
    }
}
