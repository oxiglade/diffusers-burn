//! 2D UNet Building Blocks
//!

use burn::{
    config::Config,
    module::Module,
    nn,
    tensor::{
        backend::Backend,
        module::{avg_pool2d, interpolate},
        ops::{InterpolateMode, InterpolateOptions},
        Tensor,
    },
};

use crate::utils::pad_with_zeros;

use super::{
    attention::{
        AttentionBlock, AttentionBlockConfig, SpatialTransformer, SpatialTransformerConfig,
    },
    resnet::{ResnetBlock2D, ResnetBlock2DConfig},
};

use alloc::vec;
use alloc::vec::Vec;

#[derive(Config, Debug)]
struct Downsample2DConfig {
    in_channels: usize,
    use_conv: bool,
    out_channels: usize,
    padding: usize,
}

#[derive(Module, Debug)]
struct Downsample2D<B: Backend> {
    conv: Option<nn::conv::Conv2d<B>>,
    padding: usize,
}

impl Downsample2DConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> Downsample2D<B> {
        let conv = if self.use_conv {
            let conv = nn::conv::Conv2dConfig::new([self.in_channels, self.out_channels], [3, 3])
                .with_stride([2, 2])
                .with_padding(nn::PaddingConfig2d::Explicit(self.padding, self.padding))
                .init(device);

            Some(conv)
        } else {
            None
        };

        Downsample2D {
            conv,
            padding: self.padding,
        }
    }
}

impl<B: Backend> Downsample2D<B> {
    fn pad_tensor(xs: Tensor<B, 4>, padding: usize) -> Tensor<B, 4> {
        if padding == 0 {
            let xs = pad_with_zeros(xs, 4 - 1, 0, 1);
            return pad_with_zeros(xs, 4 - 2, 0, 1);
        }

        return xs;
    }

    fn forward(&self, xs: Tensor<B, 4>) -> Tensor<B, 4> {
        match &self.conv {
            None => avg_pool2d(xs, [2, 2], [2, 2], [0, 0], true, false),
            Some(conv) => conv.forward(Self::pad_tensor(xs, self.padding)),
        }
    }
}

#[derive(Config, Debug)]
struct Upsample2DConfig {
    in_channels: usize,
    out_channels: usize,
}

// This does not support the conv-transpose mode.
#[derive(Module, Debug)]
struct Upsample2D<B: Backend> {
    conv: nn::conv::Conv2d<B>,
}

impl Upsample2DConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> Upsample2D<B> {
        let conv = nn::conv::Conv2dConfig::new([self.in_channels, self.out_channels], [3, 3])
            .with_padding(nn::PaddingConfig2d::Explicit(1, 1))
            .init(device);

        Upsample2D { conv }
    }
}

impl<B: Backend> Upsample2D<B> {
    fn forward(&self, xs: Tensor<B, 4>, size: Option<(usize, usize)>) -> Tensor<B, 4> {
        let xs = match size {
            None => {
                let [_bsize, _channels, height, width] = xs.dims();
                interpolate(
                    xs,
                    [2 * height, 2 * width],
                    InterpolateOptions::new(InterpolateMode::Nearest),
                )
            }
            Some((h, w)) => interpolate(
                xs,
                [h, w],
                InterpolateOptions::new(InterpolateMode::Nearest),
            ),
        };

        self.conv.forward(xs)
    }
}

#[derive(Config, Debug)]
pub struct DownEncoderBlock2DConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    #[config(default = 1)]
    pub n_layers: usize,
    #[config(default = 1e-6)]
    pub resnet_eps: f64,
    #[config(default = 32)]
    pub resnet_groups: usize,
    #[config(default = 1.)]
    pub output_scale_factor: f64,
    #[config(default = true)]
    pub add_downsample: bool,
    #[config(default = 1)]
    pub downsample_padding: usize,
}

#[derive(Module, Debug)]
pub struct DownEncoderBlock2D<B: Backend> {
    resnets: Vec<ResnetBlock2D<B>>,
    downsampler: Option<Downsample2D<B>>,
}

impl DownEncoderBlock2DConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> DownEncoderBlock2D<B> {
        let resnets: Vec<_> = {
            (0..(self.n_layers))
                .map(|i| {
                    let in_channels = if i == 0 {
                        self.in_channels
                    } else {
                        self.out_channels
                    };

                    let conv_cfg = ResnetBlock2DConfig::new(in_channels)
                        .with_out_channels(Some(self.out_channels))
                        .with_groups(self.resnet_groups)
                        .with_eps(self.resnet_eps)
                        .with_output_scale_factor(self.output_scale_factor);

                    conv_cfg.init(device)
                })
                .collect()
        };

        let downsampler = if self.add_downsample {
            let downsample_cfg = Downsample2DConfig {
                in_channels: self.out_channels,
                use_conv: true,
                out_channels: self.out_channels,
                padding: self.downsample_padding,
            };
            Some(downsample_cfg.init(device))
        } else {
            None
        };

        DownEncoderBlock2D {
            resnets,
            downsampler,
        }
    }
}

impl<B: Backend> DownEncoderBlock2D<B> {
    fn forward(&self, xs: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut xs = xs.clone();
        for resnet in self.resnets.iter() {
            xs = resnet.forward(xs, None)
        }
        match &self.downsampler {
            Some(downsampler) => downsampler.forward(xs),
            None => xs,
        }
    }
}

#[derive(Config, Debug)]
pub struct UpDecoderBlock2DConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    #[config(default = 1)]
    pub n_layers: usize,
    #[config(default = 1e-6)]
    pub resnet_eps: f64,
    #[config(default = 32)]
    pub resnet_groups: usize,
    #[config(default = 1.)]
    pub output_scale_factor: f64,
    #[config(default = true)]
    pub add_upsample: bool,
}

#[derive(Module, Debug)]
pub struct UpDecoderBlock2D<B: Backend> {
    resnets: Vec<ResnetBlock2D<B>>,
    upsampler: Option<Upsample2D<B>>,
}

impl UpDecoderBlock2DConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> UpDecoderBlock2D<B> {
        let resnets: Vec<_> = {
            (0..(self.n_layers))
                .map(|i| {
                    let in_channels = if i == 0 {
                        self.in_channels
                    } else {
                        self.out_channels
                    };

                    let conv_cfg = ResnetBlock2DConfig::new(in_channels)
                        .with_out_channels(Some(self.out_channels))
                        .with_groups(self.resnet_groups)
                        .with_eps(self.resnet_eps)
                        .with_output_scale_factor(self.output_scale_factor);

                    conv_cfg.init(device)
                })
                .collect()
        };

        let upsampler = if self.add_upsample {
            let upsample = Upsample2DConfig {
                in_channels: self.out_channels,
                out_channels: self.out_channels,
            };

            Some(upsample.init(device))
        } else {
            None
        };

        UpDecoderBlock2D { resnets, upsampler }
    }
}

impl<B: Backend> UpDecoderBlock2D<B> {
    fn forward(&self, xs: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut xs = xs.clone();
        for resnet in self.resnets.iter() {
            xs = resnet.forward(xs, None)
        }
        match &self.upsampler {
            Some(upsampler) => upsampler.forward(xs, None),
            None => xs,
        }
    }
}

#[derive(Config, Debug)]
pub struct UNetMidBlock2DConfig {
    in_channels: usize,
    temb_channels: Option<usize>,
    #[config(default = 1)]
    pub n_layers: usize,
    #[config(default = 1e-6)]
    pub resnet_eps: f64,
    pub resnet_groups: Option<usize>,
    pub attn_num_head_channels: Option<usize>,
    // attention_type "default"
    #[config(default = 1.)]
    pub output_scale_factor: f64,
}

#[derive(Module, Debug)]
struct AttentionResnetBlock2D<B: Backend> {
    attention_block: AttentionBlock<B>,
    resnet_block: ResnetBlock2D<B>,
}

#[derive(Module, Debug)]
pub struct UNetMidBlock2D<B: Backend> {
    resnet: ResnetBlock2D<B>,
    attn_resnets: Vec<AttentionResnetBlock2D<B>>,
}

impl UNetMidBlock2DConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> UNetMidBlock2D<B> {
        let resnet_groups = self
            .resnet_groups
            .unwrap_or_else(|| usize::min(self.in_channels / 4, 32));

        let resnet = ResnetBlock2DConfig::new(self.in_channels)
            .with_eps(self.resnet_eps)
            .with_groups(resnet_groups)
            .with_output_scale_factor(self.output_scale_factor)
            .with_temb_channels(self.temb_channels)
            .init(device);

        let mut attn_resnets = vec![];
        for _index in 0..self.n_layers {
            let attention_block = AttentionBlockConfig::new(self.in_channels)
                .with_n_head_channels(self.attn_num_head_channels)
                .with_n_groups(resnet_groups)
                .with_rescale_output_factor(self.output_scale_factor)
                .with_eps(self.resnet_eps)
                .init(device);

            let resnet_block = ResnetBlock2DConfig::new(self.in_channels)
                .with_eps(self.resnet_eps)
                .with_groups(resnet_groups)
                .with_output_scale_factor(self.output_scale_factor)
                .with_temb_channels(self.temb_channels)
                .init(device);

            attn_resnets.push(AttentionResnetBlock2D {
                attention_block,
                resnet_block,
            })
        }

        UNetMidBlock2D {
            resnet,
            attn_resnets,
        }
    }
}

impl<B: Backend> UNetMidBlock2D<B> {
    pub fn forward(&self, xs: Tensor<B, 4>, temb: Option<Tensor<B, 2>>) -> Tensor<B, 4> {
        let mut xs = self.resnet.forward(xs, temb.clone());
        for block in self.attn_resnets.iter() {
            xs = block
                .resnet_block
                .forward(block.attention_block.forward(xs), temb.clone())
        }

        xs
    }
}

#[derive(Config, Debug)]
pub struct UNetMidBlock2DCrossAttnConfig {
    in_channels: usize,
    temb_channels: Option<usize>,
    #[config(default = 1)]
    pub n_layers: usize,
    #[config(default = 1e-6)]
    pub resnet_eps: f64,
    // Note: Should default to 32
    pub resnet_groups: Option<usize>,
    #[config(default = 1)]
    pub attn_num_head_channels: usize,
    // attention_type "default"
    #[config(default = 1.)]
    pub output_scale_factor: f64,
    #[config(default = 1280)]
    pub cross_attn_dim: usize,
    pub sliced_attention_size: Option<usize>,
    #[config(default = false)]
    pub use_linear_projection: bool,
    #[config(default = 1)]
    pub transformer_layers_per_block: usize,
}

#[derive(Module, Debug)]
struct SpatialTransformerResnetBlock2D<B: Backend> {
    spatial_transformer: SpatialTransformer<B>,
    resnet_block: ResnetBlock2D<B>,
}

#[derive(Module, Debug)]
pub struct UNetMidBlock2DCrossAttn<B: Backend> {
    resnet: ResnetBlock2D<B>,
    attn_resnets: Vec<SpatialTransformerResnetBlock2D<B>>,
}

impl UNetMidBlock2DCrossAttnConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> UNetMidBlock2DCrossAttn<B> {
        let resnet_groups = self
            .resnet_groups
            .unwrap_or_else(|| usize::min(self.in_channels / 4, 32));
        let resnet = ResnetBlock2DConfig::new(self.in_channels)
            .with_eps(self.resnet_eps)
            .with_groups(resnet_groups)
            .with_output_scale_factor(self.output_scale_factor)
            .with_temb_channels(self.temb_channels)
            .init(device);

        let mut attn_resnets = vec![];
        for _index in 0..self.n_layers {
            let spatial_transformer = SpatialTransformerConfig::new(
                self.in_channels,
                self.attn_num_head_channels,
                self.in_channels / self.attn_num_head_channels,
            )
            .with_depth(1)
            .with_n_groups(resnet_groups)
            .with_d_context(Some(self.cross_attn_dim))
            .with_sliced_attn_size(self.sliced_attention_size)
            .with_use_linear_projection(self.use_linear_projection)
            .init(device);

            let resnet_block = ResnetBlock2DConfig::new(self.in_channels)
                .with_eps(self.resnet_eps)
                .with_groups(resnet_groups)
                .with_output_scale_factor(self.output_scale_factor)
                .with_temb_channels(self.temb_channels)
                .init(device);

            attn_resnets.push(SpatialTransformerResnetBlock2D {
                spatial_transformer,
                resnet_block,
            })
        }

        UNetMidBlock2DCrossAttn {
            resnet,
            attn_resnets,
        }
    }
}

impl<B: Backend> UNetMidBlock2DCrossAttn<B> {
    pub fn forward(
        &self,
        xs: Tensor<B, 4>,
        temb: Option<Tensor<B, 2>>,
        encoder_hidden_states: Option<Tensor<B, 3>>,
    ) -> Tensor<B, 4> {
        let mut xs = self.resnet.forward(xs, temb.clone());
        for block in self.attn_resnets.iter() {
            let trans = block
                .spatial_transformer
                .forward(xs, encoder_hidden_states.clone());
            xs = self.resnet.forward(trans, temb.clone());
        }

        xs
    }
}

#[derive(Config, Debug, Copy)]
pub struct DownBlock2DConfig {
    in_channels: usize,
    out_channels: usize,
    temb_channels: Option<usize>,
    #[config(default = 1)]
    pub n_layers: usize,
    #[config(default = 1e-6)]
    pub resnet_eps: f64,
    // resnet_time_scale_shift: "default"
    // resnet_act_fn: "swish"
    #[config(default = 32)]
    pub resnet_groups: usize,
    #[config(default = 1.)]
    pub output_scale_factor: f64,
    #[config(default = true)]
    pub add_downsample: bool,
    #[config(default = 1)]
    pub downsample_padding: usize,
}

#[derive(Module, Debug)]
pub struct DownBlock2D<B: Backend> {
    resnets: Vec<ResnetBlock2D<B>>,
    downsampler: Option<Downsample2D<B>>,
}

impl DownBlock2DConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> DownBlock2D<B> {
        let resnets = (0..self.n_layers)
            .map(|_| {
                ResnetBlock2DConfig::new(self.out_channels)
                    .with_eps(self.resnet_eps)
                    .with_groups(self.resnet_groups)
                    .with_output_scale_factor(self.output_scale_factor)
                    .with_temb_channels(self.temb_channels)
                    .init(device)
            })
            .collect();

        let downsampler = if self.add_downsample {
            Some(
                Downsample2DConfig::new(
                    self.out_channels,
                    true,
                    self.out_channels,
                    self.downsample_padding,
                )
                .init(device),
            )
        } else {
            None
        };

        DownBlock2D {
            resnets,
            downsampler,
        }
    }
}

impl<B: Backend> DownBlock2D<B> {
    pub fn forward(
        &self,
        xs: Tensor<B, 4>,
        temb: Option<Tensor<B, 2>>,
    ) -> (Tensor<B, 4>, Vec<Tensor<B, 4>>) {
        let mut xs = xs;
        let mut output_states = vec![];
        for resnet in self.resnets.iter() {
            xs = resnet.forward(xs, temb.clone());
            output_states.push(xs.clone());
        }

        if let Some(downsampler) = &self.downsampler {
            xs = downsampler.forward(xs);
            output_states.push(xs.clone());
        }

        (xs, output_states)
    }
}

#[derive(Config, Debug)]
pub struct CrossAttnDownBlock2DConfig {
    in_channels: usize,
    out_channels: usize,
    temb_channels: Option<usize>,
    pub downblock: DownBlock2DConfig,
    #[config(default = 1)]
    pub attn_num_head_channels: usize,
    #[config(default = 1280)]
    pub cross_attention_dim: usize,
    // attention_type: "default"
    pub sliced_attention_size: Option<usize>,
    #[config(default = false)]
    pub use_linear_projection: bool,
}

#[derive(Module, Debug)]
pub struct CrossAttnDownBlock2D<B: Backend> {
    downblock: DownBlock2D<B>,
    attentions: Vec<SpatialTransformer<B>>,
}

impl CrossAttnDownBlock2DConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> CrossAttnDownBlock2D<B> {
        let mut downblock = self.downblock;
        downblock.in_channels = self.in_channels;
        downblock.out_channels = self.out_channels;
        downblock.temb_channels = self.temb_channels;
        let downblock = self.downblock.init(device);

        let attentions = (0..self.downblock.n_layers)
            .map(|_| {
                SpatialTransformerConfig::new(
                    self.out_channels,
                    self.attn_num_head_channels,
                    self.out_channels / self.attn_num_head_channels,
                )
                .with_depth(1)
                .with_d_context(Some(self.cross_attention_dim))
                .with_n_groups(self.downblock.resnet_groups)
                .with_sliced_attn_size(self.sliced_attention_size)
                .with_use_linear_projection(self.use_linear_projection)
                .init(device)
            })
            .collect();

        CrossAttnDownBlock2D {
            downblock,
            attentions,
        }
    }
}

impl<B: Backend> CrossAttnDownBlock2D<B> {
    pub fn forward(
        &self,
        xs: Tensor<B, 4>,
        temb: Option<Tensor<B, 2>>,
        encoder_hidden_states: Option<Tensor<B, 3>>,
    ) -> (Tensor<B, 4>, Vec<Tensor<B, 4>>) {
        let mut xs = xs;
        let mut output_states = vec![];
        for (resnet, attn) in self.downblock.resnets.iter().zip(self.attentions.iter()) {
            xs = resnet.forward(xs, temb.clone());
            xs = attn.forward(xs, encoder_hidden_states.clone());
            output_states.push(xs.clone());
        }

        if let Some(downsampler) = &self.downblock.downsampler {
            xs = downsampler.forward(xs);
            output_states.push(xs.clone());
        }

        (xs, output_states)
    }
}

#[derive(Config, Debug)]
pub struct UpBlock2DConfig {
    in_channels: usize,
    prev_output_channels: usize,
    out_channels: usize,
    temb_channels: Option<usize>,
    #[config(default = 1)]
    pub n_layers: usize,
    #[config(default = 1e-6)]
    pub resnet_eps: f64,
    // resnet_time_scale_shift: "default"
    // resnet_act_fn: "swish"
    #[config(default = 32)]
    pub resnet_groups: usize,
    #[config(default = 1.)]
    pub output_scale_factor: f64,
    #[config(default = true)]
    pub add_upsample: bool,
}

#[derive(Module, Debug)]
pub struct UpBlock2D<B: Backend> {
    resnets: Vec<ResnetBlock2D<B>>,
    upsampler: Option<Upsample2D<B>>,
}

impl UpBlock2DConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> UpBlock2D<B> {
        let resnets = (0..self.n_layers)
            .map(|i| {
                let res_skip_channels = if i == self.n_layers - 1 {
                    self.in_channels
                } else {
                    self.out_channels
                };

                let resnet_in_channels = if i == 0 {
                    self.prev_output_channels
                } else {
                    self.out_channels
                };

                let in_channels = resnet_in_channels + res_skip_channels;

                ResnetBlock2DConfig::new(in_channels)
                    .with_out_channels(Some(self.out_channels))
                    .with_temb_channels(self.temb_channels)
                    .with_eps(self.resnet_eps)
                    .with_output_scale_factor(self.output_scale_factor)
                    .init(device)
            })
            .collect();

        let upsampler = if self.add_upsample {
            let upsampler =
                Upsample2DConfig::new(self.out_channels, self.out_channels).init(device);
            Some(upsampler)
        } else {
            None
        };

        UpBlock2D { resnets, upsampler }
    }
}

impl<B: Backend> UpBlock2D<B> {
    pub fn forward(
        &self,
        xs: Tensor<B, 4>,
        res_xs: &[Tensor<B, 4>],
        temb: Option<Tensor<B, 2>>,
        upsample_size: Option<(usize, usize)>,
    ) -> Tensor<B, 4> {
        let mut xs = xs;
        for (index, resnet) in self.resnets.iter().enumerate() {
            xs = Tensor::cat(
                vec![xs.clone(), res_xs[res_xs.len() - index - 1].clone()],
                1,
            );
            xs = resnet.forward(xs, temb.clone());
        }

        match &self.upsampler {
            Some(upsampler) => upsampler.forward(xs, upsample_size),
            None => xs,
        }
    }
}

#[derive(Config, Debug)]
pub struct CrossAttnUpBlock2DConfig {
    in_channels: usize,
    prev_output_channels: usize,
    out_channels: usize,
    temb_channels: Option<usize>,
    pub upblock: UpBlock2DConfig,
    #[config(default = 1)]
    pub attn_num_head_channels: usize,
    #[config(default = 1280)]
    pub cross_attention_dim: usize,
    // attention_type: "default"
    pub sliced_attention_size: Option<usize>,
    #[config(default = false)]
    pub use_linear_projection: bool,
}

#[derive(Module, Debug)]
pub struct CrossAttnUpBlock2D<B: Backend> {
    pub upblock: UpBlock2D<B>,
    pub attentions: Vec<SpatialTransformer<B>>,
}

impl CrossAttnUpBlock2DConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> CrossAttnUpBlock2D<B> {
        let mut upblock_config = self.upblock.clone();
        upblock_config.in_channels = self.in_channels;
        upblock_config.prev_output_channels = self.prev_output_channels;
        upblock_config.out_channels = self.out_channels;
        upblock_config.temb_channels = self.temb_channels;
        let upblock = upblock_config.init(device);

        let attentions = (0..self.upblock.n_layers)
            .map(|_| {
                SpatialTransformerConfig::new(
                    self.out_channels,
                    self.attn_num_head_channels,
                    self.out_channels / self.attn_num_head_channels,
                )
                .with_depth(1)
                .with_d_context(Some(self.cross_attention_dim))
                .with_n_groups(self.upblock.resnet_groups)
                .with_sliced_attn_size(self.sliced_attention_size)
                .with_use_linear_projection(self.use_linear_projection)
                .init(device)
            })
            .collect();

        CrossAttnUpBlock2D {
            upblock,
            attentions,
        }
    }
}

impl<B: Backend> CrossAttnUpBlock2D<B> {
    pub fn forward(
        &self,
        xs: Tensor<B, 4>,
        res_xs: &[Tensor<B, 4>],
        temb: Option<Tensor<B, 2>>,
        upsample_size: Option<(usize, usize)>,
        encoder_hidden_states: Option<Tensor<B, 3>>,
    ) -> Tensor<B, 4> {
        let mut xs = xs;
        for (index, resnet) in self.upblock.resnets.iter().enumerate() {
            xs = Tensor::cat(
                vec![xs.clone(), res_xs[res_xs.len() - index - 1].clone()],
                1,
            );
            xs = resnet.forward(xs, temb.clone());
            xs = self.attentions[index].forward(xs, encoder_hidden_states.clone());
        }

        match &self.upblock.upsampler {
            Some(upsampler) => upsampler.forward(xs, upsample_size),
            None => xs,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestBackend;
    use burn::tensor::{Distribution, Shape, TensorData, Tolerance};

    #[test]
    fn test_downsample_2d_no_conv() {
        let device = Default::default();
        let tensor: Tensor<TestBackend, 4> = Tensor::from_data(
            TensorData::from([
                [
                    [[0.0351, 0.4179], [0.0137, 0.6947]],
                    [[0.9526, 0.5386], [0.2856, 0.1839]],
                    [[0.3215, 0.4595], [0.6777, 0.3946]],
                    [[0.5221, 0.4230], [0.2774, 0.1069]],
                ],
                [
                    [[0.8941, 0.8696], [0.5735, 0.8750]],
                    [[0.6718, 0.4144], [0.1038, 0.2629]],
                    [[0.7467, 0.9415], [0.5005, 0.6309]],
                    [[0.6534, 0.2019], [0.3670, 0.8074]],
                ],
            ]),
            &device,
        );

        let downsample_2d = Downsample2DConfig::new(4, false, 4, 0).init(&device);
        let output = downsample_2d.forward(tensor);

        output.into_data().assert_approx_eq::<f32>(
            &TensorData::from([
                [[[0.2904]], [[0.4902]], [[0.4633]], [[0.3323]]],
                [[[0.8031]], [[0.3632]], [[0.7049]], [[0.5074]]],
            ]),
            Tolerance::rel_abs(1e-3, 1e-3),
        );
    }

    #[test]
    fn test_pad_tensor_0() {
        let device = Default::default();
        let tensor: Tensor<TestBackend, 4> = Tensor::from_data(
            TensorData::from([
                [
                    [[0.8600, 0.9473], [0.2543, 0.6181]],
                    [[0.3889, 0.7722], [0.6736, 0.0454]],
                    [[0.2809, 0.4672], [0.1632, 0.3959]],
                    [[0.5317, 0.0831], [0.8353, 0.3654]],
                ],
                [
                    [[0.6106, 0.4130], [0.7932, 0.8800]],
                    [[0.8750, 0.1991], [0.7018, 0.7865]],
                    [[0.7470, 0.2071], [0.2699, 0.4425]],
                    [[0.7763, 0.0227], [0.6210, 0.0730]],
                ],
            ]),
            &device,
        );

        let output = Downsample2D::pad_tensor(tensor, 0);

        output.into_data().assert_approx_eq::<f32>(
            &TensorData::from([
                [
                    [
                        [0.8600, 0.9473, 0.0000],
                        [0.2543, 0.6181, 0.0000],
                        [0.0000, 0.0000, 0.0000],
                    ],
                    [
                        [0.3889, 0.7722, 0.0000],
                        [0.6736, 0.0454, 0.0000],
                        [0.0000, 0.0000, 0.0000],
                    ],
                    [
                        [0.2809, 0.4672, 0.0000],
                        [0.1632, 0.3959, 0.0000],
                        [0.0000, 0.0000, 0.0000],
                    ],
                    [
                        [0.5317, 0.0831, 0.0000],
                        [0.8353, 0.3654, 0.0000],
                        [0.0000, 0.0000, 0.0000],
                    ],
                ],
                [
                    [
                        [0.6106, 0.4130, 0.0000],
                        [0.7932, 0.8800, 0.0000],
                        [0.0000, 0.0000, 0.0000],
                    ],
                    [
                        [0.8750, 0.1991, 0.0000],
                        [0.7018, 0.7865, 0.0000],
                        [0.0000, 0.0000, 0.0000],
                    ],
                    [
                        [0.7470, 0.2071, 0.0000],
                        [0.2699, 0.4425, 0.0000],
                        [0.0000, 0.0000, 0.0000],
                    ],
                    [
                        [0.7763, 0.0227, 0.0000],
                        [0.6210, 0.0730, 0.0000],
                        [0.0000, 0.0000, 0.0000],
                    ],
                ],
            ]),
            Tolerance::rel_abs(1e-3, 1e-3),
        );
    }

    #[test]
    fn test_down_encoder_block2d() {
        let device = Default::default();
        TestBackend::seed(&device, 0);
        let block = DownEncoderBlock2DConfig::new(32, 32).init::<TestBackend>(&device);

        let tensor: Tensor<TestBackend, 4> =
            Tensor::random([4, 32, 32, 32], Distribution::Default, &device);
        let output = block.forward(tensor.clone());

        assert_eq!(output.shape(), Shape::new([4, 32, 16, 16]));
    }

    #[test]
    fn test_up_decoder_block2d() {
        let device = Default::default();
        TestBackend::seed(&device, 0);
        let block = UpDecoderBlock2DConfig::new(32, 32).init::<TestBackend>(&device);

        let tensor: Tensor<TestBackend, 4> =
            Tensor::random([4, 32, 32, 32], Distribution::Default, &device);
        let output = block.forward(tensor.clone());

        assert_eq!(output.shape(), Shape::new([4, 32, 64, 64]));
    }
}
