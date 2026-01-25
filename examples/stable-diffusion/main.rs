#![recursion_limit = "256"]
// Stable Diffusion Example
//
// This example generates images from text prompts using Stable Diffusion.
// Weights are automatically downloaded from Hugging Face Hub on first run.
//
// # Usage
//
//    cargo run --release -- --prompt "a photo of a cat"
//
// The BPE vocabulary file will be downloaded automatically if not present.

use std::fs;
use std::io::Read;
use std::path::PathBuf;

use clap::{Parser, ValueEnum};

use burn::tensor::Tensor;
use hf_hub::api::sync::Api;

use diffusers_burn::pipelines::stable_diffusion::{
    generate_image_ddim, StableDiffusion, StableDiffusionConfig,
};
use diffusers_burn::pipelines::weights::{
    load_clip_safetensors, load_unet_safetensors, load_vae_safetensors,
};
use diffusers_burn::transformers::{SimpleTokenizer, SimpleTokenizerConfig};

const GUIDANCE_SCALE: f64 = 7.5;

#[cfg(feature = "wgpu")]
type Backend = burn::backend::Wgpu;

#[cfg(feature = "torch")]
type Backend = burn::backend::LibTorch<f32>;

#[cfg(feature = "ndarray")]
type Backend = burn::backend::NdArray<f32>;

#[derive(Debug, Clone, Copy, ValueEnum)]
enum StableDiffusionVersion {
    #[value(name = "v1-5")]
    V1_5,
    #[value(name = "v2-1")]
    V2_1,
}

impl StableDiffusionVersion {
    fn repo_id(&self) -> &'static str {
        match self {
            StableDiffusionVersion::V1_5 => "runwayml/stable-diffusion-v1-5",
            StableDiffusionVersion::V2_1 => "stabilityai/stable-diffusion-2-1",
        }
    }

    fn clip_repo_id(&self) -> &'static str {
        match self {
            // SD 1.5 uses OpenAI's CLIP
            StableDiffusionVersion::V1_5 => "openai/clip-vit-large-patch14",
            // SD 2.1 has CLIP in the main repo
            StableDiffusionVersion::V2_1 => "stabilityai/stable-diffusion-2-1",
        }
    }

    fn tokenizer_config(&self) -> SimpleTokenizerConfig {
        match self {
            StableDiffusionVersion::V1_5 => SimpleTokenizerConfig::v1_5(),
            StableDiffusionVersion::V2_1 => SimpleTokenizerConfig::v2_1(),
        }
    }
}

#[derive(Parser)]
#[command(author, version, about = "Generate images with Stable Diffusion", long_about = None)]
struct Args {
    /// The prompt to be used for image generation.
    #[arg(
        long,
        default_value = "A very realistic photo of a rusty robot walking on a sandy beach"
    )]
    prompt: String,

    /// The negative prompt (what to avoid in the image).
    #[arg(long, default_value = "")]
    negative_prompt: String,

    /// The height in pixels of the generated image.
    #[arg(long)]
    height: Option<usize>,

    /// The width in pixels of the generated image.
    #[arg(long)]
    width: Option<usize>,

    /// The UNet weight file, in .safetensors format (auto-downloaded if not specified).
    #[arg(long, value_name = "FILE")]
    unet_weights: Option<String>,

    /// The CLIP weight file, in .safetensors format (auto-downloaded if not specified).
    #[arg(long, value_name = "FILE")]
    clip_weights: Option<String>,

    /// The VAE weight file, in .safetensors format (auto-downloaded if not specified).
    #[arg(long, value_name = "FILE")]
    vae_weights: Option<String>,

    /// The file specifying the vocabulary to use for tokenization (auto-downloaded if not specified).
    #[arg(long, value_name = "FILE")]
    vocab_file: Option<String>,

    /// The number of steps to run the diffusion for.
    #[arg(long, default_value_t = 30)]
    n_steps: usize,

    /// The random seed to be used for the generation.
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// The name of the final image to generate.
    #[arg(long, value_name = "FILE", default_value = "output.png")]
    output: String,

    /// The Stable Diffusion version to use.
    #[arg(long, value_enum, default_value = "v1-5")]
    sd_version: StableDiffusionVersion,
}

/// Downloads a file from Hugging Face Hub if not already cached.
fn download_hf_file(repo_id: &str, filename: &str) -> anyhow::Result<PathBuf> {
    println!("  Downloading {} from {}...", filename, repo_id);
    let api = Api::new()?;
    let repo = api.model(repo_id.to_string());
    let path = repo.get(filename)?;
    println!("  Cached at: {}", path.display());
    Ok(path)
}

/// Downloads the BPE vocabulary file from OpenAI's GitHub.
fn download_bpe_vocab() -> anyhow::Result<PathBuf> {
    // Cache in HF cache directory for consistency
    let cache_dir = dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("huggingface")
        .join("clip");
    fs::create_dir_all(&cache_dir)?;

    let vocab_path = cache_dir.join("bpe_simple_vocab_16e6.txt");

    if vocab_path.exists() {
        println!("  Using cached vocabulary at: {}", vocab_path.display());
        return Ok(vocab_path);
    }

    println!("  Downloading BPE vocabulary from OpenAI GitHub...");
    let url = "https://github.com/openai/CLIP/raw/main/clip/bpe_simple_vocab_16e6.txt.gz";

    // Download gzipped file
    let response = ureq::get(url).call()?;
    let mut gz_data = Vec::new();
    response.into_reader().read_to_end(&mut gz_data)?;

    // Decompress
    let mut decoder = flate2::read::GzDecoder::new(&gz_data[..]);
    let mut content = String::new();
    decoder.read_to_string(&mut content)?;

    // Save to cache
    fs::write(&vocab_path, &content)?;
    println!("  Cached at: {}", vocab_path.display());

    Ok(vocab_path)
}

fn tensor_to_image(tensor: Tensor<Backend, 4>) -> image::RgbImage {
    // tensor shape: [1, 3, height, width], values in [0, 1]
    let [_, _, height, width] = tensor.dims();

    // Convert to [0, 255] u8 values
    let tensor = tensor * 255.0;
    let data: Vec<f32> = tensor.into_data().to_vec().unwrap();

    // Create image buffer
    let mut img = image::RgbImage::new(width as u32, height as u32);

    for y in 0..height {
        for x in 0..width {
            let r = data[0 * height * width + y * width + x] as u8;
            let g = data[1 * height * width + y * width + x] as u8;
            let b = data[2 * height * width + y * width + x] as u8;
            img.put_pixel(x as u32, y as u32, image::Rgb([r, g, b]));
        }
    }

    img
}

fn run(args: Args) -> anyhow::Result<()> {
    let device = Default::default();

    // Build configuration
    println!("Building configuration for {:?}...", args.sd_version);
    let sd_config = match args.sd_version {
        StableDiffusionVersion::V1_5 => StableDiffusionConfig::v1_5(None, args.height, args.width),
        StableDiffusionVersion::V2_1 => StableDiffusionConfig::v2_1(None, args.height, args.width),
    };

    // Download or use provided vocab file
    // The tokenizer expects the OpenAI CLIP BPE vocabulary format
    println!("\nPreparing vocabulary file...");
    let vocab_path = match &args.vocab_file {
        Some(path) => PathBuf::from(path),
        None => download_bpe_vocab()?,
    };

    // Build tokenizer
    println!("\nLoading tokenizer from {}...", vocab_path.display());
    let tokenizer = SimpleTokenizer::new(&vocab_path, args.sd_version.tokenizer_config())?;

    // Tokenize prompts
    println!("Tokenizing prompt: \"{}\"", args.prompt);
    let tokens = tokenizer.encode(&args.prompt)?;
    let uncond_tokens = tokenizer.encode(&args.negative_prompt)?;
    println!("  Prompt tokens: {} tokens", tokens.len());
    println!("  Negative prompt tokens: {} tokens", uncond_tokens.len());

    // Download weights
    println!("\nPreparing model weights...");

    let clip_weights = match &args.clip_weights {
        Some(path) => PathBuf::from(path),
        None => download_hf_file(args.sd_version.clip_repo_id(), "model.safetensors")?,
    };

    let vae_weights = match &args.vae_weights {
        Some(path) => PathBuf::from(path),
        None => download_hf_file(
            args.sd_version.repo_id(),
            "vae/diffusion_pytorch_model.safetensors",
        )?,
    };

    let unet_weights = match &args.unet_weights {
        Some(path) => PathBuf::from(path),
        None => download_hf_file(
            args.sd_version.repo_id(),
            "unet/diffusion_pytorch_model.safetensors",
        )?,
    };

    // Build scheduler
    println!("\nBuilding DDIM scheduler with {} steps...", args.n_steps);
    let scheduler = sd_config.build_ddim_scheduler::<Backend>(args.n_steps, &device);

    // Build models
    println!("Building CLIP text encoder...");
    let clip = sd_config.build_clip_transformer::<Backend>(&device);

    println!("Building VAE...");
    let vae = sd_config.build_vae::<Backend>(&device);

    println!("Building UNet...");
    let unet = sd_config.build_unet::<Backend>(&device, 4);

    // Load weights
    println!("\nLoading CLIP weights...");
    let clip = load_clip_safetensors::<Backend, _, _>(clip, &clip_weights, &device)?;

    println!("Loading VAE weights...");
    let vae = load_vae_safetensors::<Backend, _, _>(vae, &vae_weights, &device)?;

    println!("Loading UNet weights...");
    let unet = load_unet_safetensors::<Backend, _, _>(unet, &unet_weights, &device)?;

    // Assemble pipeline
    let pipeline = StableDiffusion {
        clip,
        vae,
        unet,
        width: sd_config.width,
        height: sd_config.height,
    };

    // Generate image
    println!("\nGenerating image...");
    println!("  Size: {}x{}", sd_config.width, sd_config.height);
    println!("  Steps: {}", args.n_steps);
    println!("  Guidance scale: {}", GUIDANCE_SCALE);
    println!("  Seed: {}", args.seed);

    let image_tensor = generate_image_ddim(
        &pipeline,
        &scheduler,
        &tokens,
        &uncond_tokens,
        GUIDANCE_SCALE,
        args.seed,
        &device,
    );

    // Save image
    println!("\nSaving image to {}...", args.output);
    let img = tensor_to_image(image_tensor);
    img.save(&args.output)?;

    println!("Done!");
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    run(args)
}
