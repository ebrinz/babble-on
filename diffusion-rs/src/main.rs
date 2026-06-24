//! Candle port of Plaid-1B.
//!   (no args)            validate the forward pass against the reference
//!   generate [steps] [seq_len]   sample text on Metal and decode

mod model;
mod sampler;
mod trng;

use anyhow::{Context, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use model::{load_safetensors, Config, DiffusionModel};

const MODEL_DIR: &str = "../models/plaid1b";
const TOKENIZER: &str = "../sidecar/misc/owt2_tokenizer.json";

#[derive(serde::Deserialize, Debug)]
struct Meta {
    dim: usize,
    n_blocks: usize,
    n_heads: usize,
    embed_dim: usize,
    vocab_size: usize,
    readout_scale: f64,
}

fn load_meta() -> Result<Meta> {
    Ok(serde_json::from_str(
        &std::fs::read_to_string(format!("{MODEL_DIR}/meta.json")).context("read meta.json")?,
    )?)
}

fn config(meta: &Meta) -> Config {
    Config {
        dim: meta.dim,
        n_blocks: meta.n_blocks,
        n_heads: meta.n_heads,
        embed_dim: meta.embed_dim,
        vocab_size: meta.vocab_size,
        readout_scale: meta.readout_scale,
    }
}

fn max_abs(a: &Tensor, b: &Tensor) -> Result<f32> {
    Ok((a - b)?.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?)
}

fn validate() -> Result<()> {
    let device = Device::Cpu;
    let meta = load_meta()?;
    let weights = load_safetensors(&format!("{MODEL_DIR}/plaid1b.safetensors"))?;
    let val = candle_core::safetensors::load(format!("{MODEL_DIR}/validation.safetensors"), &device)?;
    let z = val["z"].to_dtype(DType::F32)?;
    let gamma = val["gamma"].to_dtype(DType::F32)?;
    let ref_xr = val["x_reconst"].to_dtype(DType::F32)?;
    let ref_logits = val["logits"].to_dtype(DType::F32)?;

    println!("building model (cpu) ...");
    let m = DiffusionModel::load(&weights, config(&meta), device)?;
    let x_selfcond = z.zeros_like()?;
    let (logits, xr) = m.forward(&z, &gamma, &x_selfcond)?;

    let ref_lg_scale = ref_logits.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
    let xr_d = max_abs(&xr, &ref_xr)?;
    let lg_d = max_abs(&logits, &ref_logits)?;
    println!("x_reconst: max|Δ| = {xr_d:.3e}");
    println!("logits:    max|Δ| = {lg_d:.3e} (ref max|x| = {ref_lg_scale:.3e})");
    let ok = xr_d < 1e-2 && lg_d / ref_lg_scale.max(1.0) < 1e-2;
    println!("{}", if ok { "✅ forward matches the reference" } else { "❌ mismatch" });
    Ok(())
}

fn generate(steps: usize, seq_len: usize, entropy_file: Option<String>) -> Result<()> {
    let device = Device::new_metal(0).unwrap_or(Device::Cpu);
    println!("device: {device:?}");
    let meta = load_meta()?;
    let weights = load_safetensors(&format!("{MODEL_DIR}/plaid1b.safetensors"))?;
    let scalar = |k: &str| -> Result<f64> {
        Ok(weights[k].to_dtype(DType::F32)?.to_scalar::<f32>()? as f64)
    };
    let gamma_0 = scalar("bounds.gamma_0")?;
    let gamma_1 = scalar("bounds.gamma_1")?;
    let sched = sampler::NoiseSchedule::load(&weights)?;

    println!("building model ...");
    let m = DiffusionModel::load(&weights, config(&meta), device)?;
    let tok = tokenizers::Tokenizer::from_file(TOKENIZER).map_err(anyhow::Error::msg)?;

    // If an entropy file is given, those bytes seed the initial latent z1 (the
    // literal Gaussian noise tensor); per-step ancestral noise stays PRNG.
    let entropy = entropy_file.map(std::fs::read).transpose()?;
    if entropy.is_some() {
        println!("seeding initial latent from entropy file");
    }
    let mut first = true;
    let mut noise = |shape: &[usize]| -> candle_core::Result<Tensor> {
        if first {
            first = false;
            if let Some(raw) = &entropy {
                return trng::bytes_to_gaussians(raw, shape);
            }
        }
        Tensor::randn(0f32, 1f32, shape, &Device::Cpu)?.to_dtype(DType::F64)
    };

    println!("sampling: {seq_len} tokens, {steps} steps ...");
    let t0 = std::time::Instant::now();
    let ids = sampler::generate(&m, &sched, gamma_0, gamma_1, 1, seq_len, steps, 0.9, &mut noise)?;
    let dt = t0.elapsed().as_secs_f64();
    println!("done in {dt:.1}s ({:.0} ms/step)\n", dt / steps as f64 * 1000.0);

    let row: Vec<u32> = ids.i(0)?.to_vec1()?;
    let text = tok.decode(&row, false).map_err(anyhow::Error::msg)?;
    println!("{}", text);
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(|s| s.as_str()) == Some("generate") {
        let steps = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(48);
        let seq_len = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(64);
        let entropy_file = args.get(4).cloned();
        generate(steps, seq_len, entropy_file)
    } else {
        validate()
    }
}
