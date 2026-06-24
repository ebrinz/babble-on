//! CLI for the candle Plaid port (thin wrapper over the `diffusion_rs` lib).
//!   (no args)                      validate the forward pass vs the reference
//!   generate [steps] [seq] [file]  sample on Metal; optional entropy file

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use diffusion_rs::model::{load_safetensors, Config, DiffusionModel};
use diffusion_rs::Engine;

const MODEL_DIR: &str = "../models/plaid1b";
const TOKENIZER: &str = "../sidecar/misc/owt2_tokenizer.json";

#[derive(serde::Deserialize)]
struct Meta {
    dim: usize,
    n_blocks: usize,
    n_heads: usize,
    embed_dim: usize,
    vocab_size: usize,
    readout_scale: f64,
}

fn validate() -> Result<()> {
    let device = Device::Cpu;
    let meta: Meta = serde_json::from_str(
        &std::fs::read_to_string(format!("{MODEL_DIR}/meta.json")).context("read meta.json")?,
    )?;
    let weights = load_safetensors(&format!("{MODEL_DIR}/plaid1b.safetensors"))?;
    let val = candle_core::safetensors::load(format!("{MODEL_DIR}/validation.safetensors"), &device)?;
    let z = val["z"].to_dtype(DType::F32)?;
    let gamma = val["gamma"].to_dtype(DType::F32)?;
    let ref_xr = val["x_reconst"].to_dtype(DType::F32)?;
    let ref_logits = val["logits"].to_dtype(DType::F32)?;

    let cfg = Config {
        dim: meta.dim,
        n_blocks: meta.n_blocks,
        n_heads: meta.n_heads,
        embed_dim: meta.embed_dim,
        vocab_size: meta.vocab_size,
        readout_scale: meta.readout_scale,
    };
    println!("building model (cpu) ...");
    let m = DiffusionModel::load(&weights, cfg, device)?;
    let (logits, xr) = m.forward(&z, &gamma, &z.zeros_like()?)?;

    let max_abs = |a: &Tensor, b: &Tensor| -> Result<f32> {
        Ok((a - b)?.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?)
    };
    let ref_lg = ref_logits.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
    let xr_d = max_abs(&xr, &ref_xr)?;
    let lg_d = max_abs(&logits, &ref_logits)?;
    println!("x_reconst max|Δ| = {xr_d:.3e}   logits max|Δ| = {lg_d:.3e}");
    println!("{}", if xr_d < 1e-2 && lg_d / ref_lg.max(1.0) < 1e-2 {
        "✅ forward matches the reference"
    } else {
        "❌ mismatch"
    });
    Ok(())
}

fn generate(steps: usize, seq_len: usize, entropy_file: Option<String>) -> Result<()> {
    let entropy = entropy_file.map(std::fs::read).transpose()?;
    println!("loading model ...");
    let eng = Engine::load(MODEL_DIR, TOKENIZER, true)?;
    println!("device: {}", eng.device_label);
    if entropy.is_some() {
        println!("seeding initial latent from entropy file");
    }
    println!("sampling: {seq_len} tokens, {steps} steps ...");
    let t0 = std::time::Instant::now();
    let text = eng.generate(
        steps,
        seq_len,
        0.9,
        (steps / 12).max(1),
        entropy.as_deref(),
        |i, total, _partial| eprint!("\r  step {i}/{total}   "),
    )?;
    let dt = t0.elapsed().as_secs_f64();
    eprintln!();
    println!("done in {dt:.1}s ({:.0} ms/step)\n", dt / steps as f64 * 1000.0);
    println!("{text}");
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(|s| s.as_str()) == Some("generate") {
        let steps = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(48);
        let seq_len = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(64);
        generate(steps, seq_len, args.get(4).cloned())
    } else {
        validate()
    }
}
