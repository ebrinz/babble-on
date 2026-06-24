//! Candle port of Plaid-1B — milestone 1: load the converted weights, the
//! arch metadata, and the validation reference, and report what we got.

use anyhow::{Context, Result};
use candle_core::Device;

const MODEL_DIR: &str = "../models/plaid1b";

#[derive(serde::Deserialize, Debug)]
struct Meta {
    dim: usize,
    n_blocks: usize,
    n_heads: usize,
    embed_dim: usize,
    vocab_size: usize,
    gamma_0: f64,
    gamma_1: f64,
    readout_scale: f64,
}

fn main() -> Result<()> {
    let metal = Device::new_metal(0);
    let device = metal.unwrap_or(Device::Cpu);
    println!("compute device: {device:?}");

    let meta: Meta = serde_json::from_str(
        &std::fs::read_to_string(format!("{MODEL_DIR}/meta.json")).context("read meta.json")?,
    )?;
    println!("meta: {meta:?}");

    let weights = candle_core::safetensors::load(
        format!("{MODEL_DIR}/plaid1b.safetensors"),
        &Device::Cpu,
    )
    .context("load plaid1b.safetensors")?;
    println!("loaded {} weight tensors", weights.len());

    // Spot-check a few expected keys + shapes.
    for key in [
        "emb.matrix",
        "model.input_linear.weight",
        "model.blocks.0.attn_qkv.weight",
        "model.blocks.0.mlp.fc1.weight",
        "model.output_linear.weight",
        "sched.W1",
        "bounds.gamma_0",
    ] {
        match weights.get(key) {
            Some(t) => println!("  {key:>34}  {:?}  {:?}", t.shape().dims(), t.dtype()),
            None => println!("  {key:>34}  MISSING"),
        }
    }

    let val = candle_core::safetensors::load(
        format!("{MODEL_DIR}/validation.safetensors"),
        &Device::Cpu,
    )
    .context("load validation.safetensors")?;
    println!("validation reference:");
    let mut keys: Vec<_> = val.keys().cloned().collect();
    keys.sort();
    for k in keys {
        println!("  {k:>12}  {:?}", val[&k].shape().dims());
    }
    Ok(())
}
