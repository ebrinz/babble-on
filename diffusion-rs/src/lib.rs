//! All-Rust Plaid-1B continuous text-diffusion engine (candle).
//!
//! `Engine::load` reads the converted weights once and keeps the model resident;
//! `Engine::generate` runs the VDM sampler — optionally seeding the initial
//! latent from TrueRNG entropy — and streams the text as it crystallizes via a
//! per-step callback.

pub mod model;
pub mod sampler;
pub mod trng;

use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Tensor, D};
use model::{load_safetensors, Config, DiffusionModel};

#[derive(serde::Deserialize)]
struct Meta {
    dim: usize,
    n_blocks: usize,
    n_heads: usize,
    embed_dim: usize,
    vocab_size: usize,
    readout_scale: f64,
}

pub struct Engine {
    model: DiffusionModel,
    sched: sampler::NoiseSchedule,
    gamma_0: f64,
    gamma_1: f64,
    tokenizer: tokenizers::Tokenizer,
    pub device_label: String,
}

impl Engine {
    /// Load the model (resident). `model_dir` holds plaid1b.safetensors +
    /// meta.json; `tokenizer_path` the owt2 tokenizer json.
    pub fn load(model_dir: &str, tokenizer_path: &str, prefer_metal: bool) -> Result<Self> {
        let device = if prefer_metal {
            Device::new_metal(0).unwrap_or(Device::Cpu)
        } else {
            Device::Cpu
        };
        let device_label = format!("{device:?}");
        let meta: Meta = serde_json::from_str(&std::fs::read_to_string(format!(
            "{model_dir}/meta.json"
        ))?)?;
        let weights = load_safetensors(&format!("{model_dir}/plaid1b.safetensors"))?;
        let scalar = |k: &str| -> Result<f64> {
            Ok(weights[k].to_dtype(DType::F32)?.to_scalar::<f32>()? as f64)
        };
        let gamma_0 = scalar("bounds.gamma_0")?;
        let gamma_1 = scalar("bounds.gamma_1")?;
        let sched = sampler::NoiseSchedule::load(&weights)?;
        let cfg = Config {
            dim: meta.dim,
            n_blocks: meta.n_blocks,
            n_heads: meta.n_heads,
            embed_dim: meta.embed_dim,
            vocab_size: meta.vocab_size,
            readout_scale: meta.readout_scale,
        };
        let model = DiffusionModel::load(&weights, cfg, device)?;
        let tokenizer =
            tokenizers::Tokenizer::from_file(tokenizer_path).map_err(anyhow::Error::msg)?;
        Ok(Engine { model, sched, gamma_0, gamma_1, tokenizer, device_label })
    }

    fn decode(&self, ids: &[u32]) -> String {
        self.tokenizer.decode(ids, false).unwrap_or_default()
    }

    /// Generate one sample. `entropy` (if given) seeds the initial latent z1.
    /// `on_step(i, total, partial_text)` fires every `preview_every` steps with
    /// the current best-guess decode. Returns the final text.
    pub fn generate(
        &self,
        steps: usize,
        seq_len: usize,
        score_temp: f64,
        preview_every: usize,
        entropy: Option<&[u8]>,
        mut on_step: impl FnMut(usize, usize, String),
    ) -> Result<String> {
        let entropy_vec = entropy.map(|e| e.to_vec());
        let mut first = true;
        let mut noise = |shape: &[usize]| -> candle_core::Result<Tensor> {
            if first {
                first = false;
                if let Some(raw) = &entropy_vec {
                    return trng::bytes_to_gaussians(raw, shape);
                }
            }
            Tensor::randn(0f32, 1f32, shape, &Device::Cpu)?.to_dtype(DType::F64)
        };

        let mut on_preview = |i: usize, total: usize, logits: &Tensor| {
            let decoded = logits
                .argmax(D::Minus1)
                .and_then(|t| t.i(0))
                .and_then(|t| t.to_device(&Device::Cpu))
                .and_then(|t| t.to_vec1::<u32>());
            if let Ok(ids) = decoded {
                on_step(i, total, self.decode(&ids));
            }
        };

        let ids = sampler::generate(
            &self.model, &self.sched, self.gamma_0, self.gamma_1, 1, seq_len, steps,
            score_temp, preview_every, &mut on_preview, &mut noise,
        )?;
        let row: Vec<u32> = ids.i(0)?.to_vec1()?;
        Ok(self.decode(&row))
    }
}
