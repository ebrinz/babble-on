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

    /// Decode each token id to its surface string (per-position), so the UI can
    /// diff positions across denoise steps.
    fn per_token(&self, ids: &[u32]) -> Vec<String> {
        ids.iter()
            .map(|&id| self.tokenizer.decode(&[id], false).unwrap_or_default())
            .collect()
    }

    /// Build the prompt prefix embeddings [1, k, embed_dim] (f64 CPU) and length.
    fn prefix(&self, prompt: Option<&str>, seq_len: usize) -> Result<(Option<Tensor>, usize)> {
        let Some(p) = prompt.map(str::trim).filter(|p| !p.is_empty()) else {
            return Ok((None, 0));
        };
        let enc = self.tokenizer.encode(p, false).map_err(anyhow::Error::msg)?;
        let ids = enc.get_ids();
        // Cap the prefix to half the sequence so there's room to generate.
        let k = ids.len().min(seq_len / 2);
        if k == 0 {
            return Ok((None, 0));
        }
        let ids_t = Tensor::from_vec(ids[..k].to_vec(), k, &Device::Cpu)?;
        let emb = self.model.embedding().to_device(&Device::Cpu)?.to_dtype(DType::F64)?;
        let pe = emb.index_select(&ids_t, 0)?.reshape((1, k, self.model.embed_dim()))?;
        Ok((Some(pe), k))
    }

    /// Generate one sample. `entropy` (if given) seeds the initial latent z1;
    /// `prompt` (if given) is inpainted at the prefix. `on_step(i, total,
    /// per_position_tokens)` fires every `preview_every` steps. Returns the
    /// final per-position tokens.
    #[allow(clippy::too_many_arguments)]
    pub fn generate(
        &self,
        steps: usize,
        seq_len: usize,
        score_temp: f64,
        initial_noise_scale: f64,
        ddim: bool,
        preview_every: usize,
        entropy: Option<&[u8]>,
        prompt: Option<&str>,
        mut on_step: impl FnMut(usize, usize, Vec<String>),
    ) -> Result<Vec<String>> {
        let (prefix_emb, prefix_len) = self.prefix(prompt, seq_len)?;

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
                on_step(i, total, self.per_token(&ids));
            }
        };

        let ids = sampler::generate(
            &self.model, &self.sched, self.gamma_0, self.gamma_1, 1, seq_len, steps,
            score_temp, initial_noise_scale, ddim, preview_every, &mut on_preview,
            prefix_emb.as_ref(), prefix_len, &mut noise,
        )?;
        let row: Vec<u32> = ids.i(0)?.to_vec1()?;
        Ok(self.per_token(&row))
    }
}
