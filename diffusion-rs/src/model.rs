//! Candle port of Plaid-1B's `DiffusionModel` forward pass.
//!
//! Mirrors `sidecar/lib/models.py` (the published Plaid model) exactly, for the
//! inference case where the self-conditioning mask is all-ones:
//!   - input rescale, input/selfcond/gamma projections
//!   - 24 rotary transformer blocks (RMSNorm, QKV attention, gelu MLP)
//!   - LayerNorm + the baked muP readout scale (0.125)
//!   - the VDM output head: logits = bias + x·Woutᵀ + z_scaled·embᵀ, and
//!     x_reconst = softmax(logits)·emb
//!
//! All forward math is f32 (Metal-friendly); the f64 noise schedule lives in
//! the sampler, not here.

use std::collections::HashMap;

use candle_core::{DType, Device, Tensor, D};

pub struct Config {
    pub dim: usize,
    pub n_blocks: usize,
    pub n_heads: usize,
    pub embed_dim: usize,
    pub vocab_size: usize,
    pub readout_scale: f64,
}

type W = HashMap<String, Tensor>;

/// y = x @ wᵀ for weight stored as [out, in]; x is [.., in].
fn linear(x: &Tensor, w: &Tensor) -> candle_core::Result<Tensor> {
    let dims = x.dims().to_vec();
    let inn = *dims.last().unwrap();
    let out = w.dim(0)?;
    let n: usize = dims[..dims.len() - 1].iter().product();
    let y = x.reshape((n, inn))?.matmul(&w.t()?)?; // [n, out]
    let mut shape = dims;
    *shape.last_mut().unwrap() = out;
    y.reshape(shape)
}

/// sigmoid via exp/recip — candle's Metal backend has no `sigmoid` kernel.
fn sigmoid(x: &Tensor) -> candle_core::Result<Tensor> {
    (x.neg()?.exp()? + 1.0)?.recip()
}

fn rms_norm(x: &Tensor, weight: &Tensor, eps: f64) -> candle_core::Result<Tensor> {
    let variance = x.sqr()?.mean_keepdim(D::Minus1)?;
    let normed = x.broadcast_div(&(variance + eps)?.sqrt()?)?;
    normed.broadcast_mul(weight)
}

fn layer_norm(x: &Tensor, weight: &Tensor, eps: f64) -> candle_core::Result<Tensor> {
    let mean = x.mean_keepdim(D::Minus1)?;
    let xc = x.broadcast_sub(&mean)?;
    let variance = xc.sqr()?.mean_keepdim(D::Minus1)?;
    let normed = xc.broadcast_div(&(variance + eps)?.sqrt()?)?;
    normed.broadcast_mul(weight)
}

/// gelu, tanh approximation (matches flash_attn FusedMLP "gelu_approx").
fn gelu_tanh(x: &Tensor) -> candle_core::Result<Tensor> {
    x.gelu() // candle's `gelu` is the tanh approximation; `gelu_erf` is exact
}

/// rotate_half: [-x2, x1] over the last dim split in half.
fn rotate_half(x: &Tensor) -> candle_core::Result<Tensor> {
    let d = x.dim(D::Minus1)?;
    let x1 = x.narrow(D::Minus1, 0, d / 2)?;
    let x2 = x.narrow(D::Minus1, d / 2, d / 2)?;
    Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)
}

pub struct Rotary {
    cos: Tensor, // [seq, head_dim]
    sin: Tensor,
}

impl Rotary {
    fn new(seq: usize, head_dim: usize, device: &Device) -> candle_core::Result<Self> {
        let half = head_dim / 2;
        let inv_freq: Vec<f32> = (0..half)
            .map(|i| 1f32 / 10000f32.powf((2 * i) as f32 / head_dim as f32))
            .collect();
        let inv = Tensor::from_vec(inv_freq, (1, half), device)?;
        let t: Vec<f32> = (0..seq).map(|i| i as f32).collect();
        let t = Tensor::from_vec(t, (seq, 1), device)?;
        let freqs = t.matmul(&inv)?; // [seq, half]
        let emb = Tensor::cat(&[&freqs, &freqs], D::Minus1)?; // [seq, head_dim]
        Ok(Rotary { cos: emb.cos()?, sin: emb.sin()? })
    }

    /// Apply to q or k of shape [n, seq, heads, head_dim].
    fn apply(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let (_n, seq, _h, hd) = x.dims4()?;
        let cos = self.cos.reshape((1, seq, 1, hd))?;
        let sin = self.sin.reshape((1, seq, 1, hd))?;
        x.broadcast_mul(&cos)? + rotate_half(x)?.broadcast_mul(&sin)?
    }
}

struct Block {
    rmsnorm1: Tensor,
    attn_qkv: Tensor,
    attn_out: Tensor,
    rmsnorm2: Tensor,
    fc1: Tensor,
    fc2: Tensor,
}

impl Block {
    fn load(w: &W, i: usize, dev: &Device) -> candle_core::Result<Self> {
        let g = |name: &str| -> candle_core::Result<Tensor> {
            w.get(&format!("model.blocks.{i}.{name}"))
                .ok_or_else(|| candle_core::Error::Msg(format!("missing model.blocks.{i}.{name}")))?
                .to_device(dev)
        };
        Ok(Block {
            rmsnorm1: g("rmsnorm1.weight")?,
            attn_qkv: g("attn_qkv.weight")?,
            attn_out: g("attn_out.weight")?,
            rmsnorm2: g("rmsnorm2.weight")?,
            fc1: g("mlp.fc1.weight")?,
            fc2: g("mlp.fc2.weight")?,
        })
    }

    fn forward(&self, x: &Tensor, rot: &Rotary, n_heads: usize, residual_scale: f64)
        -> candle_core::Result<Tensor> {
        let (n, seq, dim) = x.dims3()?;
        let head_dim = dim / n_heads;

        // --- attention ---
        let h = rms_norm(x, &self.rmsnorm1, 1e-5)?;
        let qkv = linear(&h, &self.attn_qkv)?; // [n, seq, 3*dim]
        let qkv = qkv.reshape((n, seq, 3, n_heads, head_dim))?;
        let q = rot.apply(&qkv.narrow(2, 0, 1)?.squeeze(2)?)?; // [n, seq, heads, hd]
        let k = rot.apply(&qkv.narrow(2, 1, 1)?.squeeze(2)?)?;
        let v = qkv.narrow(2, 2, 1)?.squeeze(2)?;
        // -> [n, heads, seq, hd]
        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;
        let scale = 1.0 / (head_dim as f64).sqrt();
        let scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?; // [n, heads, seq, seq]
        let probs = candle_nn::ops::softmax(&scores, D::Minus1)?;
        let ctx = probs.matmul(&v)?; // [n, heads, seq, hd]
        let ctx = ctx.transpose(1, 2)?.contiguous()?.reshape((n, seq, dim))?;
        let attn = linear(&ctx, &self.attn_out)?;
        let x = (x + (attn * residual_scale)?)?;

        // --- mlp ---
        let h = rms_norm(&x, &self.rmsnorm2, 1e-5)?;
        let h = linear(&gelu_tanh(&linear(&h, &self.fc1)?)?, &self.fc2)?;
        x + (h * residual_scale)?
    }
}

pub struct DiffusionModel {
    cfg: Config,
    device: Device,
    input_linear: Tensor,
    selfcond_linear: Tensor,
    gamma_linear: Tensor,
    blocks: Vec<Block>,
    output_norm: Tensor,
    output_weight: Tensor,
    output_bias: Tensor,
    embedding: Tensor, // normalized [vocab, embed_dim]
}

impl DiffusionModel {
    pub fn load(w: &W, cfg: Config, device: Device) -> candle_core::Result<Self> {
        let g = |name: &str| -> candle_core::Result<Tensor> {
            w.get(name)
                .ok_or_else(|| candle_core::Error::Msg(format!("missing {name}")))?
                .to_device(&device)
        };
        let blocks = (0..cfg.n_blocks)
            .map(|i| Block::load(w, i, &device))
            .collect::<candle_core::Result<Vec<_>>>()?;

        // EmbeddingMatrix.forward normalizes each row to unit L2 norm.
        let raw_emb = g("emb.matrix")?;
        let norm = raw_emb.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
        let embedding = raw_emb.broadcast_div(&(norm + 1e-8)?)?;

        Ok(DiffusionModel {
            input_linear: g("model.input_linear.weight")?,
            selfcond_linear: g("model.selfcond_linear.weight")?,
            gamma_linear: g("model.gamma_linear.weight")?,
            blocks,
            output_norm: g("model.output_norm.weight")?,
            output_weight: g("model.output_linear.weight")?,
            output_bias: g("model.output_linear.bias")?,
            embedding,
            cfg,
            device,
        })
    }

    pub fn embedding(&self) -> &Tensor {
        &self.embedding
    }

    pub fn embed_dim(&self) -> usize {
        self.cfg.embed_dim
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// gamma: [n] f32. z, x_selfcond: [n, seq, embed_dim] f32. Self-cond mask is
    /// all-ones (inference). Returns (logits [n,seq,V], x_reconst [n,seq,emb]).
    pub fn forward(&self, z: &Tensor, gamma: &Tensor, x_selfcond: &Tensor)
        -> candle_core::Result<(Tensor, Tensor)> {
        let (n, seq, _e) = z.dims3()?;
        let ed = self.cfg.embed_dim as f64;

        let g = gamma.reshape((n, 1, 1))?; // [n,1,1]
        let alpha_sq = sigmoid(&g.neg()?)?;
        let sigma_sq = sigmoid(&g)?;
        let alpha = alpha_sq.sqrt()?;
        let z_var = ((&alpha_sq / ed)? + &sigma_sq)?;
        let x = z.broadcast_div(&z_var.sqrt()?)?;

        let mut x = linear(&x, &self.input_linear)?;
        x = (x + linear(&(x_selfcond * ed.sqrt())?, &self.selfcond_linear)?)?;

        // gamma embedding: linspace(-5,5,32).exp() outer gamma -> [sin|cos] -> linear
        let half = 32usize;
        let lin: Vec<f32> = (0..half)
            .map(|i| (-5.0 + 10.0 * i as f32 / (half as f32 - 1.0)).exp())
            .collect();
        let lin = Tensor::from_vec(lin, (1, half), &self.device)?;
        let ge = gamma.reshape((n, 1))?.broadcast_mul(&lin)?; // [n,32]
        let ge = Tensor::cat(&[&ge.sin()?, &ge.cos()?], D::Minus1)?; // [n,64]
        let ge = linear(&ge, &self.gamma_linear)?.reshape((n, 1, self.cfg.dim))?;
        x = x.broadcast_add(&ge)?;

        let rot = Rotary::new(seq, self.cfg.dim / self.cfg.n_heads, &self.device)?;
        let residual_scale = 1.0 / (self.cfg.n_blocks as f64).sqrt();
        for b in &self.blocks {
            x = b.forward(&x, &rot, self.cfg.n_heads, residual_scale)?;
        }

        let x = layer_norm(&x, &self.output_norm, 1e-5)?;
        let x = (x * self.cfg.readout_scale)?;

        // logits = bias + x·Woutᵀ + z_scaled·embᵀ   (self-cond mask = 1)
        let mut logits = linear(&x, &self.output_weight)?;
        logits = logits.broadcast_add(&self.output_bias.reshape((1, 1, self.cfg.vocab_size))?)?;
        let z_scaled = z.broadcast_mul(&alpha.broadcast_div(&sigma_sq)?)?; // [n,seq,emb]
        logits = (logits + linear(&z_scaled, &self.embedding)?)?; // emb is [V,emb] -> linear gives [..,V]

        let probs = candle_nn::ops::softmax(&logits, D::Minus1)?;
        // x_reconst = softmax(logits) @ emb   ([n,seq,V] @ [V,emb])
        let (_n, _s, vsz) = probs.dims3()?;
        let x_reconst = probs
            .reshape((n * seq, vsz))?
            .matmul(&self.embedding)?
            .reshape((n, seq, self.cfg.embed_dim))?;
        Ok((logits, x_reconst))
    }
}

/// Load every tensor from a safetensors file as f32 on CPU.
pub fn load_safetensors(path: &str) -> candle_core::Result<W> {
    let raw = candle_core::safetensors::load(path, &Device::Cpu)?;
    let mut out = HashMap::new();
    for (k, v) in raw {
        out.insert(k, v.to_dtype(DType::F32)?);
    }
    Ok(out)
}
