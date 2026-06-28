//! VDM reverse sampler for the candle Plaid port — mirrors the ancestral
//! sampling in `sidecar/sidecar.py`. The numerically-sensitive schedule math
//! runs in plain f64; the heavy model forward runs on the model's device (f32).

use candle_core::{DType, Device, Tensor, D};

use crate::model::DiffusionModel;

fn softplus(x: f64) -> f64 {
    // numerically stable ln(1+e^x)
    if x > 20.0 { x } else { (x.exp()).ln_1p() }
}

/// Plaid's learned monotonic noise schedule γ̃(t), normalized to [0,1].
pub struct NoiseSchedule {
    w1: Vec<f64>, // softplus(W1), len 1024
    b1: Vec<f64>, // len 1024
    w2: Vec<f64>, // 0.01*softplus(W2), len 1024
}

impl NoiseSchedule {
    pub fn load(w: &std::collections::HashMap<String, Tensor>) -> candle_core::Result<Self> {
        let w1: Vec<f64> = w["sched.W1"].to_dtype(DType::F64)?.flatten_all()?.to_vec1()?;
        let b1: Vec<f64> = w["sched.b1"].to_dtype(DType::F64)?.flatten_all()?.to_vec1()?;
        let w2: Vec<f64> = w["sched.W2"].to_dtype(DType::F64)?.flatten_all()?.to_vec1()?;
        Ok(NoiseSchedule {
            w1: w1.iter().map(|&x| softplus(x)).collect(),
            b1,
            w2: w2.iter().map(|&x| 0.01 * softplus(x)).collect(),
        })
    }

    fn gamma_tilde(&self, t: f64) -> f64 {
        let mut acc = 0.0;
        for j in 0..self.w1.len() {
            let h = ((t - 0.5) * self.w1[j] + self.b1[j]).tanh();
            acc += h * self.w2[j];
        }
        acc
    }

    /// Normalized schedule in [0,1].
    pub fn at(&self, t: f64) -> f64 {
        let g0 = self.gamma_tilde(0.0);
        let g1 = self.gamma_tilde(1.0);
        (self.gamma_tilde(t) - g0) / (g1 - g0)
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// One realization's worth of f64 CPU Gaussian noise, shape [n, seq, embed_dim].
pub type NoiseFn<'a> = dyn FnMut(&[usize]) -> candle_core::Result<Tensor> + 'a;

#[allow(clippy::too_many_arguments)]
pub fn generate(
    model: &DiffusionModel,
    sched: &NoiseSchedule,
    gamma_0: f64,
    gamma_1: f64,
    n_samples: usize,
    seq_len: usize,
    steps: usize,
    score_temp: f64,
    initial_noise_scale: f64,
    ddim: bool,
    preview_every: usize,
    on_preview: &mut dyn FnMut(usize, usize, &Tensor),
    prefix_emb: Option<&Tensor>, // [n, prefix_len, embed_dim] f64 CPU
    prefix_len: usize,
    noise: &mut NoiseFn,
) -> candle_core::Result<Tensor> {
    let embed_dim = model.embed_dim();
    let dev = model.device().clone();
    let cpu = Device::Cpu;
    let shape = [n_samples, seq_len, embed_dim];

    let mut z = (noise(&shape)? * initial_noise_scale)?; // f64 CPU
    let mut x_selfcond = Tensor::zeros((n_samples, seq_len, embed_dim), DType::F32, &dev)?;

    let g = |frac: f64| gamma_0 + (gamma_1 - gamma_0) * sched.at(frac);
    let mut gamma_t_val = g(1.0);

    for i in 0..steps {
        let t = 1.0 - (i as f64) / ((steps - 1) as f64);
        let s = t - 1.0 / steps as f64;
        let gamma_s = g(s);
        gamma_t_val = g(t);
        let a2s = sigmoid(-gamma_s);
        let a2t = sigmoid(-gamma_t_val);
        let at = a2t.sqrt();
        let st = sigmoid(gamma_t_val).sqrt();

        // Prefix inpainting: clamp the leading positions to the prompt
        // embeddings re-noised to this step's level, so the model crystallizes
        // around a fixed prompt.  z_prefix = α_t·emb + σ_t·noise
        if let (Some(pe), true) = (prefix_emb, prefix_len > 0) {
            let alpha_t = a2t.sqrt();
            let sigma_t = (1.0 - a2t).sqrt();
            let pnoise = noise(&[n_samples, prefix_len, embed_dim])?;
            let clamp = ((pe * alpha_t)? + (pnoise * sigma_t)?)?;
            let rest = z.narrow(1, prefix_len, seq_len - prefix_len)?;
            z = Tensor::cat(&[&clamp, &rest], 1)?;
        }

        // model forward (device, f32)
        let zf = z.to_dtype(DType::F32)?.to_device(&dev)?;
        let gamma_vec = Tensor::from_vec(vec![gamma_t_val as f32; n_samples], n_samples, &dev)?;
        let (logits, x_reconst) = model.forward(&zf, &gamma_vec, &x_selfcond)?;
        x_selfcond = x_reconst.clone();
        if preview_every > 0 && i > 0 && i % preview_every == 0 {
            on_preview(i, steps, &logits);
        }
        let xr = x_reconst.to_device(&cpu)?.to_dtype(DType::F64)?;

        // eps = (z - at*xr)/st/score_temp ; xr = (z - st*eps)/at
        let eps = ((&z - (&xr * at)?)? * (1.0 / (st * score_temp)))?;
        let xr = ((&z - (&eps * st)?)? * (1.0 / at))?;

        if t > 0.0 {
            if ddim {
                // Deterministic DDIM step: no fresh noise, so (absent a prompt)
                // the whole trajectory is fixed by the initial latent z1.
                let alpha_s = a2s.sqrt();
                let sigma_s = (1.0 - a2s).sqrt();
                z = ((&xr * alpha_s)? + (&eps * sigma_s)?)?;
            } else {
                // Ancestral step: inject fresh Gaussian noise each step.
                let c = -((gamma_s - gamma_t_val).exp() - 1.0); // -expm1(gamma_s-gamma_t)
                let coef1 = (1.0 - c) * a2s.sqrt() / a2t.sqrt();
                let coef2 = c * a2s.sqrt();
                let coef3 = (c * (1.0 - a2s)).sqrt();
                let nz = noise(&shape)?;
                z = (((&z * coef1)? + (&xr * coef2)?)? + (&nz * coef3)?)?;
            }
        }
    }

    let zf = z.to_dtype(DType::F32)?.to_device(&dev)?;
    let gamma_vec = Tensor::from_vec(vec![gamma_t_val as f32; n_samples], n_samples, &dev)?;
    let (logits, _) = model.forward(&zf, &gamma_vec, &x_selfcond)?;
    logits.argmax(D::Minus1) // [n, seq] token ids
}
