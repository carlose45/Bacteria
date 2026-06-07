// Continuous-Time Recurrent Neural Network para neuroevolución
//
// Dinámica: τ_i · dy_i/dt = -y_i + Σ_j w[i][j]·tanh(y_j + b_j) + Σ_k w_in[i][k]·sensor_k
// Integración Euler con dt=1. Acción = argmax(W_out · tanh(y+b) + b_out).

use crate::world::{N_NEUR, N_SENS, N_ACT};
use crate::bacteria::XorShift32;

#[derive(Clone)]
pub struct Ctrnn {
    // ── Genoma (evolucionado por crossover) ──────────────────────────────────
    pub w:        [[f32; N_NEUR]; N_NEUR],   // pesos recurrentes [destino][origen]
    pub w_in:     [[f32; N_SENS]; N_NEUR],   // pesos de entrada  [neurona][sensor]
    pub bias:     [f32; N_NEUR],              // sesgos neuronales
    pub tau_raw:  [f32; N_NEUR],              // constantes de tiempo en escala log: τ = exp(tau_raw)
    pub w_out:    [[f32; N_NEUR]; N_ACT],     // lectura de salida [acción][neurona]
    pub bias_out: [f32; N_ACT],               // sesgos de salida
    // ── Estado dinámico (reiniciado en reproducción) ─────────────────────────
    pub y:        [f32; N_NEUR],
}

impl Ctrnn {
    pub fn new(rng: &mut XorShift32) -> Self {
        let mut c = Self::zeroed();
        for row in c.w.iter_mut()      { for x in row { *x = Self::rand(rng); } }
        for row in c.w_in.iter_mut()   { for x in row { *x = Self::rand(rng); } }
        for x in c.bias.iter_mut()     { *x = Self::rand(rng) * 2.0; }
        for x in c.tau_raw.iter_mut()  { *x = Self::rand(rng); }
        for row in c.w_out.iter_mut()  { for x in row { *x = Self::rand(rng); } }
        for x in c.bias_out.iter_mut() { *x = Self::rand(rng) * 0.5; }
        c
    }

    pub fn zeroed() -> Self {
        Self {
            w:        [[0.0; N_NEUR]; N_NEUR],
            w_in:     [[0.0; N_SENS]; N_NEUR],
            bias:     [0.0; N_NEUR],
            tau_raw:  [0.0; N_NEUR],
            w_out:    [[0.0; N_NEUR]; N_ACT],
            bias_out: [0.0; N_ACT],
            y:        [0.0; N_NEUR],
        }
    }

    fn rand(rng: &mut XorShift32) -> f32 {
        (rng.next_u32() as f32 / u32::MAX as f32 - 0.5) * 2.0
    }

    // Un paso de integración Euler + selección de acción con ruido exploratorio
    pub fn step(&mut self, sensors: &[f32; N_SENS], rng: &mut XorShift32) -> usize {
        // Calcular dy para cada neurona
        let mut dy = [0f32; N_NEUR];
        for i in 0..N_NEUR {
            let tau = self.tau_raw[i].exp().clamp(0.5, 8.0);
            let mut net = 0.0f32;
            for j in 0..N_NEUR {
                net += self.w[i][j] * (self.y[j] + self.bias[j]).tanh();
            }
            for k in 0..N_SENS {
                net += self.w_in[i][k] * sensors[k];
            }
            dy[i] = (1.0 / tau) * (-self.y[i] + net);
        }
        for i in 0..N_NEUR {
            self.y[i] = (self.y[i] + dy[i]).clamp(-10.0, 10.0);
        }

        // Lectura lineal → logits de acción
        let mut logits = [0f32; N_ACT];
        for a in 0..N_ACT {
            for j in 0..N_NEUR {
                logits[a] += self.w_out[a][j] * (self.y[j] + self.bias[j]).tanh();
            }
            logits[a] += self.bias_out[a];
            // Ruido exploratorio (equivalente a epsilon del transformer)
            logits[a] += (rng.next_u32() as f32 / u32::MAX as f32 - 0.5) * 0.5;
        }

        // Argmax → índice de acción (0=quedar, 1=N, 2=S, 3=E, 4=W)
        logits.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
}

// Mezcla genómica de dos CTRNN con mutación
pub fn crossover_ctrnn(a: &Ctrnn, b: &Ctrnn, pa: u32, rng: &mut XorShift32) -> Ctrnn {
    let mut c = Ctrnn::zeroed();

    for i in 0..N_NEUR {
        for j in 0..N_NEUR {
            c.w[i][j] = mutate(pick(a.w[i][j], b.w[i][j], pa, rng), rng);
        }
        for k in 0..N_SENS {
            c.w_in[i][k] = mutate(pick(a.w_in[i][k], b.w_in[i][k], pa, rng), rng);
        }
        c.bias[i]    = mutate(pick(a.bias[i],    b.bias[i],    pa, rng), rng);
        c.tau_raw[i] = mutate(pick(a.tau_raw[i], b.tau_raw[i], pa, rng), rng);
    }
    for act in 0..N_ACT {
        for j in 0..N_NEUR {
            c.w_out[act][j] = mutate(pick(a.w_out[act][j], b.w_out[act][j], pa, rng), rng);
        }
        c.bias_out[act] = mutate(pick(a.bias_out[act], b.bias_out[act], pa, rng), rng);
    }
    // y queda en cero: el estado dinámico no se hereda
    c
}

#[inline]
fn pick(av: f32, bv: f32, pa: u32, rng: &mut XorShift32) -> f32 {
    if rng.next_u32() % 100 < pa { av } else { bv }
}

#[inline]
fn mutate(v: f32, rng: &mut XorShift32) -> f32 {
    if rng.next_u32() % 100 < 4 {
        v + (rng.next_u32() as f32 / u32::MAX as f32 - 0.5) * 0.4
    } else {
        v
    }
}
