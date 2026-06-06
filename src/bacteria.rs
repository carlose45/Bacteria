use std::collections::VecDeque;
use crate::world::*;

// ── RNG ───────────────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct XorShift32 { pub state: u32 }

impl XorShift32 {
    pub fn new(seed: u32) -> Self {
        Self { state: if seed == 0 { 0x1234_5678 } else { seed } }
    }
    pub fn next_u32(&mut self) -> u32 {
        let mut x = self.state;
        x ^= x << 13; x ^= x >> 17; x ^= x << 5;
        self.state = x; x
    }
}

// ── MiniTransformer ───────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct MiniTransformer {
    pub embedding_matrix: [[f32; DIM]; 256],
    pub attention_w_q:    [[f32; DIM]; DIM],
    pub attention_w_k:    [[f32; DIM]; DIM],
    pub attention_w_v:    [[f32; DIM]; DIM],
    pub ff_w1:            [[f32; DIM]; 16],
    pub ff_w2:            [[f32; 16]; DIM],
    pub learning_rate:    f32,
}

impl MiniTransformer {
    pub fn new(rng: &mut XorShift32) -> Self {
        let mut emb = [[0f32; DIM]; 256];
        let mut wq  = [[0f32; DIM]; DIM];
        let mut wk  = [[0f32; DIM]; DIM];
        let mut wv  = [[0f32; DIM]; DIM];
        let mut ff1 = [[0f32; DIM]; 16];
        let mut ff2 = [[0f32; 16]; DIM];
        for r in emb.iter_mut() { for x in r.iter_mut() { *x = Self::rand(rng); } }
        for r in wq.iter_mut()  { for x in r.iter_mut() { *x = Self::rand(rng); } }
        for r in wk.iter_mut()  { for x in r.iter_mut() { *x = Self::rand(rng); } }
        for r in wv.iter_mut()  { for x in r.iter_mut() { *x = Self::rand(rng); } }
        for r in ff1.iter_mut() { for x in r.iter_mut() { *x = Self::rand(rng); } }
        for r in ff2.iter_mut() { for x in r.iter_mut() { *x = Self::rand(rng); } }
        Self { embedding_matrix: emb, attention_w_q: wq, attention_w_k: wk,
               attention_w_v: wv, ff_w1: ff1, ff_w2: ff2, learning_rate: 0.001 }
    }

    fn rand(rng: &mut XorShift32) -> f32 {
        (rng.next_u32() as f32 / 4294967296.0 - 0.5) * 1.0
    }

    pub fn forward(&self, value: u8, position: usize, state: &[f32; DIM],
                   q_grad: &[f32; 8], rng: &mut XorShift32, epsilon: f32,
    ) -> ([u8; DIM], [f32; DIM]) {
        let ve = self.embedding_matrix[value as usize];
        let pe = self.embedding_matrix[position % 256];
        let mut emb = [0f32; DIM];
        for i in 0..8 { emb[i] = ve[i] + pe[i] + state[i] * 0.1 + q_grad[i] * 0.5; }
        emb[8] = ve[8] + pe[8] + state[8] * 0.1;
        emb[9] = ve[9] + pe[9] + state[9] * 0.1;
        let attended = self.multihead_attention(&emb, state);
        let logits   = self.feedforward(&attended);
        let mut bits = [0u8; DIM];
        let explore  = (rng.next_u32() as f32 / 4294967296.0) < epsilon;
        for i in 0..DIM {
            bits[i] = if explore {
                (rng.next_u32() & 1) as u8
            } else {
                let noise = (rng.next_u32() as f32 / 4294967296.0 - 0.5) * 0.2;
                if logits[i] + noise > 0.0 { 1 } else { 0 }
            };
        }
        (bits, logits)
    }

    pub fn learn(&mut self, value: u8, position: usize, state: &[f32; DIM],
                 q_grad: &[f32; 8], logits: &[f32; DIM], reward: f32) {
        let ve = self.embedding_matrix[value as usize];
        let pe = self.embedding_matrix[position % 256];
        let mut emb = [0f32; DIM];
        for i in 0..8 { emb[i] = ve[i] + pe[i] + state[i] * 0.1 + q_grad[i] * 0.5; }
        emb[8] = ve[8] + pe[8] + state[8] * 0.1;
        emb[9] = ve[9] + pe[9] + state[9] * 0.1;
        let attended = self.multihead_attention(&emb, state);
        let hidden   = self.mm16(&self.ff_w1, &attended);
        let activated: [f32; 16] = hidden.map(|x| x.tanh());
        let lr = self.learning_rate;
        for i in 0..DIM {
            let d = if logits[i] > 0.0 { 1.0 } else { -1.0 };
            for j in 0..16 { self.ff_w2[i][j] += lr * reward * d * activated[j]; }
        }
        for i in 0..16 {
            let d = if hidden[i] > 0.0 { 1.0 } else { -1.0 };
            for j in 0..DIM { self.ff_w1[i][j] += lr * reward * d * attended[j]; }
        }
    }

    fn multihead_attention(&self, emb: &[f32; DIM], state: &[f32; DIM]) -> [f32; DIM] {
        let q1 = self.mm_dim(&self.attention_w_q, emb);
        let k1 = self.mm_dim(&self.attention_w_k, emb);
        let v1 = self.mm_dim(&self.attention_w_v, emb);
        let q2 = self.mm_dim(&self.attention_w_q, state);
        let k2 = self.mm_dim(&self.attention_w_k, state);
        let v2 = self.mm_dim(&self.attention_w_v, state);
        let s1 = self.dot(&q1, &k1) * 0.1;
        let s2 = self.dot(&q2, &k2) * 0.1;
        let mut out = [0f32; DIM];
        for i in 0..DIM { out[i] = (v1[i] * s1.tanh() + v2[i] * s2.tanh()) * 0.5; }
        out
    }

    fn feedforward(&self, input: &[f32; DIM]) -> [f32; DIM] {
        let hidden    = self.mm16(&self.ff_w1, input);
        let activated: [f32; 16] = hidden.map(|x| x.tanh());
        let mut out = [0f32; DIM];
        for i in 0..DIM { for j in 0..16 { out[i] += self.ff_w2[i][j] * activated[j]; } }
        out
    }

    fn mm_dim(&self, m: &[[f32; DIM]; DIM], v: &[f32; DIM]) -> [f32; DIM] {
        let mut r = [0f32; DIM];
        for i in 0..DIM { for j in 0..DIM { r[i] += m[i][j] * v[j]; } }
        r
    }
    fn mm16(&self, m: &[[f32; DIM]; 16], v: &[f32; DIM]) -> [f32; 16] {
        let mut r = [0f32; 16];
        for i in 0..16 { for j in 0..DIM { r[i] += m[i][j] * v[j]; } }
        r
    }
    fn dot(&self, a: &[f32; DIM], b: &[f32; DIM]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }
}

// ── Bacteria ──────────────────────────────────────────────────────────────────

pub struct Bacteria {
    pub position:    usize,
    pub state:       [f32; DIM],
    pub transformer: MiniTransformer,
    pub visits:      Box<[f32; GRID_SIZE]>,  // en heap para no saturar el stack
    pub recent:      VecDeque<usize>,
    pub rewards:     VecDeque<f32>,
    pub age:         u32,
    pub cooldown:    u32,
    pub rng:         XorShift32,
    pub energy:      f32,
}

impl Bacteria {
    pub fn new(rng: &mut XorShift32) -> Self {
        Self {
            position:    0,
            state:       [0.0; DIM],
            transformer: MiniTransformer::new(rng),
            visits:      Box::new([0.0; GRID_SIZE]),
            recent:      VecDeque::with_capacity(500),
            rewards:     VecDeque::with_capacity(200),
            age:         0,
            cooldown:    0,
            rng:         XorShift32::new(rng.next_u32()),
            energy:      MAX_BACTERIA_ENERGY * 0.5,
        }
    }

    pub fn fitness(&self) -> f32 {
        if self.rewards.is_empty() { return 0.0; }
        self.rewards.iter().sum::<f32>() / self.rewards.len() as f32
    }

    pub fn recent_diversity(&self) -> f32 {
        if self.recent.is_empty() { return 0.0; }
        let mut seen = [false; GRID_SIZE];
        for &p in &self.recent { seen[p] = true; }
        seen.iter().filter(|&&b| b).count() as f32 / self.recent.len() as f32
    }

    pub fn coverage(&self) -> f32 {
        self.visits.iter().filter(|&&v| v > 1.0).count() as f32 / GRID_SIZE as f32
    }

    pub fn is_starving(&self) -> bool {
        self.energy <= 0.0 || (self.age > STARVATION_AGE && self.recent_diversity() < 0.05)
    }

    pub fn hunger(&self) -> f32 {
        (1.0 - self.energy / MAX_BACTERIA_ENERGY).clamp(0.0, 1.0)
    }

    pub fn step(&mut self, memory: &[u8], quorum: &[f32], food: &[f32], crowding: &[u8]) -> (usize, u8, f32) {
        let value_read = memory[self.position];
        let prev_state = self.state;
        let q_grad     = quorum_neighborhood(self.position, quorum);
        let (bits, logits) = self.transformer.forward(
            value_read, self.position, &prev_state, &q_grad, &mut self.rng, 0.20);

        let new_pos = bits.iter().enumerate()
            .fold(0usize, |acc, (i, &b)| acc + ((b as usize) << i))
            % GRID_SIZE;

        for i in 0..DIM { self.state[i] = if bits[i] == 1 { 0.5 } else { -0.5 }; }

        for v in self.visits.iter_mut() { *v *= 0.9990; }
        let novelty     = 1.0 / (1.0 + self.visits[new_pos]);
        let memory_diff = ((memory[new_pos] as i32 - memory[self.position] as i32).abs() as f32) / 255.0;
        let recency     = self.recent.iter().filter(|&&p| p == new_pos).count() as f32;

        let q     = quorum[new_pos];
        let q_eff = q.min(20.0);
        let colony_bonus = (q_eff / (QUORUM_THRESH + q_eff)) * 1.2;

        let recency_weight = if q > QUORUM_SAT_THRESH { 0.15 }
                             else if q > QUORUM_THRESH { 0.005 }
                             else                      { 0.3   };
        let recency_penalty = recency_weight * recency;

        // Hambre: gana energía en celda con comida, pierde por metabolismo
        let food_val = food[new_pos].max(0.0);
        if food_val > FOOD_SIGNAL_THR {
            self.energy = (self.energy + FOOD_ENERGY_GAIN).min(MAX_BACTERIA_ENERGY);
        } else {
            self.energy = (self.energy - METABOLISM_RATE).max(0.0);
        }
        let h = self.hunger();  // 0.0 = llena, 1.0 = hambrienta

        // Comida vale más cuanto más hambrienta está
        let food_bonus = (food_val / (FOOD_SIGNAL_THR + food_val)) * (1.5 + h * 2.5);

        // Crowding más costoso cuanto más hambrienta (compiten por recursos escasos)
        let crowd = crowding[new_pos] as f32;
        let crowding_penalty = ((crowd - 1.0).max(0.0) / 4.0).min(1.0) * (2.0 + h * 2.0);

        let reward = if new_pos == self.position { -0.8 }
                     else { novelty + 0.1 * memory_diff - recency_penalty + colony_bonus + food_bonus - crowding_penalty };

        self.transformer.learn(value_read, self.position, &prev_state, &q_grad, &logits, reward);

        self.recent.push_back(self.position);
        if self.recent.len() > 500 { self.recent.pop_front(); }

        self.position = new_pos;
        self.visits[new_pos] += 1.0;

        self.rewards.push_back(reward);
        if self.rewards.len() > 200 { self.rewards.pop_front(); }
        self.age += 1;
        if self.cooldown > 0 { self.cooldown -= 1; }

        let store = (q * 20.0).min(255.0) as u8;
        (new_pos, store, reward)
    }
}

// ── Crossover ─────────────────────────────────────────────────────────────────

pub fn crossover(a: &Bacteria, b: &Bacteria, rng: &mut XorShift32) -> Bacteria {
    let fa = a.recent_diversity().max(0.01);
    let fb = b.recent_diversity().max(0.01);
    let pa = (fa / (fa + fb) * 100.0) as u32;

    let mut emb = [[0f32; DIM]; 256];
    let mut wq  = [[0f32; DIM]; DIM];
    let mut wk  = [[0f32; DIM]; DIM];
    let mut wv  = [[0f32; DIM]; DIM];
    let mut ff1 = [[0f32; DIM]; 16];
    let mut ff2 = [[0f32; 16]; DIM];

    for i in 0..256 { for j in 0..DIM {
        emb[i][j] = if rng.next_u32() % 100 < pa
            { a.transformer.embedding_matrix[i][j] } else { b.transformer.embedding_matrix[i][j] };
    }}
    for i in 0..DIM { for j in 0..DIM {
        wq[i][j] = if rng.next_u32() % 100 < pa { a.transformer.attention_w_q[i][j] } else { b.transformer.attention_w_q[i][j] };
        wk[i][j] = if rng.next_u32() % 100 < pa { a.transformer.attention_w_k[i][j] } else { b.transformer.attention_w_k[i][j] };
        wv[i][j] = if rng.next_u32() % 100 < pa { a.transformer.attention_w_v[i][j] } else { b.transformer.attention_w_v[i][j] };
    }}
    for i in 0..16 { for j in 0..DIM {
        ff1[i][j] = if rng.next_u32() % 100 < pa { a.transformer.ff_w1[i][j] } else { b.transformer.ff_w1[i][j] };
    }}
    for i in 0..DIM { for j in 0..16 {
        ff2[i][j] = if rng.next_u32() % 100 < pa { a.transformer.ff_w2[i][j] } else { b.transformer.ff_w2[i][j] };
    }}

    let t = MiniTransformer {
        embedding_matrix: emb, attention_w_q: wq, attention_w_k: wk,
        attention_w_v: wv, ff_w1: ff1, ff_w2: ff2,
        learning_rate: (a.transformer.learning_rate + b.transformer.learning_rate) / 2.0,
    };
    Bacteria {
        position:    if rng.next_u32() % 2 == 0 { a.position } else { b.position },
        state:       [0.0; DIM],
        transformer: t,
        visits:      Box::new([0.0; GRID_SIZE]),
        recent:      VecDeque::with_capacity(50),
        rewards:     VecDeque::with_capacity(200),
        age:         0,
        cooldown:    COOLDOWN,
        rng:         XorShift32::new(rng.next_u32()),
        energy:      (a.energy + b.energy) * 0.4,  // hereda 40% del promedio parental
    }
}
