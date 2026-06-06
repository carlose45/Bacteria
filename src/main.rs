use std::collections::VecDeque;
use std::fs::File;
use std::io::Write;
use rayon::prelude::*;

// ── Geometría del mundo ───────────────────────────────────────────────────────
const GRID_SIDE: usize = 32;                  // 32×32
const GRID_SIZE: usize = GRID_SIDE * GRID_SIDE; // 1024 celdas
const DIM:       usize = 10;                  // bits del transformer = log2(1024)

// ── Constantes bacteria ───────────────────────────────────────────────────────
const MAX_POP:        usize = 64;
const MAX_AGE:        u32   = 80_000;
const REPRODUCE_PROB: u32   = 30;
const COOLDOWN:       u32   = 2000;
const STARVATION_AGE: u32   = 5000;
const QUORUM_DECAY:        f32   = 0.985;
const QUORUM_THRESH:       f32   = 5.0;
const QUORUM_SAT_THRESH:   f32   = 180.0; // núcleo saturado → presión de salida
const QUORUM_EVENT_THRESH: f32   = 8.0;
const QUORUM_DEPOSIT:      f32   = 12.0 / MAX_POP as f32; // escala con población
const MAX_SNAPS:           usize = 30;
const SNAP_INTERVAL_SECS:  u64   = 15;

// ── Constantes comida ─────────────────────────────────────────────────────────
const MAX_FOOD:         usize = 15;
const FOOD_REGEN:       f32   = 0.3;
const FOOD_MAX_ENERGY:  f32   = 400.0;
const FOOD_EATEN:       f32   = 15.0 * 12.0 / MAX_POP as f32;
const FOOD_SIGNAL:      f32   = 4.0;
const FOOD_SIGNAL_THR:  f32   = 3.0;
const FOOD_DECAY:       f32   = 0.990;
const FOOD_FLEE_THRESH: f32   = 100.0; // huye solo de colonias grandes; forrajeros pueden comerla

// ── XorShift32 ───────────────────────────────────────────────────────────────

struct XorShift32 { state: u32 }

impl XorShift32 {
    fn new(seed: u32) -> Self {
        Self { state: if seed == 0 { 0x1234_5678 } else { seed } }
    }
    fn next_u32(&mut self) -> u32 {
        let mut x = self.state;
        x ^= x << 13; x ^= x >> 17; x ^= x << 5;
        self.state = x; x
    }
}

// ── Quorum neighborhood (toroidal GRID_SIDE×GRID_SIDE) ───────────────────────

fn quorum_neighborhood(pos: usize, quorum: &[f32]) -> [f32; 8] {
    let row = pos / GRID_SIDE;
    let col = pos % GRID_SIDE;
    let s   = GRID_SIDE;
    let nbrs = [
        ((row + s - 1) % s) * s + col,
        ((row + s - 1) % s) * s + (col + 1) % s,
        row * s             + (col + 1) % s,
        ((row + 1) % s)     * s + (col + 1) % s,
        ((row + 1) % s)     * s + col,
        ((row + 1) % s)     * s + (col + s - 1) % s,
        row * s             + (col + s - 1) % s,
        ((row + s - 1) % s) * s + (col + s - 1) % s,
    ];
    let q0 = quorum[pos];
    let mut g = [0f32; 8];
    for i in 0..8 { g[i] = (quorum[nbrs[i]] - q0).tanh(); }
    g
}

// ── MiniTransformer (DIM=10 bits → 1024 posiciones) ──────────────────────────

struct MiniTransformer {
    embedding_matrix: [[f32; DIM]; 256], // posición usa % 256 (encoding periódico)
    attention_w_q:    [[f32; DIM]; DIM],
    attention_w_k:    [[f32; DIM]; DIM],
    attention_w_v:    [[f32; DIM]; DIM],
    ff_w1:            [[f32; DIM]; 16],
    ff_w2:            [[f32; 16]; DIM],
    learning_rate:    f32,
}

impl MiniTransformer {
    fn new(rng: &mut XorShift32) -> Self {
        let mut emb     = [[0f32; DIM]; 256];
        let mut wq      = [[0f32; DIM]; DIM];
        let mut wk      = [[0f32; DIM]; DIM];
        let mut wv      = [[0f32; DIM]; DIM];
        let mut ff1     = [[0f32; DIM]; 16];
        let mut ff2     = [[0f32; 16]; DIM];
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

    fn forward(&self, value: u8, position: usize, state: &[f32; DIM],
               q_grad: &[f32; 8], rng: &mut XorShift32, epsilon: f32
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
        let explore = (rng.next_u32() as f32 / 4294967296.0) < epsilon;
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

    fn learn(&mut self, value: u8, position: usize, state: &[f32; DIM],
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

// ── Bacteria ─────────────────────────────────────────────────────────────────

struct Bacteria {
    position:    usize,
    state:       [f32; DIM],
    transformer: MiniTransformer,
    visits:      [f32; GRID_SIZE],
    recent:      VecDeque<usize>,
    rewards:     VecDeque<f32>,
    age:         u32,
    cooldown:    u32,
    rng:         XorShift32,
}

impl Bacteria {
    fn new(rng: &mut XorShift32) -> Self {
        Self {
            position:    0,
            state:       [0.0; DIM],
            transformer: MiniTransformer::new(rng),
            visits:      [0.0; GRID_SIZE],
            recent:      VecDeque::with_capacity(500),
            rewards:     VecDeque::with_capacity(200),
            age:         0,
            cooldown:    0,
            rng:         XorShift32::new(rng.next_u32()),
        }
    }

    fn fitness(&self) -> f32 {
        if self.rewards.is_empty() { return 0.0; }
        self.rewards.iter().sum::<f32>() / self.rewards.len() as f32
    }

    fn recent_diversity(&self) -> f32 {
        if self.recent.is_empty() { return 0.0; }
        let mut seen = [false; GRID_SIZE];
        for &p in &self.recent { seen[p] = true; }
        seen.iter().filter(|&&b| b).count() as f32 / self.recent.len() as f32
    }

    fn coverage(&self) -> f32 {
        let covered = self.visits[..GRID_SIZE].iter().filter(|&&v| v > 1.0).count();
        covered as f32 / GRID_SIZE as f32
    }

    fn step(&mut self, memory: &[u8], quorum: &[f32], food: &[f32]) -> (usize, u8, f32) {
        let value_read = memory[self.position];
        let prev_state = self.state;
        let q_grad = quorum_neighborhood(self.position, quorum);
        let (bits, logits) = self.transformer.forward(
            value_read, self.position, &prev_state, &q_grad, &mut self.rng, 0.20);

        let new_pos = bits.iter().enumerate()
            .fold(0usize, |acc, (i, &b)| acc + ((b as usize) << i))
            % GRID_SIZE;

        for i in 0..DIM { self.state[i] = if bits[i] == 1 { 0.5 } else { -0.5 }; }

        // Olvido más rápido para mantener novelty viva con 64 agentes
        for v in self.visits.iter_mut() { *v *= 0.9990; }
        let novelty     = 1.0 / (1.0 + self.visits[new_pos]);
        let memory_diff = ((memory[new_pos] as i32 - memory[self.position] as i32).abs() as f32) / 255.0;
        let recency     = self.recent.iter().filter(|&&p| p == new_pos).count() as f32;

        let q     = quorum[new_pos];
        let q_eff = q.min(20.0);
        let colony_bonus = (q_eff / (QUORUM_THRESH + q_eff)) * 1.2;

        let recency_weight = if q > QUORUM_SAT_THRESH {
            0.15  // núcleo saturado: presión de salida hacia el borde
        } else if q > QUORUM_THRESH {
            0.005 // colonia normal: quieren quedarse
        } else {
            0.3   // exterior: exploración activa
        };
        let recency_penalty = recency_weight * recency;

        let food_val   = food[new_pos].max(0.0);
        let food_bonus = (food_val / (FOOD_SIGNAL_THR + food_val)) * 1.5;

        let reward = if new_pos == self.position {
            -0.8
        } else {
            novelty + 0.1 * memory_diff - recency_penalty + colony_bonus + food_bonus
        };

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

    fn is_starving(&self) -> bool {
        self.age > STARVATION_AGE && self.recent_diversity() < 0.05
    }
}

// ── Comida ────────────────────────────────────────────────────────────────────

struct FoodAgent {
    position: usize,
    energy:   f32,
    age:      u32,
    recent:   VecDeque<usize>,
}

impl FoodAgent {
    fn new(rng: &mut XorShift32) -> Self {
        Self {
            position: (rng.next_u32() as usize) % GRID_SIZE,
            energy:   50.0,
            age:      0,
            recent:   VecDeque::with_capacity(200),
        }
    }

    fn diversity(&self) -> f32 {
        if self.recent.is_empty() { return 0.0; }
        let mut seen = [false; GRID_SIZE];
        for &p in &self.recent { seen[p] = true; }
        seen.iter().filter(|&&b| b).count() as f32 / self.recent.len() as f32
    }

    fn step(&mut self, quorum: &[f32]) -> usize {
        self.energy = (self.energy + FOOD_REGEN).min(FOOD_MAX_ENERGY);
        self.age += 1;

        if quorum[self.position] > FOOD_FLEE_THRESH {
            let row = self.position / GRID_SIDE;
            let col = self.position % GRID_SIDE;
            let s   = GRID_SIDE;
            let candidates = [
                ((row + s - 1) % s) * s + col,
                ((row + 1)     % s) * s + col,
                row * s + (col + 1) % s,
                row * s + (col + s - 1) % s,
            ];
            self.position = *candidates.iter()
                .min_by(|&&a, &&b| quorum[a].partial_cmp(&quorum[b]).unwrap())
                .unwrap();
        }

        self.recent.push_back(self.position);
        if self.recent.len() > 200 { self.recent.pop_front(); }
        self.position
    }
}

// ── Crossover ────────────────────────────────────────────────────────────────

fn crossover(a: &Bacteria, b: &Bacteria, rng: &mut XorShift32) -> Bacteria {
    let fa = a.recent_diversity().max(0.01);
    let fb = b.recent_diversity().max(0.01);
    let pa = (fa / (fa + fb) * 100.0) as u32;

    let mut emb  = [[0f32; DIM]; 256];
    let mut wq   = [[0f32; DIM]; DIM];
    let mut wk   = [[0f32; DIM]; DIM];
    let mut wv   = [[0f32; DIM]; DIM];
    let mut ff1  = [[0f32; DIM]; 16];
    let mut ff2  = [[0f32; 16]; DIM];

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
        visits:      [0.0; GRID_SIZE],
        recent:      VecDeque::with_capacity(50),
        rewards:     VecDeque::with_capacity(200),
        age:         0,
        cooldown:    COOLDOWN,
        rng:         XorShift32::new(rng.next_u32()),
    }
}

// ── Visualización ────────────────────────────────────────────────────────────

fn combined_visits(population: &[Bacteria]) -> Vec<f32> {
    let mut cv = vec![0f32; GRID_SIZE];
    for b in population { for i in 0..GRID_SIZE { cv[i] += b.visits[i]; } }
    cv
}

fn colony_zones(quorum: &[f32]) -> usize {
    quorum.iter().filter(|&&q| q > QUORUM_THRESH).count()
}

fn cell_char(pos: usize, population: &[Bacteria], food_agents: &[FoodAgent],
             cv: &[f32], quorum: &[f32], food: &[f32]) -> String {
    if let Some((idx, _)) = population.iter().enumerate().find(|(_, b)| b.position == pos) {
        return format!("{:<3}", format!("{:X}", idx));
    }
    if food_agents.iter().any(|fa| fa.position == pos) { return "F  ".into(); }
    if quorum[pos] > 8.0                               { return "@  ".into(); }
    if quorum[pos] > QUORUM_THRESH                     { return "*  ".into(); }
    if food[pos] > FOOD_SIGNAL_THR                     { return "f  ".into(); }
    if cv[pos] > 100.0                                 { return "+  ".into(); }
    if cv[pos] > 20.0                                  { return ".  ".into(); }
    "   ".into()
}

fn cell_char_color(pos: usize, population: &[Bacteria], food_agents: &[FoodAgent],
                   cv: &[f32], quorum: &[f32], food: &[f32]) -> String {
    if let Some((idx, _)) = population.iter().enumerate().find(|(_, b)| b.position == pos) {
        return format!("\x1B[1;92m{:<3}\x1B[0m", format!("{:X}", idx));
    }
    if food_agents.iter().any(|fa| fa.position == pos) { return "\x1B[1;33mF  \x1B[0m".into(); }
    if quorum[pos] > 8.0          { return "\x1B[1;35m@  \x1B[0m".into(); }
    if quorum[pos] > QUORUM_THRESH { return "\x1B[1;36m*  \x1B[0m".into(); }
    if food[pos] > FOOD_SIGNAL_THR { return "\x1B[1;32mf  \x1B[0m".into(); }
    if cv[pos] > 100.0             { return "\x1B[1;33m+  \x1B[0m".into(); }
    if cv[pos] > 20.0              { return "\x1B[37m.  \x1B[0m".into(); }
    "   ".into()
}

fn print_map(population: &[Bacteria], food_agents: &[FoodAgent],
             quorum: &[f32], food: &[f32], step: u32) {
    let cv    = combined_visits(population);
    let zones = colony_zones(quorum);
    let q_max = quorum.iter().cloned().fold(0f32, f32::max);
    let f_max = food.iter().cloned().fold(0f32, f32::max);
    print!("\x1B[2J\x1B[H");
    println!("  paso={:>9}  pop={}  food={}  colonias={}  q_max={:.1}  f_max={:.1}",
        step, population.len(), food_agents.len(), zones, q_max, f_max);
    for (i, b) in population.iter().enumerate() {
        print!("  {:X}:{:>3} fit={:+.2}", i, b.position, b.fitness());
        if (i + 1) % 4 == 0 { println!(); }
    }
    println!();
    print!("     ");
    for col in 0..GRID_SIDE { print!("{:<3}", format!("{:X}", col)); }
    println!();
    for row in 0..GRID_SIDE {
        print!(" {:2X}  ", row);
        for col in 0..GRID_SIDE {
            print!("{}", cell_char_color(row * GRID_SIDE + col, population, food_agents, &cv, quorum, food));
        }
        println!();
    }
    println!("\n  0-F.. bacteria  F comida  @ núcleo  * colonia  f señal  + tibio  . frío");
}

fn save_snapshot(population: &[Bacteria], food_agents: &[FoodAgent],
                 quorum: &[f32], food: &[f32], step: u32, index: usize, trigger: &str) {
    let cv    = combined_visits(population);
    let zones = colony_zones(quorum);
    let q_max = quorum.iter().cloned().fold(0f32, f32::max);
    let f_max = food.iter().cloned().fold(0f32, f32::max);
    if let Ok(mut f_out) = File::create(format!("snapshot_{:02}.txt", index)) {
        let _ = writeln!(f_out,
            "Snap {:02} | {} | paso={} | pop={} | food={} | colonias={} | q_max={:.1} | f_max={:.1}",
            index, trigger, step, population.len(), food_agents.len(), zones, q_max, f_max);
        for (i, b) in population.iter().enumerate() {
            let _ = writeln!(f_out,
                "  {:X}  pos={:>4}  div={:>3.0}%  cov={:>3.0}%  fit={:+.3}  age={}",
                i, b.position, b.recent_diversity()*100.0, b.coverage()*100.0, b.fitness(), b.age);
        }
        for (i, fa) in food_agents.iter().enumerate() {
            let _ = writeln!(f_out,
                "  F{:X}  pos={:>4}  div={:>3.0}%  energy={:>6.1}  age={}",
                i, fa.position, fa.diversity()*100.0, fa.energy, fa.age);
        }
        let _ = writeln!(f_out);
        let mut header = String::from("      ");
        for col in 0..GRID_SIDE { header.push_str(&format!("{:<3}", format!("{:X}", col))); }
        let _ = writeln!(f_out, "{}", header);
        for row in 0..GRID_SIDE {
            let mut line = format!(" {:2X}   ", row);
            for col in 0..GRID_SIDE {
                let pos = row * GRID_SIDE + col;
                if let Some((idx, _)) = population.iter().enumerate().find(|(_, b)| b.position == pos) {
                    line.push_str(&format!("{:<3}", format!("{:X}", idx)));
                } else {
                    line.push_str(cell_char(pos, population, food_agents, &cv, quorum, food).as_str());
                }
            }
            let _ = writeln!(f_out, "{}", line);
        }
    }
}

fn save_history(history: &VecDeque<String>) {
    if let Ok(mut f) = File::create("ultimo_historico.txt") {
        for line in history { let _ = writeln!(f, "{}", line); }
    }
}

// ── Main ─────────────────────────────────────────────────────────────────────

fn main() {
    let seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos() as u32;
    let mut rng = XorShift32::new(seed);

    let mut memory: Vec<u8>  = (0..GRID_SIZE).map(|i| ((i * 37 + 13) % 256) as u8).collect();
    let mut quorum: Vec<f32> = vec![0.0; GRID_SIZE];
    let mut food:   Vec<f32> = vec![0.0; GRID_SIZE];

    let mut population: Vec<Bacteria> = {
        let step = GRID_SIZE / 4;
        (0..4).map(|i| {
            let mut b = Bacteria::new(&mut rng);
            b.position = i * step;
            b
        }).collect()
    };

    let mut food_agents: Vec<FoodAgent> = (0..(MAX_FOOD / 3).max(3))
        .map(|_| FoodAgent::new(&mut rng))
        .collect();

    let mut history: VecDeque<String> = VecDeque::with_capacity(100);
    let mut last_snapshot   = std::time::Instant::now();
    let mut last_print      = std::time::Instant::now();
    let mut last_event_snap = std::time::Instant::now();
    let mut snapshot_count  = 0usize;
    let mut q_prev_above    = false;
    let mut step = 0u32;

    loop {
        let mem_snap  = memory.clone();
        let qm_snap   = quorum.clone();
        let food_snap = food.clone();

        let updates: Vec<(usize, u8)> = if population.len() >= 48 {
            population.par_iter_mut()
                .map(|b| { let (p, s, _) = b.step(&mem_snap, &qm_snap, &food_snap); (p, s) })
                .collect()
        } else {
            population.iter_mut()
                .map(|b| { let (p, s, _) = b.step(&mem_snap, &qm_snap, &food_snap); (p, s) })
                .collect()
        };

        for &(new_pos, store) in &updates {
            memory[new_pos] = store;
            let fv = food_snap[new_pos];
            let food_factor = 1.0 + fv / (FOOD_SIGNAL_THR + fv); // 1x sin comida, 2x con comida saturada
            quorum[new_pos] += QUORUM_DEPOSIT * food_factor;
            let row = new_pos / GRID_SIDE;
            let col = new_pos % GRID_SIDE;
            let s   = GRID_SIDE;
            for &n in &[
                ((row + s - 1) % s) * s + col,
                ((row + 1)     % s) * s + col,
                row * s + (col + 1) % s,
                row * s + (col + s - 1) % s,
            ] { quorum[n] += QUORUM_DEPOSIT * 0.3; }
        }
        for q in quorum.iter_mut() { *q *= QUORUM_DECAY; }

        // Agentes de comida: expanden parche, huyen si amenazadas
        for fa in food_agents.iter_mut() {
            let pos = fa.step(&qm_snap);
            let row = pos / GRID_SIDE;
            let col = pos % GRID_SIDE;
            let s   = GRID_SIDE;
            let cardinals = [
                ((row + s - 1) % s) * s + col,
                ((row + 1)     % s) * s + col,
                row * s + (col + 1) % s,
                row * s + (col + s - 1) % s,
            ];
            // Presión de quórum: siente el máximo entre su celda y sus vecinos cardinales
            let q_max = cardinals.iter().fold(qm_snap[pos], |acc, &n| acc.max(qm_snap[n]));
            let q_pressure = (q_max / QUORUM_SAT_THRESH).min(1.0);
            fa.energy -= q_pressure * FOOD_EATEN;
            food[pos] += FOOD_SIGNAL;
            quorum[pos] += QUORUM_DEPOSIT * 1.0; // feromona más fuerte: la comida llama a las bacterias
            for &n in &cardinals { food[n] += FOOD_SIGNAL * 0.5; }
        }

        for &(new_pos, _) in &updates {
            if food[new_pos] > 0.0 {
                food[new_pos] = (food[new_pos] - FOOD_EATEN).max(0.0);
                for fa in food_agents.iter_mut() {
                    if fa.position == new_pos { fa.energy -= FOOD_EATEN; }
                }
            }
        }
        for fv in food.iter_mut() { *fv *= FOOD_DECAY; }

        food_agents.retain(|fa| fa.energy > 0.0);
        if food_agents.is_empty() { food_agents.push(FoodAgent::new(&mut rng)); }

        let current_n = food_agents.len();
        let mut new_food: Vec<FoodAgent> = vec![];
        for fa in food_agents.iter_mut() {
            if fa.energy > 200.0 && current_n + new_food.len() < MAX_FOOD {
                fa.energy /= 2.0;
                let row = fa.position / GRID_SIDE;
                let col = fa.position % GRID_SIDE;
                let s   = GRID_SIDE;
                let candidates = [
                    ((row + s - 1) % s) * s + col,
                    ((row + 1)     % s) * s + col,
                    row * s + (col + 1) % s,
                    row * s + (col + s - 1) % s,
                ];
                let child_pos = candidates[rng.next_u32() as usize % 4];
                new_food.push(FoodAgent { position: child_pos, energy: fa.energy,
                                          age: 0, recent: VecDeque::with_capacity(200) });
            }
        }
        food_agents.extend(new_food);

        // Reproducción y muerte de bacterias
        let mut offspring: Vec<Bacteria> = vec![];
        for i in 0..population.len() {
            for j in (i + 1)..population.len() {
                if population[i].position == population[j].position
                    && population[i].cooldown == 0 && population[j].cooldown == 0
                    && rng.next_u32() % 100 < REPRODUCE_PROB
                {
                    offspring.push(crossover(&population[i], &population[j], &mut rng));
                }
            }
        }
        for child in offspring {
            if population.len() >= MAX_POP {
                let idx = population.iter().enumerate()
                    .min_by(|a, b| a.1.recent_diversity().partial_cmp(&b.1.recent_diversity()).unwrap())
                    .map(|(i, _)| i).unwrap();
                population.remove(idx);
            }
            population.push(child);
        }
        if population.len() > 1 {
            population.retain(|b| b.age <= MAX_AGE && !b.is_starving());
        }
        if population.is_empty() { population.push(Bacteria::new(&mut rng)); }

        if step % 10 == 0 {
            let q_now = quorum.iter().cloned().fold(0f32, f32::max);
            let f_now = food.iter().cloned().fold(0f32, f32::max);
            let zones = colony_zones(&quorum);
            let avg_fit = population.iter().map(|b| b.fitness()).sum::<f32>()
                          / population.len() as f32;
            history.push_back(format!(
                "paso={:>9}  pop={}  food={}  col={}  q={:.1}  f={:.1}  fit={:+.3}",
                step, population.len(), food_agents.len(), zones, q_now, f_now, avg_fit
            ));
            if history.len() > 100 { history.pop_front(); }
        }

        if last_print.elapsed().as_millis() >= 500 {
            print_map(&population, &food_agents, &quorum, &food, step);
            last_print = std::time::Instant::now();
        }

        let q_now   = quorum.iter().cloned().fold(0f32, f32::max);
        let q_above = q_now > QUORUM_EVENT_THRESH;
        if q_above && !q_prev_above && snapshot_count < MAX_SNAPS
            && last_event_snap.elapsed().as_secs() >= 5
        {
            snapshot_count += 1;
            save_snapshot(&population, &food_agents, &quorum, &food, step, snapshot_count, "EVENTO");
            save_history(&history);
            last_event_snap = std::time::Instant::now();
        }
        q_prev_above = q_above;

        if snapshot_count < MAX_SNAPS && last_snapshot.elapsed().as_secs() >= SNAP_INTERVAL_SECS {
            snapshot_count += 1;
            save_snapshot(&population, &food_agents, &quorum, &food, step, snapshot_count, "TIME");
            save_history(&history);
            last_snapshot = std::time::Instant::now();
        }

        step += 1;
    }
}
