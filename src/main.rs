use std::collections::VecDeque;
use std::fs::File;
use std::io::Write;
use rayon::prelude::*;

// ── Constantes bacteria ───────────────────────────────────────────────────────
const MAX_POP:        usize = 12;
const MAX_AGE:        u32   = 80_000;
const REPRODUCE_PROB: u32   = 30;
const COOLDOWN:       u32   = 2000;
const STARVATION_AGE: u32   = 5000;
const QUORUM_DECAY:        f32   = 0.985;
const QUORUM_THRESH:       f32   = 5.0;
const QUORUM_EVENT_THRESH: f32   = 8.0;
const MAX_SNAPS:           usize = 30;
const SNAP_INTERVAL_SECS:  u64   = 15;

// ── Constantes comida ─────────────────────────────────────────────────────────
const MAX_FOOD:        usize = 5;
const FOOD_REGEN:      f32   = 0.3;   // energía ganada por paso
const FOOD_EATEN:      f32   = 15.0;  // energía perdida cuando bacteria la alcanza
const FOOD_SIGNAL:     f32   = 2.0;   // señal depositada en posición por paso
const FOOD_SIGNAL_THR: f32   = 3.0;   // umbral para mostrar en mapa y calcular bonus
const FOOD_DECAY:      f32   = 0.990; // decay del campo de comida

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

// ── Quorum neighborhood ──────────────────────────────────────────────────────

fn quorum_neighborhood(pos: usize, quorum: &[f32]) -> [f32; 8] {
    let row = pos / 16;
    let col = pos % 16;
    let nbrs = [
        ((row + 15) % 16) * 16 + col,
        ((row + 15) % 16) * 16 + (col +  1) % 16,
        row * 16             + (col +  1) % 16,
        ((row +  1) % 16) * 16 + (col +  1) % 16,
        ((row +  1) % 16) * 16 + col,
        ((row +  1) % 16) * 16 + (col + 15) % 16,
        row * 16             + (col + 15) % 16,
        ((row + 15) % 16) * 16 + (col + 15) % 16,
    ];
    let q0 = quorum[pos];
    let mut g = [0f32; 8];
    for i in 0..8 { g[i] = (quorum[nbrs[i]] - q0).tanh(); }
    g
}

// ── MiniTransformer ──────────────────────────────────────────────────────────

struct MiniTransformer {
    embedding_matrix: [[f32; 8]; 256],
    attention_w_q:    [[f32; 8]; 8],
    attention_w_k:    [[f32; 8]; 8],
    attention_w_v:    [[f32; 8]; 8],
    ff_w1:            [[f32; 8]; 16],
    ff_w2:            [[f32; 16]; 8],
    learning_rate:    f32,
}

impl MiniTransformer {
    fn new(rng: &mut XorShift32) -> Self {
        Self {
            embedding_matrix: std::array::from_fn(|_| std::array::from_fn(|_| Self::rand(rng))),
            attention_w_q:    std::array::from_fn(|_| std::array::from_fn(|_| Self::rand(rng))),
            attention_w_k:    std::array::from_fn(|_| std::array::from_fn(|_| Self::rand(rng))),
            attention_w_v:    std::array::from_fn(|_| std::array::from_fn(|_| Self::rand(rng))),
            ff_w1:            std::array::from_fn(|_| std::array::from_fn(|_| Self::rand(rng))),
            ff_w2:            std::array::from_fn(|_| std::array::from_fn(|_| Self::rand(rng))),
            learning_rate: 0.001,
        }
    }

    fn rand(rng: &mut XorShift32) -> f32 {
        (rng.next_u32() as f32 / 4294967296.0 - 0.5) * 1.0
    }

    fn forward(&self, value: u8, position: usize, state: &[f32; 8], q_grad: &[f32; 8], rng: &mut XorShift32, epsilon: f32) -> ([u8; 8], [f32; 8]) {
        let ve = self.embedding_matrix[value as usize];
        let pe = self.embedding_matrix[position % 256];
        let mut emb = [0f32; 8];
        for i in 0..8 { emb[i] = ve[i] + pe[i] + state[i] * 0.1 + q_grad[i] * 0.5; }

        let attended = self.multihead_attention(&emb, state);
        let logits   = self.feedforward(&attended);

        let mut bits = [0u8; 8];
        let explore = (rng.next_u32() as f32 / 4294967296.0) < epsilon;
        for i in 0..8 {
            bits[i] = if explore {
                (rng.next_u32() & 1) as u8
            } else {
                let noise = (rng.next_u32() as f32 / 4294967296.0 - 0.5) * 0.2;
                if logits[i] + noise > 0.0 { 1 } else { 0 }
            };
        }
        (bits, logits)
    }

    fn multihead_attention(&self, emb: &[f32; 8], state: &[f32; 8]) -> [f32; 8] {
        let q1 = self.mm8(&self.attention_w_q, emb);
        let k1 = self.mm8(&self.attention_w_k, emb);
        let v1 = self.mm8(&self.attention_w_v, emb);
        let q2 = self.mm8(&self.attention_w_q, state);
        let k2 = self.mm8(&self.attention_w_k, state);
        let v2 = self.mm8(&self.attention_w_v, state);
        let s1 = self.dot(&q1, &k1) * 0.125;
        let s2 = self.dot(&q2, &k2) * 0.125;
        let mut out = [0f32; 8];
        for i in 0..8 { out[i] = (v1[i] * s1.tanh() + v2[i] * s2.tanh()) * 0.5; }
        out
    }

    fn feedforward(&self, input: &[f32; 8]) -> [f32; 8] {
        let hidden    = self.mm16(&self.ff_w1, input);
        let activated: [f32; 16] = hidden.map(|x| x.tanh());
        let mut out = [0f32; 8];
        for i in 0..8 { for j in 0..16 { out[i] += self.ff_w2[i][j] * activated[j]; } }
        out
    }

    fn learn(&mut self, value: u8, position: usize, state: &[f32; 8], q_grad: &[f32; 8], logits: &[f32; 8], reward: f32) {
        let ve = self.embedding_matrix[value as usize];
        let pe = self.embedding_matrix[position % 256];
        let mut emb = [0f32; 8];
        for i in 0..8 { emb[i] = ve[i] + pe[i] + state[i] * 0.1 + q_grad[i] * 0.5; }
        let attended = self.multihead_attention(&emb, state);
        let hidden   = self.mm16(&self.ff_w1, &attended);
        let activated: [f32; 16] = hidden.map(|x| x.tanh());
        let lr = self.learning_rate;
        for i in 0..8 {
            let d = if logits[i] > 0.0 { 1.0 } else { -1.0 };
            for j in 0..16 { self.ff_w2[i][j] += lr * reward * d * activated[j]; }
        }
        for i in 0..16 {
            let d = if hidden[i] > 0.0 { 1.0 } else { -1.0 };
            for j in 0..8 { self.ff_w1[i][j] += lr * reward * d * attended[j]; }
        }
    }

    fn mm8(&self, m: &[[f32; 8]; 8], v: &[f32; 8]) -> [f32; 8] {
        let mut r = [0f32; 8];
        for i in 0..8 { for j in 0..8 { r[i] += m[i][j] * v[j]; } }
        r
    }
    fn mm16(&self, m: &[[f32; 8]; 16], v: &[f32; 8]) -> [f32; 16] {
        let mut r = [0f32; 16];
        for i in 0..16 { for j in 0..8 { r[i] += m[i][j] * v[j]; } }
        r
    }
    fn dot(&self, a: &[f32; 8], b: &[f32; 8]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }
}

// ── Bacteria ─────────────────────────────────────────────────────────────────

struct Bacteria {
    position:    usize,
    state:       [f32; 8],
    transformer: MiniTransformer,
    visits:      [f32; 1000],
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
            state:       [0.0; 8],
            transformer: MiniTransformer::new(rng),
            visits:      [0.0; 1000],
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
        let mut seen = [false; 1000];
        for &p in &self.recent { seen[p] = true; }
        seen.iter().filter(|&&b| b).count() as f32 / self.recent.len() as f32
    }

    fn coverage(&self) -> f32 {
        let covered = self.visits[..256].iter().filter(|&&v| v > 1.0).count();
        covered as f32 / 256.0
    }

    fn step(&mut self, memory: &[u8], quorum: &[f32], food: &[f32]) -> (usize, u8, f32) {
        let value_read = memory[self.position];
        let prev_state = self.state;
        let q_grad = quorum_neighborhood(self.position, quorum);
        let (bits, logits) = self.transformer.forward(value_read, self.position, &prev_state, &q_grad, &mut self.rng, 0.20);

        let new_pos = bits.iter().enumerate()
            .fold(0usize, |acc, (i, &b)| acc + ((b as usize) << i))
            % memory.len();

        for i in 0..8 { self.state[i] = if bits[i] == 1 { 0.5 } else { -0.5 }; }

        for v in self.visits.iter_mut() { *v *= 0.9999; }
        let novelty     = 1.0 / (1.0 + self.visits[new_pos]);
        let memory_diff = ((memory[new_pos] as i32 - memory[self.position] as i32).abs() as f32) / 255.0;
        let recency     = self.recent.iter().filter(|&&p| p == new_pos).count() as f32;

        let q     = quorum[new_pos];
        let q_eff = q.min(20.0);
        let colony_bonus = (q_eff / (QUORUM_THRESH + q_eff)) * 1.2;

        let recency_weight  = if q > QUORUM_THRESH { 0.005 } else { 0.3 };
        let recency_penalty = recency_weight * recency;

        // Bonus por encontrar comida — motiva salir de la colonia a cazar
        let food_val   = food[new_pos].max(0.0);
        let food_bonus = (food_val / (FOOD_SIGNAL_THR + food_val)) * 0.8;

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
        self.age > STARVATION_AGE && self.recent_diversity() < 0.10
    }
}

// ── Comida ────────────────────────────────────────────────────────────────────

struct FoodAgent {
    position:    usize,
    state:       [f32; 8],
    transformer: MiniTransformer,
    energy:      f32,
    age:         u32,
    rng:         XorShift32,
    visits:      [f32; 256],
    recent:      VecDeque<usize>,
}

impl FoodAgent {
    fn new(rng: &mut XorShift32) -> Self {
        Self {
            position:    (rng.next_u32() as usize) % 256,
            state:       [0.0; 8],
            transformer: MiniTransformer::new(rng),
            energy:      50.0,
            age:         0,
            rng:         XorShift32::new(rng.next_u32()),
            visits:      [0.0; 256],
            recent:      VecDeque::with_capacity(200),
        }
    }

    fn diversity(&self) -> f32 {
        if self.recent.is_empty() { return 0.0; }
        let mut seen = [false; 256];
        for &p in &self.recent { seen[p] = true; }
        seen.iter().filter(|&&b| b).count() as f32 / self.recent.len() as f32
    }

    // La comida aprende a huir de las bacterias usando el mismo gradiente de quorum
    // pero con reward inverso: penalización por acercarse a zonas de quorum alto.
    fn step(&mut self, memory: &[u8], quorum: &[f32]) -> usize {
        let value_read = memory[self.position];
        let prev_state = self.state;
        let q_grad = quorum_neighborhood(self.position, quorum);
        let (bits, logits) = self.transformer.forward(
            value_read, self.position, &prev_state, &q_grad, &mut self.rng, 0.25,
        );

        let new_pos = bits.iter().enumerate()
            .fold(0usize, |acc, (i, &b)| acc + ((b as usize) << i))
            % 256;

        for i in 0..8 { self.state[i] = if bits[i] == 1 { 0.5 } else { -0.5 }; }

        for v in self.visits.iter_mut() { *v *= 0.9999; }
        let novelty = 1.0 / (1.0 + self.visits[new_pos]);
        let recency = self.recent.iter().filter(|&&p| p == new_pos).count() as f32;

        let q = quorum[new_pos];
        let bacteria_penalty = (q / (QUORUM_THRESH + q)) * 1.5;

        let reward = if new_pos == self.position {
            -0.8
        } else {
            novelty * 0.5 + 0.2 - 0.1 * recency - bacteria_penalty
        };

        self.transformer.learn(value_read, self.position, &prev_state, &q_grad, &logits, reward);

        self.recent.push_back(self.position);
        if self.recent.len() > 200 { self.recent.pop_front(); }

        self.position = new_pos;
        self.visits[new_pos] += 1.0;
        self.age += 1;
        self.energy += FOOD_REGEN;

        new_pos
    }
}

// ── Crossover ────────────────────────────────────────────────────────────────

fn crossover(a: &Bacteria, b: &Bacteria, rng: &mut XorShift32) -> Bacteria {
    let fa = a.recent_diversity().max(0.01);
    let fb = b.recent_diversity().max(0.01);
    let pa = (fa / (fa + fb) * 100.0) as u32;

    macro_rules! cross8x8 {
        ($field:ident) => {
            std::array::from_fn(|i| std::array::from_fn(|j|
                if rng.next_u32() % 100 < pa { a.transformer.$field[i][j] }
                else { b.transformer.$field[i][j] }
            ))
        };
    }

    let t = MiniTransformer {
        embedding_matrix: std::array::from_fn(|i| std::array::from_fn(|j|
            if rng.next_u32() % 100 < pa { a.transformer.embedding_matrix[i][j] }
            else { b.transformer.embedding_matrix[i][j] }
        )),
        attention_w_q: cross8x8!(attention_w_q),
        attention_w_k: cross8x8!(attention_w_k),
        attention_w_v: cross8x8!(attention_w_v),
        ff_w1: std::array::from_fn(|i| std::array::from_fn(|j|
            if rng.next_u32() % 100 < pa { a.transformer.ff_w1[i][j] }
            else { b.transformer.ff_w1[i][j] }
        )),
        ff_w2: std::array::from_fn(|i| std::array::from_fn(|j|
            if rng.next_u32() % 100 < pa { a.transformer.ff_w2[i][j] }
            else { b.transformer.ff_w2[i][j] }
        )),
        learning_rate: (a.transformer.learning_rate + b.transformer.learning_rate) / 2.0,
    };

    Bacteria {
        position:    if rng.next_u32() % 2 == 0 { a.position } else { b.position },
        state:       [0.0; 8],
        transformer: t,
        visits:      [0.0; 1000],
        recent:      VecDeque::with_capacity(50),
        rewards:     VecDeque::with_capacity(200),
        age:         0,
        cooldown:    COOLDOWN,
        rng:         XorShift32::new(rng.next_u32()),
    }
}

// ── Visualización ────────────────────────────────────────────────────────────

fn combined_visits(population: &[Bacteria]) -> [f32; 256] {
    let mut cv = [0f32; 256];
    for b in population { for i in 0..256 { cv[i] += b.visits[i]; } }
    cv
}

fn colony_zones(quorum: &[f32]) -> usize {
    quorum.iter().filter(|&&q| q > QUORUM_THRESH).count()
}

fn cell_terminal(pos: usize, population: &[Bacteria], food_agents: &[FoodAgent], cv: &[f32; 256], quorum: &[f32], food: &[f32]) -> String {
    if let Some((idx, _)) = population.iter().enumerate().find(|(_, b)| b.position == pos) {
        format!("\x1B[1;92m{:X}\x1B[0m  ", idx)       // verde brillante: bacteria
    } else if food_agents.iter().any(|fa| fa.position == pos) {
        "\x1B[1;33mF\x1B[0m  ".into()                  // amarillo: agente de comida
    } else if quorum[pos] > 8.0 {
        "\x1B[1;35m@\x1B[0m  ".into()                  // magenta: núcleo de colonia
    } else if quorum[pos] > QUORUM_THRESH {
        "\x1B[1;36m*\x1B[0m  ".into()                  // cyan: zona de colonia
    } else if food[pos] > FOOD_SIGNAL_THR {
        "\x1B[1;32mf\x1B[0m  ".into()                  // verde: señal de comida
    } else if cv[pos] > 100.0 {
        "\x1B[1;33m+\x1B[0m  ".into()
    } else if cv[pos] > 20.0 {
        "\x1B[37m.\x1B[0m  ".into()
    } else {
        "   ".into()
    }
}

fn cell_snapshot(pos: usize, population: &[Bacteria], food_agents: &[FoodAgent], cv: &[f32; 256], quorum: &[f32], food: &[f32]) -> &'static str {
    if population.iter().any(|b| b.position == pos)     { return "B  "; }
    if food_agents.iter().any(|fa| fa.position == pos)  { return "F  "; }
    if quorum[pos] > 8.0                                { return "@  "; }
    if quorum[pos] > QUORUM_THRESH                      { return "*  "; }
    if food[pos] > FOOD_SIGNAL_THR                      { return "f  "; }
    if cv[pos] > 100.0                                  { return "+  "; }
    if cv[pos] > 20.0                                   { return ".  "; }
    "   "
}

fn print_map(population: &[Bacteria], food_agents: &[FoodAgent], quorum: &[f32], food: &[f32], step: u32) {
    let cv    = combined_visits(population);
    let zones = colony_zones(quorum);
    let q_max = quorum.iter().cloned().fold(0f32, f32::max);
    let f_max = food.iter().cloned().fold(0f32, f32::max);
    print!("\x1B[2J\x1B[H");
    println!("  paso={:>9}  pop={}  food={}  colonias={}  q_max={:.1}  f_max={:.1}",
        step, population.len(), food_agents.len(), zones, q_max, f_max);
    for (i, b) in population.iter().enumerate() {
        println!("  B{:X}  pos={:>3}  div={:>3.0}%  cov={:>3.0}%  fit={:+.3}  age={}{}",
            i, b.position, b.recent_diversity() * 100.0, b.coverage() * 100.0,
            b.fitness(), b.age,
            if b.cooldown > 0 { "  [cd]" } else { "" });
    }
    for (i, fa) in food_agents.iter().enumerate() {
        println!("  F{:X}  pos={:>3}  div={:>3.0}%  energy={:>6.1}  age={}",
            i, fa.position, fa.diversity() * 100.0, fa.energy, fa.age);
    }
    println!();
    print!("      ");
    for col in 0..16 { print!("{:X}  ", col); }
    println!();
    for row in 0..16 {
        print!(" {:X}   ", row);
        for col in 0..16 {
            print!("{}", cell_terminal(row * 16 + col, population, food_agents, &cv, quorum, food));
        }
        println!();
    }
    println!("\n  0-F bacteria  F comida  @ núcleo  * colonia  f señal  + tibio  . frío");
}

fn save_snapshot(population: &[Bacteria], food_agents: &[FoodAgent], quorum: &[f32], food: &[f32], step: u32, index: usize, trigger: &str) {
    let cv    = combined_visits(population);
    let zones = colony_zones(quorum);
    let q_max = quorum.iter().cloned().fold(0f32, f32::max);
    let f_max = food.iter().cloned().fold(0f32, f32::max);
    if let Ok(mut f) = File::create(format!("snapshot_{:02}.txt", index)) {
        let _ = writeln!(f,
            "Snap {:02} | {} | paso={} | pop={} | food={} | colonias={} | q_max={:.1} | f_max={:.1}",
            index, trigger, step, population.len(), food_agents.len(), zones, q_max, f_max);
        for (i, b) in population.iter().enumerate() {
            let _ = writeln!(f,
                "  B{:X}  pos={:>3}  div={:>3.0}%  cov={:>3.0}%  fit={:+.3}  age={}",
                i, b.position, b.recent_diversity()*100.0, b.coverage()*100.0, b.fitness(), b.age);
        }
        for (i, fa) in food_agents.iter().enumerate() {
            let _ = writeln!(f,
                "  F{:X}  pos={:>3}  div={:>3.0}%  energy={:>6.1}  age={}",
                i, fa.position, fa.diversity()*100.0, fa.energy, fa.age);
        }
        let _ = writeln!(f);
        let mut header = String::from("      ");
        for col in 0..16 { header.push_str(&format!("{:X}  ", col)); }
        let _ = writeln!(f, "{}", header);
        for row in 0..16 {
            let mut line = format!(" {:X}   ", row);
            for col in 0..16 {
                let pos = row * 16 + col;
                if let Some((idx, _)) = population.iter().enumerate()
                    .find(|(_, b)| b.position == pos)
                {
                    line.push_str(&format!("{:X}  ", idx));
                } else {
                    line.push_str(cell_snapshot(pos, population, food_agents, &cv, quorum, food));
                }
            }
            let _ = writeln!(f, "{}", line);
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

    let mut memory: Vec<u8> = (0..1000).map(|i| ((i * 37 + 13) % 256) as u8).collect();
    let mut quorum: Vec<f32> = vec![0.0; 256];
    let mut food:   Vec<f32> = vec![0.0; 256];

    let mut population: Vec<Bacteria> = {
        let positions = [0usize, 64, 128, 192];
        positions.iter().map(|&pos| {
            let mut b = Bacteria::new(&mut rng);
            b.position = pos;
            b
        }).collect()
    };

    let mut food_agents: Vec<FoodAgent> = (0..3)
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
        // ── Fase 1: bacterias computan en paralelo (snapshot read-only) ────────
        let mem_snap  = memory.clone();
        let qm_snap   = quorum.clone();
        let food_snap = food.clone();

        let updates: Vec<(usize, u8)> = if population.len() >= 48 {
            population
                .par_iter_mut()
                .map(|b| { let (p, s, _) = b.step(&mem_snap, &qm_snap, &food_snap); (p, s) })
                .collect()
        } else {
            population
                .iter_mut()
                .map(|b| { let (p, s, _) = b.step(&mem_snap, &qm_snap, &food_snap); (p, s) })
                .collect()
        };

        // ── Fase 2: aplica writes de bacterias ────────────────────────────────
        for &(new_pos, store) in &updates {
            memory[new_pos] = store;
            quorum[new_pos] += 1.0;
            let row = new_pos / 16;
            let col = new_pos % 16;
            for &n in &[
                ((row + 15) % 16) * 16 + col,
                ((row +  1) % 16) * 16 + col,
                row * 16 + (col +  1) % 16,
                row * 16 + (col + 15) % 16,
            ] { quorum[n] += 0.3; }
        }
        for q in quorum.iter_mut() { *q *= QUORUM_DECAY; }

        // ── Agentes de comida: step + depósito de señal ───────────────────────
        for fa in food_agents.iter_mut() {
            let pos = fa.step(&mem_snap, &qm_snap);
            food[pos] += FOOD_SIGNAL;
        }

        // Bacterias comen: depletan campo y hieren al agente presente
        for &(new_pos, _) in &updates {
            if food[new_pos] > 0.0 {
                food[new_pos] = (food[new_pos] - FOOD_EATEN).max(0.0);
                for fa in food_agents.iter_mut() {
                    if fa.position == new_pos {
                        fa.energy -= FOOD_EATEN;
                    }
                }
            }
        }
        for fv in food.iter_mut() { *fv *= FOOD_DECAY; }

        // Muerte de agentes de comida por inanición
        if food_agents.len() > 1 {
            food_agents.retain(|fa| fa.energy > 0.0);
        }
        if food_agents.is_empty() {
            food_agents.push(FoodAgent::new(&mut rng));
        }

        // Reproducción de agentes de comida cuando tienen energía suficiente
        let current_n = food_agents.len();
        let mut new_food: Vec<FoodAgent> = vec![];
        for fa in food_agents.iter_mut() {
            if fa.energy > 200.0 && current_n + new_food.len() < MAX_FOOD {
                fa.energy /= 2.0;
                let mut child = FoodAgent::new(&mut rng);
                child.position = fa.position;
                child.energy   = fa.energy;
                new_food.push(child);
            }
        }
        food_agents.extend(new_food);

        // ── Reproducción y muerte de bacterias ───────────────────────────────
        let mut offspring: Vec<Bacteria> = vec![];
        for i in 0..population.len() {
            for j in (i + 1)..population.len() {
                if population[i].position == population[j].position
                    && population[i].cooldown == 0
                    && population[j].cooldown == 0
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

        // ── Historial cada 10 pasos ───────────────────────────────────────────
        if step % 10 == 0 {
            let q_now = quorum.iter().cloned().fold(0f32, f32::max);
            let f_now = food.iter().cloned().fold(0f32, f32::max);
            let zones = colony_zones(&quorum);
            let divs  = population.iter()
                .map(|b| format!("{:.0}%", b.recent_diversity()*100.0))
                .collect::<Vec<_>>().join(" ");
            history.push_back(format!(
                "paso={:>9}  pop={}  food={}  col={}  q={:.1}  f={:.1}  div=[{}]",
                step, population.len(), food_agents.len(), zones, q_now, f_now, divs
            ));
            if history.len() > 100 { history.pop_front(); }
        }

        // ── Mapa en terminal (máximo 2 veces/segundo) ─────────────────────────
        if last_print.elapsed().as_millis() >= 500 {
            print_map(&population, &food_agents, &quorum, &food, step);
            last_print = std::time::Instant::now();
        }

        // ── Snapshot de evento cuando quorum supera umbral ────────────────────
        let q_now   = quorum.iter().cloned().fold(0f32, f32::max);
        let q_above = q_now > QUORUM_EVENT_THRESH;
        if q_above && !q_prev_above
            && snapshot_count < MAX_SNAPS
            && last_event_snap.elapsed().as_secs() >= 5
        {
            snapshot_count += 1;
            save_snapshot(&population, &food_agents, &quorum, &food, step, snapshot_count, "EVENTO");
            save_history(&history);
            last_event_snap = std::time::Instant::now();
        }
        q_prev_above = q_above;

        // ── Snapshot periódico ────────────────────────────────────────────────
        if snapshot_count < MAX_SNAPS && last_snapshot.elapsed().as_secs() >= SNAP_INTERVAL_SECS {
            snapshot_count += 1;
            save_snapshot(&population, &food_agents, &quorum, &food, step, snapshot_count, "TIME");
            save_history(&history);
            last_snapshot = std::time::Instant::now();
        }

        step += 1;
    }
}
