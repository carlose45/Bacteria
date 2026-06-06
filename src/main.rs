use std::collections::VecDeque;
use std::fs::File;
use std::io::Write;

const MAX_POP:        usize = 12;
const MAX_AGE:        u32   = 80_000; // edad máxima — garantiza recambio generacional
const REPRODUCE_PROB: u32   = 30;   // % de probabilidad al encontrarse
const COOLDOWN:       u32   = 2000; // pasos antes de poder reproducirse
const STARVATION_AGE: u32   = 5000; // edad mínima para morir de inanición

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

    fn forward(&self, value: u8, position: usize, state: &[f32; 8], rng: &mut XorShift32, epsilon: f32) -> ([u8; 8], [f32; 8]) {
        let ve = self.embedding_matrix[value as usize];
        let pe = self.embedding_matrix[position % 256];
        let mut emb = [0f32; 8];
        for i in 0..8 { emb[i] = ve[i] + pe[i] + state[i] * 0.1; }

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

    fn learn(&mut self, value: u8, position: usize, state: &[f32; 8], logits: &[f32; 8], reward: f32) {
        let ve = self.embedding_matrix[value as usize];
        let pe = self.embedding_matrix[position % 256];
        let mut emb = [0f32; 8];
        for i in 0..8 { emb[i] = ve[i] + pe[i] + state[i] * 0.1; }
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
        }
    }

    fn fitness(&self) -> f32 {
        if self.rewards.is_empty() { return 0.0; }
        self.rewards.iter().sum::<f32>() / self.rewards.len() as f32
    }

    // Diversidad reciente: posiciones únicas en el buffer / longitud del buffer
    // Independiente de la edad — mide qué tan bien explora AHORA
    fn recent_diversity(&self) -> f32 {
        if self.recent.is_empty() { return 0.0; }
        let mut seen = [false; 1000];
        for &p in &self.recent { seen[p] = true; }
        seen.iter().filter(|&&b| b).count() as f32 / self.recent.len() as f32
    }

    // Cobertura acumulada (solo para display — satura al 100% con la edad)
    fn coverage(&self) -> f32 {
        let covered = self.visits[..256].iter().filter(|&&v| v > 1.0).count();
        covered as f32 / 256.0
    }

    // Devuelve (nueva_posicion, valor_a_escribir, reward)
    fn step(&mut self, memory: &[u8], rng: &mut XorShift32) -> (usize, u8, f32) {
        let value_read = memory[self.position];
        let prev_state = self.state;
        let (bits, logits) = self.transformer.forward(value_read, self.position, &prev_state, rng, 0.20);

        let new_pos = bits.iter().enumerate()
            .fold(0usize, |acc, (i, &b)| acc + ((b as usize) << i))
            % memory.len();

        for i in 0..8 { self.state[i] = if bits[i] == 1 { 0.5 } else { -0.5 }; }

        for v in self.visits.iter_mut() { *v *= 0.9999; }
        let novelty     = 1.0 / (1.0 + self.visits[new_pos]);
        let memory_diff = ((memory[new_pos] as i32 - memory[self.position] as i32).abs() as f32) / 255.0;
        let recency     = self.recent.iter().filter(|&&p| p == new_pos).count() as f32;

        let reward = if new_pos == self.position {
            -0.8
        } else {
            novelty + 0.1 * memory_diff - 0.3 * recency
        };

        self.transformer.learn(value_read, self.position, &prev_state, &logits, reward);

        self.recent.push_back(self.position);
        if self.recent.len() > 500 { self.recent.pop_front(); }

        self.position = new_pos;
        self.visits[new_pos] += 1.0;

        self.rewards.push_back(reward);
        if self.rewards.len() > 200 { self.rewards.pop_front(); }
        self.age += 1;
        if self.cooldown > 0 { self.cooldown -= 1; }

        let store = ((new_pos * 3 + self.age as usize) % 256) as u8;
        (new_pos, store, reward)
    }

    fn is_starving(&self) -> bool {
        self.age > STARVATION_AGE && self.recent_diversity() < 0.10
    }
}

// ── Crossover ────────────────────────────────────────────────────────────────

fn crossover(a: &Bacteria, b: &Bacteria, rng: &mut XorShift32) -> Bacteria {
    let fa = a.recent_diversity().max(0.01);
    let fb = b.recent_diversity().max(0.01);
    let pa = (fa / (fa + fb) * 100.0) as u32; // probabilidad de heredar de A

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
    }
}

// ── Visualización ────────────────────────────────────────────────────────────

fn combined_visits(population: &[Bacteria]) -> [f32; 256] {
    let mut cv = [0f32; 256];
    for b in population { for i in 0..256 { cv[i] += b.visits[i]; } }
    cv
}

fn cell(pos: usize, population: &[Bacteria], cv: &[f32; 256]) -> String {
    if let Some((idx, _)) = population.iter().enumerate().find(|(_, b)| b.position == pos) {
        format!("\x1B[1;92m{:X}\x1B[0m  ", idx)
    } else {
        let v = cv[pos];
        if      v > 500.0 { "\x1B[1;31m#\x1B[0m  ".into() }
        else if v > 100.0 { "\x1B[1;33m+\x1B[0m  ".into() }
        else if v > 20.0  { "\x1B[37m.\x1B[0m  ".into()   }
        else              { "   ".into()                    }
    }
}

fn print_map(population: &[Bacteria], step: u32) {
    let cv = combined_visits(population);
    print!("\x1B[2J\x1B[H");
    println!("  paso={:>9}  pop={}", step, population.len());
    for (i, b) in population.iter().enumerate() {
        println!("  B{:X}  pos={:>3}  div={:>3.0}%  cov={:>3.0}%  fit={:+.3}  age={}{}",
            i, b.position, b.recent_diversity() * 100.0, b.coverage() * 100.0,
            b.fitness(), b.age,
            if b.cooldown > 0 { "  [cd]" } else { "" });
    }
    println!();
    print!("      ");
    for col in 0..16 { print!("{:X}  ", col); }
    println!();
    for row in 0..16 {
        print!(" {:X}   ", row);
        for col in 0..16 { print!("{}", cell(row * 16 + col, population, &cv)); }
        println!();
    }
    println!("\n  0-9 bacteria  # caliente  + tibio  . frío");
}

fn save_snapshot(population: &[Bacteria], step: u32, index: usize) {
    let cv = combined_visits(population);
    if let Ok(mut f) = File::create(format!("snapshot_{:02}.txt", index)) {
        let _ = writeln!(f, "Snap {:02} | paso={} | pop={}", index, step, population.len());
        for (i, b) in population.iter().enumerate() {
            let _ = writeln!(f, "  B{:X}  pos={:>3}  div={:>3.0}%  cov={:>3.0}%  fit={:+.3}  age={}", i, b.position, b.recent_diversity()*100.0, b.coverage()*100.0, b.fitness(), b.age);
        }
        let _ = writeln!(f);
        let mut header = String::from("      ");
        for col in 0..16 { header.push_str(&format!("{:X}  ", col)); }
        let _ = writeln!(f, "{}", header);
        for row in 0..16 {
            let mut line = format!(" {:X}   ", row);
            for col in 0..16 {
                let pos = row * 16 + col;
                if let Some((idx, _)) = population.iter().enumerate().find(|(_, b)| b.position == pos) {
                    line.push_str(&format!("{:X}  ", idx));
                } else {
                    let v = cv[pos];
                    if      v > 500.0 { line.push_str("#  "); }
                    else if v > 100.0 { line.push_str("+  "); }
                    else if v > 20.0  { line.push_str(".  "); }
                    else              { line.push_str("   "); }
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
    let mut population: Vec<Bacteria> = {
        let positions = [0usize, 64, 128, 192];
        positions.iter().map(|&pos| {
            let mut b = Bacteria::new(&mut rng);
            b.position = pos;
            b
        }).collect()
    };
    let mut history: VecDeque<String> = VecDeque::with_capacity(100);
    let mut last_snapshot = std::time::Instant::now();
    let mut snapshot_count = 0usize;
    let mut step = 0u32;

    loop {
        // Paso de cada bacteria
        for b in &mut population {
            let (new_pos, store, _) = b.step(&memory, &mut rng);
            memory[new_pos] = store;
        }

        // Detección de encuentros y reproducción
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
                // Mata al de menor diversidad reciente (independiente de la edad)
                let idx = population.iter().enumerate()
                    .min_by(|a, b| a.1.recent_diversity().partial_cmp(&b.1.recent_diversity()).unwrap())
                    .map(|(i, _)| i).unwrap();
                population.remove(idx);
            }
            population.push(child);
        }

        // Muerte por vejez o inanición (mínimo 1 bacteria viva)
        if population.len() > 1 {
            population.retain(|b| b.age <= MAX_AGE && !b.is_starving());
        }
        if population.is_empty() { population.push(Bacteria::new(&mut rng)); }

        // Historial (cada 100 pasos)
        if step % 100 == 0 {
            let divs = population.iter()
                .map(|b| format!("{:.0}%", b.recent_diversity()*100.0))
                .collect::<Vec<_>>().join(" ");
            history.push_back(format!("paso={:>9}  pop={}  div=[{}]", step, population.len(), divs));
            if history.len() > 100 { history.pop_front(); }
        }

        // Mapa en terminal (cada 500 pasos)
        if step % 500 == 0 { print_map(&population, step); }

        // Snapshot cada 60 segundos, máximo 10
        if snapshot_count < 10 && last_snapshot.elapsed().as_secs() >= 60 {
            snapshot_count += 1;
            save_snapshot(&population, step, snapshot_count);
            save_history(&history);
            last_snapshot = std::time::Instant::now();
        }

        step += 1;
    }
}
