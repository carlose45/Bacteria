use std::collections::VecDeque;
use std::fs::File;
use std::io::Write;

const MAX_POP:        usize = 12;
const MAX_AGE:        u32   = 80_000;
const REPRODUCE_PROB: u32   = 30;
const COOLDOWN:       u32   = 2000;
const STARVATION_AGE: u32   = 5000;
const QUORUM_DECAY:        f32   = 0.985; // semivida ~46 pasos — E[Q_random]=3.1 < umbral
const QUORUM_THRESH:       f32   = 5.0;   // umbral colonia: cluster de 2+ bacterias lo supera
const QUORUM_EVENT_THRESH: f32   = 8.0;   // dispara snapshot de evento al superar este valor
const MAX_SNAPS:           usize = 30;    // máximo de snapshots totales
const SNAP_INTERVAL_SECS:  u64   = 15;   // snapshot periódico cada 15 segundos

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

// Devuelve el gradiente de quorum en 8 direcciones (N NE E SE S SW W NW)
// relativo a la posición actual, en la cuadrícula 16×16 toroidal.
// Cada componente está en [-1, 1] via tanh.
fn quorum_neighborhood(pos: usize, quorum: &[f32]) -> [f32; 8] {
    let row = pos / 16;
    let col = pos % 16;
    let nbrs = [
        ((row + 15) % 16) * 16 + col,                // N
        ((row + 15) % 16) * 16 + (col +  1) % 16,   // NE
        row * 16             + (col +  1) % 16,      // E
        ((row +  1) % 16) * 16 + (col +  1) % 16,   // SE
        ((row +  1) % 16) * 16 + col,                // S
        ((row +  1) % 16) * 16 + (col + 15) % 16,   // SW
        row * 16             + (col + 15) % 16,      // W
        ((row + 15) % 16) * 16 + (col + 15) % 16,   // NW
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

    // Devuelve (nueva_posicion, valor_a_escribir, reward)
    // quorum: señal social acumulada por todas las bacterias en cada posición
    fn step(&mut self, memory: &[u8], quorum: &[f32], rng: &mut XorShift32) -> (usize, u8, f32) {
        let value_read = memory[self.position];
        let prev_state = self.state;
        let q_grad = quorum_neighborhood(self.position, quorum);
        let (bits, logits) = self.transformer.forward(value_read, self.position, &prev_state, &q_grad, rng, 0.20);

        let new_pos = bits.iter().enumerate()
            .fold(0usize, |acc, (i, &b)| acc + ((b as usize) << i))
            % memory.len();

        for i in 0..8 { self.state[i] = if bits[i] == 1 { 0.5 } else { -0.5 }; }

        for v in self.visits.iter_mut() { *v *= 0.9999; }
        let novelty     = 1.0 / (1.0 + self.visits[new_pos]);
        let memory_diff = ((memory[new_pos] as i32 - memory[self.position] as i32).abs() as f32) / 255.0;
        let recency     = self.recent.iter().filter(|&&p| p == new_pos).count() as f32;

        // Quorum sensing: bonus por moverse hacia zonas con presencia de colonia.
        // El transformer aprende a correlacionar memory[pos] alto con este bonus.
        let q = quorum[new_pos];
        let q_eff = q.min(20.0); // cap: evita atractor infinito, preserva gradiente periferia
        let colony_bonus = (q_eff / (QUORUM_THRESH + q_eff)) * 1.2;

        // Dentro de zona de quorum: recency mínima (0.005) para que rotas dentro
        // de la zona sin apilarte; fuera: recency completa (0.3) fuerza exploración.
        let recency_weight = if q > QUORUM_THRESH { 0.005 } else { 0.3 };
        let recency_penalty = recency_weight * recency;

        let reward = if new_pos == self.position {
            -0.8
        } else {
            novelty + 0.1 * memory_diff - recency_penalty + colony_bonus
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

        // Escribe el quorum actual en la memoria compartida.
        // Próximas bacterias que lean esta posición reciben la señal social.
        let store = (q * 20.0).min(255.0) as u8;
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

fn cell_terminal(pos: usize, population: &[Bacteria], cv: &[f32; 256], quorum: &[f32]) -> String {
    if let Some((idx, _)) = population.iter().enumerate().find(|(_, b)| b.position == pos) {
        format!("\x1B[1;92m{:X}\x1B[0m  ", idx)
    } else if quorum[pos] > 8.0 {
        "\x1B[1;35m@\x1B[0m  ".into()  // magenta: núcleo de colonia
    } else if quorum[pos] > QUORUM_THRESH {
        "\x1B[1;36m*\x1B[0m  ".into()  // cyan: zona de colonia
    } else if cv[pos] > 100.0 {
        "\x1B[1;33m+\x1B[0m  ".into()
    } else if cv[pos] > 20.0 {
        "\x1B[37m.\x1B[0m  ".into()
    } else {
        "   ".into()
    }
}

fn cell_snapshot(pos: usize, population: &[Bacteria], cv: &[f32; 256], quorum: &[f32]) -> &'static str {
    if population.iter().any(|b| b.position == pos) { return "B  "; }
    if quorum[pos] > 8.0        { return "@  "; }
    if quorum[pos] > QUORUM_THRESH { return "*  "; }
    if cv[pos] > 100.0          { return "+  "; }
    if cv[pos] > 20.0           { return ".  "; }
    "   "
}

fn print_map(population: &[Bacteria], quorum: &[f32], step: u32) {
    let cv = combined_visits(population);
    let zones = colony_zones(quorum);
    let q_max = quorum.iter().cloned().fold(0f32, f32::max);
    print!("\x1B[2J\x1B[H");
    println!("  paso={:>9}  pop={}  colonias={}  q_max={:.1}",
        step, population.len(), zones, q_max);
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
        for col in 0..16 {
            print!("{}", cell_terminal(row * 16 + col, population, &cv, quorum));
        }
        println!();
    }
    println!("\n  0-F bacteria  @ núcleo  * colonia  + tibio  . frío");
}

fn save_snapshot(population: &[Bacteria], quorum: &[f32], step: u32, index: usize, trigger: &str) {
    let cv = combined_visits(population);
    let zones = colony_zones(quorum);
    let q_max = quorum.iter().cloned().fold(0f32, f32::max);
    if let Ok(mut f) = File::create(format!("snapshot_{:02}.txt", index)) {
        let _ = writeln!(f, "Snap {:02} | {} | paso={} | pop={} | colonias={} | q_max={:.1}",
            index, trigger, step, population.len(), zones, q_max);
        for (i, b) in population.iter().enumerate() {
            let _ = writeln!(f,
                "  B{:X}  pos={:>3}  div={:>3.0}%  cov={:>3.0}%  fit={:+.3}  age={}",
                i, b.position, b.recent_diversity()*100.0,
                b.coverage()*100.0, b.fitness(), b.age);
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
                    line.push_str(cell_snapshot(pos, population, &cv, quorum));
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
    let mut quorum: Vec<f32> = vec![0.0; 256]; // señal social acumulada por posición
    let mut population: Vec<Bacteria> = {
        let positions = [0usize, 64, 128, 192];
        positions.iter().map(|&pos| {
            let mut b = Bacteria::new(&mut rng);
            b.position = pos;
            b
        }).collect()
    };
    let mut history: VecDeque<String> = VecDeque::with_capacity(100);
    let mut last_snapshot  = std::time::Instant::now();
    let mut last_print     = std::time::Instant::now();
    let mut last_event_snap = std::time::Instant::now();
    let mut snapshot_count = 0usize;
    let mut q_prev_above   = false; // estado anterior del quorum vs. umbral de evento
    let mut step = 0u32;

    loop {
        // Paso de cada bacteria — acumula quorum en posición visitada
        for b in &mut population {
            let (new_pos, store, _) = b.step(&memory, &quorum, &mut rng);
            memory[new_pos] = store;
            quorum[new_pos] += 1.0;
            // Depósito difuso: 4 vecinos cardinales reciben 0.3 → zona de 5 celdas
            let row = new_pos / 16;
            let col = new_pos % 16;
            for &n in &[
                ((row + 15) % 16) * 16 + col,
                ((row +  1) % 16) * 16 + col,
                row * 16 + (col +  1) % 16,
                row * 16 + (col + 15) % 16,
            ] { quorum[n] += 0.3; }
        }
        // Decay del campo de quorum (feromona que se evapora)
        for q in quorum.iter_mut() { *q *= QUORUM_DECAY; }

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
                let idx = population.iter().enumerate()
                    .min_by(|a, b| a.1.recent_diversity().partial_cmp(&b.1.recent_diversity()).unwrap())
                    .map(|(i, _)| i).unwrap();
                population.remove(idx);
            }
            population.push(child);
        }

        // Muerte por vejez o inanición
        if population.len() > 1 {
            population.retain(|b| b.age <= MAX_AGE && !b.is_starving());
        }
        if population.is_empty() { population.push(Bacteria::new(&mut rng)); }

        // Historial cada 10 pasos — resolución suficiente para ver dinámica del quorum
        if step % 10 == 0 {
            let q_now = quorum.iter().cloned().fold(0f32, f32::max);
            let zones  = colony_zones(&quorum);
            let divs   = population.iter()
                .map(|b| format!("{:.0}%", b.recent_diversity()*100.0))
                .collect::<Vec<_>>().join(" ");
            history.push_back(format!(
                "paso={:>9}  pop={}  col={}  q={:.1}  div=[{}]",
                step, population.len(), zones, q_now, divs
            ));
            if history.len() > 100 { history.pop_front(); }
        }

        // Mapa en terminal (máximo 2 veces/segundo)
        if last_print.elapsed().as_millis() >= 500 {
            print_map(&population, &quorum, step);
            last_print = std::time::Instant::now();
        }

        // Snapshot de EVENTO: primera vez que q_max sube sobre el umbral de evento
        let q_now    = quorum.iter().cloned().fold(0f32, f32::max);
        let q_above  = q_now > QUORUM_EVENT_THRESH;
        if q_above && !q_prev_above
            && snapshot_count < MAX_SNAPS
            && last_event_snap.elapsed().as_secs() >= 5
        {
            snapshot_count += 1;
            save_snapshot(&population, &quorum, step, snapshot_count, "EVENTO");
            save_history(&history);
            last_event_snap = std::time::Instant::now();
        }
        q_prev_above = q_above;

        // Snapshot periódico cada SNAP_INTERVAL_SECS segundos
        if snapshot_count < MAX_SNAPS && last_snapshot.elapsed().as_secs() >= SNAP_INTERVAL_SECS {
            snapshot_count += 1;
            save_snapshot(&population, &quorum, step, snapshot_count, "TIME");
            save_history(&history);
            last_snapshot = std::time::Instant::now();
        }

        step += 1;
    }
}
