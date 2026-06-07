use std::collections::VecDeque;
use crate::world::*;
use crate::ctrnn::{Ctrnn, crossover_ctrnn};

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

// ── Bacteria ──────────────────────────────────────────────────────────────────

pub struct Bacteria {
    pub position:    usize,
    pub ctrnn:       Ctrnn,
    pub visits:      Box<[f32; GRID_SIZE]>,
    pub recent:      VecDeque<usize>,
    pub rewards:     VecDeque<f32>,
    pub age:         u32,
    pub cooldown:    u32,
    pub rng:         XorShift32,
    pub energy:      f32,
    pub metabolism:  f32,  // tasa de consumo energético individual
    pub curiosity:   f32,  // multiplicador de recompensa por novedad    [0.2, 2.0]
    pub sociability: f32,  // multiplicador de recompensa por colonia    [0.2, 2.0]
    pub selectivity: f32,  // exigencia de fitness al elegir pareja      [0.0, 1.0]
    pub altruism:    f32,  // multiplica quórum depositado + paga coste  [0.2, 2.0]
}

impl Bacteria {
    pub fn new(rng: &mut XorShift32) -> Self {
        let rf = |r: &mut XorShift32| r.next_u32() as f32 / u32::MAX as f32;
        let metabolism  = METABOLISM_RATE * (1.0 - METABOLISM_SPREAD + rf(rng) * METABOLISM_SPREAD * 2.0);
        let curiosity   = 0.5 + rf(rng);          // [0.5, 1.5]
        let sociability = 0.5 + rf(rng);          // [0.5, 1.5]
        let selectivity = rf(rng);                 // [0.0, 1.0]
        let altruism    = 0.5 + rf(rng);          // [0.5, 1.5]
        Self {
            position:    0,
            ctrnn:       Ctrnn::new(rng),
            visits:      Box::new([0.0; GRID_SIZE]),
            recent:      VecDeque::with_capacity(500),
            rewards:     VecDeque::with_capacity(200),
            age:         0,
            cooldown:    0,
            rng:         XorShift32::new(rng.next_u32()),
            energy:      MAX_BACTERIA_ENERGY,
            metabolism,
            curiosity,
            sociability,
            selectivity,
            altruism,
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

    pub fn step(&mut self, memory: &[u8], quorum: &[f32], food: &[f32], crowding: &[u8], stigma: &[f32]) -> (usize, u8, f32) {
        // ── Construir vector de sensores (N_SENS = 12) ────────────────────────
        let q_grad   = quorum_neighborhood(self.position, quorum);
        let food_cur = food[self.position].max(0.0);
        let nbrs     = cardinal_neighbors(self.position);
        // Gradiente de comida: ¿hay más comida en algún vecino cardinal?
        let food_max_nbr = nbrs.iter().map(|&n| food[n].max(0.0)).fold(0.0f32, f32::max);
        let food_gradient = (food_max_nbr - food_cur).max(0.0)
                            / (FOOD_SIGNAL_THR + food_max_nbr + 1.0);

        // Memoria colectiva: valor local y gradiente cardinal
        let s_local   = stigma[self.position];
        let s_max_nbr = nbrs.iter().map(|&n| stigma[n]).fold(s_local, f32::max);

        let sensors: [f32; N_SENS] = [
            q_grad[0], q_grad[1], q_grad[2], q_grad[3],     // gradiente quórum
            q_grad[4], q_grad[5], q_grad[6], q_grad[7],
            food_cur / (FOOD_SIGNAL_THR + food_cur + 1.0),   // comida local
            food_gradient,                                    // gradiente de comida cardinal
            self.hunger(),                                    // urgencia de hambre
            (crowding[self.position] as f32 / 8.0).min(1.0), // densidad local
            s_local / (STIGMA_SAT + s_local + 1.0),          // memoria colectiva local
            (s_max_nbr - s_local).max(0.0) / (STIGMA_SAT + 1.0), // gradiente hacia más memoria
        ];

        // ── CTRNN elige acción (0=quedar, 1=N, 2=S, 3=E, 4=W) ───────────────
        let action  = self.ctrnn.step(&sensors, &mut self.rng);
        let new_pos = match action {
            1 => nbrs[0],
            2 => nbrs[1],
            3 => nbrs[2],
            4 => nbrs[3],
            _ => self.position,
        };

        // ── Métricas de exploración ───────────────────────────────────────────
        for v in self.visits.iter_mut() { *v *= 0.9990; }
        let novelty     = 1.0 / (1.0 + self.visits[new_pos]);
        let memory_diff = ((memory[new_pos] as i32 - memory[self.position] as i32).abs() as f32) / 255.0;
        let recency     = self.recent.iter().filter(|&&p| p == new_pos).count() as f32;

        // ── Quórum y colonia ──────────────────────────────────────────────────
        let q     = quorum[new_pos];
        let q_eff = q.min(20.0);
        let colony_bonus = (q_eff / (QUORUM_THRESH + q_eff)) * 1.2;

        let recency_weight = if q > QUORUM_SAT_THRESH { 0.15 }
                             else if q > QUORUM_THRESH { 0.005 }
                             else                      { 0.3   };
        let recency_penalty = recency_weight * recency;

        // ── Hambre y comida ───────────────────────────────────────────────────
        let food_val = food[new_pos].max(0.0);
        if food_val > FOOD_SIGNAL_THR {
            self.energy = (self.energy + FOOD_ENERGY_GAIN).min(MAX_BACTERIA_ENERGY);
        } else {
            self.energy = (self.energy - self.metabolism - self.altruism * ALTRUISM_COST).max(0.0);
        }
        let h = self.hunger();

        let food_bonus = (food_val / (FOOD_SIGNAL_THR + food_val)) * (1.5 + h * 3.0);

        // ── Penalización de crowding (inversamente proporcional al hambre) ────
        let crowd = crowding[new_pos] as f32;
        let crowd_tolerance    = 1.0 - h * 0.7;
        let crowding_penalty   = ((crowd - 1.0).max(0.0) / 4.0).min(1.0) * 2.5 * crowd_tolerance;

        // ── Recompensa ────────────────────────────────────────────────────────
        let reward = if new_pos == self.position { -0.8 }
                     else { novelty * self.curiosity + 0.1 * memory_diff - recency_penalty
                            + colony_bonus * self.sociability + food_bonus - crowding_penalty };

        // ── Actualizar estado interno ─────────────────────────────────────────
        self.recent.push_back(self.position);
        if self.recent.len() > 500 { self.recent.pop_front(); }

        self.position       = new_pos;
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

    let rf  = |r: &mut XorShift32| r.next_u32() as f32 / u32::MAX as f32;
    let mid = |x: f32, y: f32| (x + y) * 0.5;

    let child_metabolism  = (mid(a.metabolism,  b.metabolism)  + (rf(rng)-0.5)*METABOLISM_RATE*0.4)
                                .clamp(METABOLISM_RATE*0.3, METABOLISM_RATE*2.5);
    let child_curiosity   = (mid(a.curiosity,   b.curiosity)   + (rf(rng)-0.5)*0.2)
                                .clamp(0.2, 2.0);
    let child_sociability = (mid(a.sociability, b.sociability) + (rf(rng)-0.5)*0.2)
                                .clamp(0.2, 2.0);
    let child_selectivity = (mid(a.selectivity, b.selectivity) + (rf(rng)-0.5)*0.1)
                                .clamp(0.0, 1.0);
    let child_altruism    = (mid(a.altruism,    b.altruism)    + (rf(rng)-0.5)*0.2)
                                .clamp(0.2, 2.0);

    Bacteria {
        position:    if rng.next_u32() % 2 == 0 { a.position } else { b.position },
        ctrnn:       crossover_ctrnn(&a.ctrnn, &b.ctrnn, pa, rng),
        visits:      Box::new([0.0; GRID_SIZE]),
        recent:      VecDeque::with_capacity(50),
        rewards:     VecDeque::with_capacity(200),
        age:         0,
        cooldown:    COOLDOWN,
        rng:         XorShift32::new(rng.next_u32()),
        energy:      (a.energy + b.energy) * 0.4,
        metabolism:  child_metabolism,
        curiosity:   child_curiosity,
        sociability: child_sociability,
        selectivity: child_selectivity,
        altruism:    child_altruism,
    }
}
