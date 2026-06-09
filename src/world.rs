// Estado del tablero y constantes globales

pub const GRID_SIDE: usize = 32;
pub const GRID_SIZE: usize = GRID_SIDE * GRID_SIDE;

// CTRNN
pub const N_NEUR: usize = 12;  // neuronas recurrentes
pub const N_SENS: usize = 14;  // entradas de sensores (12 base + 2 memoria colectiva)
pub const N_ACT:  usize = 5;   // acciones: quedar + 4 cardinales

pub const MAX_POP:        usize = 64;
pub const MAX_AGE:        u32   = 80_000;
pub const REPRODUCE_PROB: u32   = 30;
pub const COOLDOWN:       u32   = 2_000;
pub const STARVATION_AGE: u32   = 5_000;

pub const QUORUM_DECAY:      f32 = 0.985;
pub const QUORUM_THRESH:     f32 = 5.0;
pub const QUORUM_SAT_THRESH: f32 = 120.0;
pub const QUORUM_DEPOSIT:    f32 = 12.0 / MAX_POP as f32;

pub const MAX_STEPS:          u32   = 1_000_000;

pub const MAX_FOOD:        usize = 3;
pub const FOOD_REGEN:      f32   = 0.15;
pub const FOOD_MAX_ENERGY: f32   = 150.0;
pub const FOOD_EATEN:      f32   = 15.0 * 12.0 / MAX_POP as f32;
pub const FOOD_SIGNAL:     f32   = 4.0;
pub const FOOD_SIGNAL_THR: f32   = 3.0;
pub const FOOD_DECAY:      f32   = 0.990;

// Hambre de bacterias
pub const MAX_BACTERIA_ENERGY: f32 = 100.0;
pub const METABOLISM_RATE:     f32 = 0.015; // tasa base — cada bacteria tiene la suya propia
pub const METABOLISM_SPREAD:   f32 = 0.40;  // variación inicial ±40% del valor base
pub const FOOD_ENERGY_GAIN:    f32 = 2.0;   // energía ganada por paso en celda con comida
pub const HUNGER_ALARM_BOOST:  f32 = 3.0;   // multiplicador quórum al encontrar comida con hambre
pub const HUNGER_REPRO_THRESH: f32 = 0.15;  // hambre máxima para reproducirse (energía > 85%)
pub const ALTRUISM_COST:       f32 = 0.005; // coste energético por unidad de altruismo por paso

// Memoria colectiva (estigma)
pub const STIGMA_DECAY:        f32 = 0.9999;  // vida media ~7000 pasos
pub const STIGMA_DEPOSIT_LIVE: f32 = 0.002;   // depósito continuo por visita acumulada
pub const STIGMA_DEPOSIT_DEATH: f32 = 0.005;  // pulso al morir — legado al sustrato
pub const STIGMA_SAT:          f32 = 20.0;    // saturación para normalización sensorial

// Snapshot inmutable del tablero enviado a cada bacteria por tick
#[derive(Clone)]
pub struct WorldState {
    pub memory:   Vec<u8>,
    pub quorum:   Vec<f32>,
    pub food:     Vec<f32>,
    pub crowding: Vec<u8>,
    pub stigma:   Vec<f32>,  // memoria colectiva — experiencia acumulada de generaciones
}

impl WorldState {
    pub fn new() -> Self {
        Self {
            memory:   (0..GRID_SIZE).map(|i| ((i * 37 + 13) % 256) as u8).collect(),
            quorum:   vec![0.0; GRID_SIZE],
            food:     vec![0.0; GRID_SIZE],
            crowding: vec![0u8; GRID_SIZE],
            stigma:   vec![0.0; GRID_SIZE],
        }
    }
}

// Gradiente de quórum en las 8 celdas vecinas (toroidal)
pub fn quorum_neighborhood(pos: usize, quorum: &[f32]) -> [f32; 8] {
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

pub fn colony_zones(quorum: &[f32]) -> usize {
    quorum.iter().filter(|&&q| q > QUORUM_THRESH).count()
}

pub fn cardinal_neighbors(pos: usize) -> [usize; 4] {
    let row = pos / GRID_SIDE;
    let col = pos % GRID_SIDE;
    let s   = GRID_SIDE;
    [
        ((row + s - 1) % s) * s + col,
        ((row + 1)     % s) * s + col,
        row * s + (col + 1) % s,
        row * s + (col + s - 1) % s,
    ]
}
