// Estado del tablero y constantes globales

pub const GRID_SIDE: usize = 32;
pub const GRID_SIZE: usize = GRID_SIDE * GRID_SIDE;
pub const DIM:       usize = 10;

pub const MAX_POP:        usize = 64;
pub const MAX_AGE:        u32   = 80_000;
pub const REPRODUCE_PROB: u32   = 30;
pub const COOLDOWN:       u32   = 2_000;
pub const STARVATION_AGE: u32   = 5_000;

pub const QUORUM_DECAY:        f32 = 0.985;
pub const QUORUM_THRESH:       f32 = 5.0;
pub const QUORUM_SAT_THRESH:   f32 = 180.0;
pub const QUORUM_EVENT_THRESH: f32 = 8.0;
pub const QUORUM_DEPOSIT:      f32 = 12.0 / MAX_POP as f32;

pub const MAX_SNAPS:          usize = 30;
pub const SNAP_INTERVAL_SECS: u64   = 15;

pub const MAX_FOOD:        usize = 15;
pub const FOOD_REGEN:      f32   = 0.3;
pub const FOOD_MAX_ENERGY: f32   = 400.0;
pub const FOOD_EATEN:      f32   = 15.0 * 12.0 / MAX_POP as f32;
pub const FOOD_SIGNAL:     f32   = 4.0;
pub const FOOD_SIGNAL_THR: f32   = 3.0;
pub const FOOD_DECAY:      f32   = 0.990;
pub const FOOD_FLEE_THRESH:f32   = 100.0;

// Hambre de bacterias
pub const MAX_BACTERIA_ENERGY: f32 = 100.0;
pub const METABOLISM_RATE:     f32 = 0.005; // energía por paso (~20K pasos sin comida)
pub const FOOD_ENERGY_GAIN:    f32 = 5.0;   // energía ganada por paso en celda con comida

// Snapshot inmutable del tablero enviado a cada bacteria por tick
#[derive(Clone)]
pub struct WorldState {
    pub memory:   Vec<u8>,
    pub quorum:   Vec<f32>,
    pub food:     Vec<f32>,
    pub crowding: Vec<u8>,   // bacterias por celda el tick anterior
}

impl WorldState {
    pub fn new() -> Self {
        Self {
            memory:   (0..GRID_SIZE).map(|i| ((i * 37 + 13) % 256) as u8).collect(),
            quorum:   vec![0.0; GRID_SIZE],
            food:     vec![0.0; GRID_SIZE],
            crowding: vec![0u8; GRID_SIZE],
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
