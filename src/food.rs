use std::collections::VecDeque;
use crate::world::*;
use crate::ctrnn::{Ctrnn, crossover_ctrnn};
use crate::bacteria::XorShift32;

pub struct FoodAgent {
    pub position: usize,
    pub energy:   f32,
    pub age:      u32,
    pub recent:   VecDeque<usize>,
    pub rng:      XorShift32,
    pub ctrnn:    Ctrnn,
}

impl FoodAgent {
    pub fn new(rng: &mut XorShift32) -> Self {
        Self {
            position: (rng.next_u32() as usize) % GRID_SIZE,
            energy:   FOOD_MAX_ENERGY * 0.5,
            age:      0,
            recent:   VecDeque::with_capacity(200),
            rng:      XorShift32::new(rng.next_u32()),
            ctrnn:    Ctrnn::new(rng),
        }
    }

    // Spawn con genoma heredado + mutación — linaje continuo al morir o reproducirse
    pub fn from_genome(parent: &Ctrnn, pos: usize, rng: &mut XorShift32) -> Self {
        let mut child_rng = XorShift32::new(rng.next_u32());
        // Crossover consigo mismo = mutación pura al 4%
        let mut child_ctrnn = crossover_ctrnn(parent, parent, 50, &mut child_rng);
        child_ctrnn.y = [0.0; N_NEUR];
        Self {
            position: pos,
            energy:   FOOD_MAX_ENERGY * 0.5,
            age:      0,
            recent:   VecDeque::with_capacity(200),
            rng:      child_rng,
            ctrnn:    child_ctrnn,
        }
    }

    pub fn diversity(&self) -> f32 {
        if self.recent.is_empty() { return 0.0; }
        let mut seen = [false; GRID_SIZE];
        for &p in &self.recent { seen[p] = true; }
        seen.iter().filter(|&&b| b).count() as f32 / self.recent.len() as f32
    }

    pub fn step(&mut self, quorum: &[f32]) -> usize {
        self.energy = (self.energy + FOOD_REGEN).min(FOOD_MAX_ENERGY);
        self.age += 1;

        // Sensores: gradiente quórum 8 vecinos + peligro local + salud propia
        let q_grad  = quorum_neighborhood(self.position, quorum);
        let q_local = quorum[self.position];

        let sensors: [f32; N_SENS] = [
            q_grad[0], q_grad[1], q_grad[2], q_grad[3],
            q_grad[4], q_grad[5], q_grad[6], q_grad[7],
            (q_local / QUORUM_SAT_THRESH).min(1.0), // presión bacteriana local
            self.energy / FOOD_MAX_ENERGY,           // nivel de salud propio
            0.0, 0.0,                                // reservado
            0.0, 0.0,                                // stigma (no usado por comida)
        ];

        let action  = self.ctrnn.step(&sensors, &mut self.rng);
        let nbrs    = cardinal_neighbors(self.position);
        let new_pos = match action {
            1 => nbrs[0],
            2 => nbrs[1],
            3 => nbrs[2],
            4 => nbrs[3],
            _ => self.position,
        };

        self.recent.push_back(self.position);
        if self.recent.len() > 200 { self.recent.pop_front(); }
        self.position = new_pos;
        self.position
    }
}
