use std::collections::VecDeque;
use crate::world::*;
use crate::bacteria::XorShift32;

pub struct FoodAgent {
    pub position: usize,
    pub energy:   f32,
    pub age:      u32,
    pub recent:   VecDeque<usize>,
    pub rng:      XorShift32,
}

impl FoodAgent {
    pub fn new(rng: &mut XorShift32) -> Self {
        Self {
            position: (rng.next_u32() as usize) % GRID_SIZE,
            energy:   50.0,
            age:      0,
            recent:   VecDeque::with_capacity(200),
            rng:      XorShift32::new(rng.next_u32()),
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

        let q_local = quorum[self.position];

        if q_local > FOOD_FLEE_THRESH {
            // Hay demasiadas bacterias: huir hacia el vecino con menos quórum
            let nbrs = cardinal_neighbors(self.position);
            self.position = *nbrs.iter()
                .min_by(|&&a, &&b| quorum[a].partial_cmp(&quorum[b]).unwrap())
                .unwrap();
        } else if q_local < QUORUM_THRESH {
            // Sin bacterias cerca: deambular para encontrar colonias
            if self.rng.next_u32() % 100 < 15 {
                let nbrs = cardinal_neighbors(self.position);
                self.position = nbrs[self.rng.next_u32() as usize % 4];
            }
        }
        // Quórum medio: quedarse (zona con bacterias, no huir aún)

        self.recent.push_back(self.position);
        if self.recent.len() > 200 { self.recent.pop_front(); }
        self.position
    }
}
