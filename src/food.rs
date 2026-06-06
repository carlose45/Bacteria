use std::collections::VecDeque;
use crate::world::*;
use crate::bacteria::XorShift32;

pub struct FoodAgent {
    pub position: usize,
    pub energy:   f32,
    pub age:      u32,
    pub recent:   VecDeque<usize>,
}

impl FoodAgent {
    pub fn new(rng: &mut XorShift32) -> Self {
        Self {
            position: (rng.next_u32() as usize) % GRID_SIZE,
            energy:   50.0,
            age:      0,
            recent:   VecDeque::with_capacity(200),
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

        if quorum[self.position] > FOOD_FLEE_THRESH {
            let nbrs = cardinal_neighbors(self.position);
            self.position = *nbrs.iter()
                .min_by(|&&a, &&b| quorum[a].partial_cmp(&quorum[b]).unwrap())
                .unwrap();
        }

        self.recent.push_back(self.position);
        if self.recent.len() > 200 { self.recent.pop_front(); }
        self.position
    }
}
