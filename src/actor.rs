// Ciclo de vida async de cada bacteria y tipos de mensajes

use std::sync::Arc;
use tokio::sync::{mpsc, oneshot};
use crate::bacteria::Bacteria;
use crate::world::WorldState;

// ── Mensajes bacteria → board ─────────────────────────────────────────────────

pub struct StepResult {
    pub id:         usize,
    pub new_pos:    usize,
    pub store:      u8,
    pub reward:     f32,
    pub position:   usize,
    pub fitness:    f32,
    pub diversity:  f32,
    pub coverage:   f32,
    pub age:        u32,
    pub cooldown:   u32,
    pub should_die:  bool,
    pub hunger:      f32,
    pub metabolism:  f32,
    pub curiosity:   f32,
    pub sociability: f32,
    pub selectivity: f32,
    pub altruism:    f32,
    // Visitas acumuladas para el mapa de calor (solo las > umbral)
    pub hot_cells:  Vec<(usize, f32)>,
}

pub enum BacteriaMsg {
    Step(StepResult),
}

// Petición del board a una bacteria para obtener su genoma
pub type GenomeRequest = oneshot::Sender<Box<Bacteria>>;

// ── Loop principal de cada bacteria ──────────────────────────────────────────

pub async fn bacteria_loop(
    mut bacteria:      Bacteria,
    id:                usize,
    mut tick_rx:       mpsc::Receiver<Arc<WorldState>>,
    action_tx:         mpsc::Sender<BacteriaMsg>,
    mut genome_req_rx: mpsc::Receiver<GenomeRequest>,
) {
    loop {
        tokio::select! {
            // Tick normal: recibe snapshot, computa, responde
            result = tick_rx.recv() => {
                let snap = match result {
                    Some(s) => s,
                    None    => break,
                };

                // CPU-bound: step del transformer (síncrono dentro de la task async)
                let (new_pos, store, reward) = bacteria.step(
                    &snap.memory, &snap.quorum, &snap.food, &snap.crowding, &snap.stigma,
                );

                let should_die = bacteria.age > crate::world::MAX_AGE
                    || bacteria.is_starving();

                // Comprime las visitas: celdas con valor > 3 (≈ visitada varias veces recientemente)
                let hot_cells: Vec<(usize, f32)> = bacteria.visits.iter().enumerate()
                    .filter(|(_, &v)| v > 3.0)
                    .map(|(i, &v)| (i, v))
                    .collect();

                let msg = BacteriaMsg::Step(StepResult {
                    id,
                    new_pos,
                    store,
                    reward,
                    position:   bacteria.position,
                    fitness:    bacteria.fitness(),
                    diversity:  bacteria.recent_diversity(),
                    coverage:   bacteria.coverage(),
                    age:        bacteria.age,
                    cooldown:   bacteria.cooldown,
                    should_die,
                    hunger:      bacteria.hunger(),
                    metabolism:  bacteria.metabolism,
                    curiosity:   bacteria.curiosity,
                    sociability: bacteria.sociability,
                    selectivity: bacteria.selectivity,
                    altruism:    bacteria.altruism,
                    hot_cells,
                });

                if action_tx.send(msg).await.is_err() { break; }
                if should_die { break; }
            }

            // El board necesita el genoma para hacer crossover con esta bacteria
            Some(reply_tx) = genome_req_rx.recv() => {
                // Pausamos brevemente para prestar el genoma
                // El board devuelve una Bacteria nueva (la hija) pero nosotros
                // seguimos siendo nosotros — solo compartimos los pesos
                let snapshot = Box::new(Bacteria {
                    position:    bacteria.position,
                    ctrnn:       bacteria.ctrnn.clone(),
                    visits:      Box::new([0.0; crate::world::GRID_SIZE]),
                    recent:      bacteria.recent.clone(),
                    rewards:     bacteria.rewards.clone(),
                    age:         bacteria.age,
                    cooldown:    bacteria.cooldown,
                    rng:         crate::bacteria::XorShift32::new(bacteria.rng.state),
                    energy:      bacteria.energy,
                    metabolism:  bacteria.metabolism,
                    curiosity:   bacteria.curiosity,
                    sociability: bacteria.sociability,
                    selectivity: bacteria.selectivity,
                    altruism:    bacteria.altruism,
                });
                let _ = reply_tx.send(snapshot);
            }
        }
    }
}
