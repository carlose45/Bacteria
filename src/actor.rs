// Ciclo de vida async de cada bacteria y tipos de mensajes

use std::sync::Arc;
use tokio::sync::{broadcast, mpsc, oneshot};
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
    pub age:        u32,
    pub cooldown:   u32,
    pub should_die: bool,
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
    mut tick_rx:       broadcast::Receiver<Arc<WorldState>>,
    action_tx:         mpsc::Sender<BacteriaMsg>,
    mut genome_req_rx: mpsc::Receiver<GenomeRequest>,
) {
    loop {
        tokio::select! {
            // Tick normal: recibe snapshot, computa, responde
            result = tick_rx.recv() => {
                let snap = match result {
                    Ok(s)  => s,
                    Err(_) => break,  // board cerró el canal → apagar
                };

                // CPU-bound: step del transformer (síncrono dentro de la task async)
                let (new_pos, store, reward) = bacteria.step(
                    &snap.memory, &snap.quorum, &snap.food,
                );

                let should_die = bacteria.age > crate::world::MAX_AGE
                    || bacteria.is_starving();

                let msg = BacteriaMsg::Step(StepResult {
                    id,
                    new_pos,
                    store,
                    reward,
                    position:   bacteria.position,
                    fitness:    bacteria.fitness(),
                    diversity:  bacteria.recent_diversity(),
                    age:        bacteria.age,
                    cooldown:   bacteria.cooldown,
                    should_die,
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
                    state:       bacteria.state,
                    transformer: bacteria.transformer.clone(),
                    visits:      Box::new([0.0; crate::world::GRID_SIZE]),
                    recent:      bacteria.recent.clone(),
                    rewards:     bacteria.rewards.clone(),
                    age:         bacteria.age,
                    cooldown:    bacteria.cooldown,
                    rng:         crate::bacteria::XorShift32::new(bacteria.rng.state),
                });
                let _ = reply_tx.send(snapshot);
            }
        }
    }
}
