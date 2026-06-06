// Coordinador central: ticks, tablero, comida, reproducción, snapshots

use std::sync::Arc;
use std::collections::VecDeque;
use std::time::Instant;
use tokio::sync::mpsc;
use crate::actor::{bacteria_loop, BacteriaMsg, GenomeRequest, StepResult};
use crate::bacteria::{crossover, Bacteria, XorShift32};
use crate::food::FoodAgent;
use crate::world::*;
use crate::display::{print_map, save_snapshot, save_history};

// Metadatos livianos que el board guarda por cada bacteria activa
struct BacteriaHandle {
    tick_tx:       mpsc::Sender<Arc<WorldState>>,
    genome_req_tx: mpsc::Sender<GenomeRequest>,
}

pub async fn board_loop(mut rng: XorShift32) {
    // ── Estado del tablero ────────────────────────────────────────────────────
    let mut world = WorldState::new();

    // ── Canales ───────────────────────────────────────────────────────────────
    // action_tx: todas las bacterias → board (canal compartido)
    let (action_tx, mut action_rx) = mpsc::channel::<BacteriaMsg>(MAX_POP * 4);

    // ── Handles activos ───────────────────────────────────────────────────────
    let mut handles: Vec<(usize, BacteriaHandle)> = Vec::new();
    let mut next_id: usize = 0;

    // ── Población inicial ─────────────────────────────────────────────────────
    let step_size = GRID_SIZE / 4;
    for i in 0..4 {
        let mut b = Bacteria::new(&mut rng);
        b.position = i * step_size;
        spawn_bacteria(b, &mut next_id, &mut handles, action_tx.clone());
    }

    // ── Agentes de comida (gestionados en el board, no son actores) ───────────
    let mut food_agents: Vec<FoodAgent> = (0..(MAX_FOOD / 3).max(3))
        .map(|_| FoodAgent::new(&mut rng))
        .collect();

    // ── Snapshots y visualización ─────────────────────────────────────────────
    let mut history: VecDeque<String> = VecDeque::with_capacity(100);
    let mut last_snapshot    = Instant::now();
    let mut last_print       = Instant::now();
    let mut last_event_snap  = Instant::now();
    let mut snapshot_count   = 0usize;
    let mut q_prev_above     = false;
    let mut step             = 0u32;

    // ── Loop principal ────────────────────────────────────────────────────────
    loop {
        let n_active = handles.len();
        if n_active == 0 {
            // Sin bacterias: crear una nueva
            let b = Bacteria::new(&mut rng);
            spawn_bacteria(b, &mut next_id, &mut handles, action_tx.clone());
            continue;
        }

        // 1. Snapshot inmutable del mundo → enviado directamente a cada bacteria
        let snap = Arc::new(world.clone());
        for (_, h) in &handles {
            let _ = h.tick_tx.send(Arc::clone(&snap)).await;
        }

        // 2. Recoger exactamente n_active acciones
        let mut results: Vec<StepResult> = Vec::with_capacity(n_active);
        let mut dead_ids: Vec<usize>     = Vec::new();

        for _ in 0..n_active {
            match action_rx.recv().await {
                Some(BacteriaMsg::Step(r)) => {
                    if r.should_die { dead_ids.push(r.id); }
                    results.push(r);
                }

                None => break,
            }
        }

        // 3. Actualizar crowding y aplicar movimientos al tablero
        world.crowding.iter_mut().for_each(|c| *c = 0);
        for r in &results {
            world.crowding[r.new_pos] = world.crowding[r.new_pos].saturating_add(1);
        }

        for r in &results {
            world.memory[r.new_pos] = r.store;
            let fv          = snap.food[r.new_pos];
            let food_factor = 1.0 + fv / (FOOD_SIGNAL_THR + fv);
            world.quorum[r.new_pos] += QUORUM_DEPOSIT * food_factor;
            for &n in &cardinal_neighbors(r.new_pos) {
                world.quorum[n] += QUORUM_DEPOSIT * 0.3;
            }
        }
        for q in world.quorum.iter_mut() { *q *= QUORUM_DECAY; }

        // 4. Agentes de comida
        let qm_snap = world.quorum.clone();
        for fa in food_agents.iter_mut() {
            let pos      = fa.step(&qm_snap);
            let nbrs     = cardinal_neighbors(pos);
            let q_max    = nbrs.iter().fold(qm_snap[pos], |a, &n| a.max(qm_snap[n]));
            let q_press  = (q_max / QUORUM_SAT_THRESH).min(1.0);
            fa.energy   -= q_press * FOOD_EATEN;
            world.food[pos]  += FOOD_SIGNAL;
            world.quorum[pos] += QUORUM_DEPOSIT * 1.0;
            for &n in &nbrs { world.food[n] += FOOD_SIGNAL * 0.5; }
        }

        // Bacterias comen comida
        for r in &results {
            if world.food[r.new_pos] > 0.0 {
                world.food[r.new_pos] = (world.food[r.new_pos] - FOOD_EATEN).max(0.0);
                for fa in food_agents.iter_mut() {
                    if fa.position == r.new_pos { fa.energy -= FOOD_EATEN; }
                }
            }
        }
        for fv in world.food.iter_mut() { *fv *= FOOD_DECAY; }

        // Ciclo de vida comida
        food_agents.retain(|fa| fa.energy > 0.0);
        if food_agents.is_empty() { food_agents.push(FoodAgent::new(&mut rng)); }
        let current_food = food_agents.len();
        let mut new_food: Vec<FoodAgent> = vec![];
        for fa in food_agents.iter_mut() {
            if fa.energy > 200.0 && current_food + new_food.len() < MAX_FOOD {
                fa.energy /= 2.0;
                let nbrs      = cardinal_neighbors(fa.position);
                let child_pos = nbrs[rng.next_u32() as usize % 4];
                new_food.push(FoodAgent { position: child_pos, energy: fa.energy,
                                          age: 0, recent: VecDeque::with_capacity(200),
                                          rng: crate::bacteria::XorShift32::new(rng.next_u32()) });
            }
        }
        food_agents.extend(new_food);

        // 5. Reproducción: detectar bacterias en misma posición
        let mut offspring: Vec<Bacteria> = vec![];
        {
            // Construir mapa posición → lista de (índice en handles, StepResult)
            let mut pos_map: std::collections::HashMap<usize, Vec<usize>> =
                std::collections::HashMap::new();
            for (i, r) in results.iter().enumerate() {
                pos_map.entry(r.position).or_default().push(i);
            }

            for (_pos, idxs) in &pos_map {
                if idxs.len() < 2 { continue; }
                for i in 0..idxs.len() {
                    for j in (i + 1)..idxs.len() {
                        let ra = &results[idxs[i]];
                        let rb = &results[idxs[j]];
                        if ra.cooldown > 0 || rb.cooldown > 0 { continue; }
                        if rng.next_u32() % 100 >= REPRODUCE_PROB { continue; }

                        // Pedir genomas a ambas bacterias via oneshot
                        let child = request_crossover(
                            ra.id, rb.id, &handles, &mut rng
                        ).await;
                        if let Some(c) = child { offspring.push(c); }
                    }
                }
            }
        }

        // Añadir hijos como nuevas tasks
        for child in offspring {
            if handles.len() >= MAX_POP {
                // Eliminar la bacteria con menor diversidad
                if let Some(worst_idx) = results.iter()
                    .min_by(|a, b| a.diversity.partial_cmp(&b.diversity).unwrap())
                    .and_then(|r| handles.iter().position(|(id, _)| *id == r.id))
                {
                    handles.remove(worst_idx);
                }
            }
            if handles.len() < MAX_POP {
                spawn_bacteria(child, &mut next_id, &mut handles, action_tx.clone());
            }
        }

        // Eliminar bacterias muertas
        handles.retain(|(id, _)| !dead_ids.contains(id));
        if handles.is_empty() { handles.retain(|_| false); } // ya vacío, ok

        // 6. Historial y visualización
        if step % 10 == 0 {
            let q_now = world.quorum.iter().cloned().fold(0f32, f32::max);
            let f_now = world.food.iter().cloned().fold(0f32, f32::max);
            let zones = colony_zones(&world.quorum);
            let avg_fit = if results.is_empty() { 0.0 }
                else { results.iter().map(|r| r.fitness).sum::<f32>() / results.len() as f32 };
            history.push_back(format!(
                "paso={:>9}  pop={}  food={}  col={}  q={:.1}  f={:.1}  fit={:+.3}",
                step, handles.len(), food_agents.len(), zones, q_now, f_now, avg_fit,
            ));
            if history.len() > 100 { history.pop_front(); }
        }

        // Visualización y snapshots usando una población "plana" para display
        // (construida desde los resultados del tick)
        let q_now   = world.quorum.iter().cloned().fold(0f32, f32::max);
        let q_above = q_now > QUORUM_EVENT_THRESH;

        if last_print.elapsed().as_millis() >= 500 {
            print_map(&results, &food_agents, &world.quorum, &world.food, step, handles.len());
            last_print = Instant::now();
        }

        if q_above && !q_prev_above && snapshot_count < MAX_SNAPS
            && last_event_snap.elapsed().as_secs() >= 5
        {
            snapshot_count += 1;
            save_snapshot(&results, &food_agents, &world.quorum, &world.food,
                          step, snapshot_count, "EVENTO");
            save_history(&history);
            last_event_snap = Instant::now();
        }
        q_prev_above = q_above;

        if snapshot_count < MAX_SNAPS && last_snapshot.elapsed().as_secs() >= SNAP_INTERVAL_SECS {
            snapshot_count += 1;
            save_snapshot(&results, &food_agents, &world.quorum, &world.food,
                          step, snapshot_count, "TIME");
            save_history(&history);
            last_snapshot = Instant::now();
        }

        if snapshot_count >= MAX_SNAPS { break; }
        step += 1;
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn spawn_bacteria(
    b:         Bacteria,
    next_id:   &mut usize,
    handles:   &mut Vec<(usize, BacteriaHandle)>,
    action_tx: mpsc::Sender<BacteriaMsg>,
) {
    let id     = *next_id;
    *next_id  += 1;
    let (tick_tx_b, tick_rx_b)         = mpsc::channel::<Arc<WorldState>>(4);
    let (genome_req_tx, genome_req_rx) = mpsc::channel::<GenomeRequest>(2);

    tokio::spawn(bacteria_loop(b, id, tick_rx_b, action_tx, genome_req_rx));
    handles.push((id, BacteriaHandle { tick_tx: tick_tx_b, genome_req_tx }));
}

async fn request_crossover(
    id_a:    usize,
    id_b:    usize,
    handles: &[(usize, BacteriaHandle)],
    rng:     &mut XorShift32,
) -> Option<Bacteria> {
    let ha = handles.iter().find(|(id, _)| *id == id_a).map(|(_, h)| &h.genome_req_tx)?;
    let hb = handles.iter().find(|(id, _)| *id == id_b).map(|(_, h)| &h.genome_req_tx)?;

    let (tx_a, rx_a) = tokio::sync::oneshot::channel::<Box<Bacteria>>();
    let (tx_b, rx_b) = tokio::sync::oneshot::channel::<Box<Bacteria>>();

    ha.send(tx_a).await.ok()?;
    hb.send(tx_b).await.ok()?;

    let ga = rx_a.await.ok()?;
    let gb = rx_b.await.ok()?;

    Some(crossover(&ga, &gb, rng))
}
