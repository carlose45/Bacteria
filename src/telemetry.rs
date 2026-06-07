use std::fs::OpenOptions;
use std::io::Write;
use crate::actor::StepResult;
use crate::food::FoodAgent;
use crate::world::WorldState;

pub fn init() {
    match std::fs::File::create("telemetry.csv") {
        Ok(mut f) => { let _ = writeln!(f, "step,pop,food_n,food_e_avg,food_age_avg,food_gen,\
                 q_max,q_avg,f_max,colonias,\
                 fit_avg,fit_min,fit_max,div_avg,hunger_avg,\
                 metabolism_avg,curiosity_avg,sociability_avg,selectivity_avg,altruism_avg"); }
        Err(e) => eprintln!("[telemetry] no se pudo crear telemetry.csv: {e}"),
    }
    match std::fs::File::create("food_lineage.csv") {
        Ok(mut f) => { let _ = writeln!(f, "step,food_gen,food_age,food_energy,food_diversity"); }
        Err(e) => eprintln!("[telemetry] no se pudo crear food_lineage.csv: {e}"),
    }
}

pub fn record_step(
    step:        u32,
    food_gen:    u32,
    food_agents: &[FoodAgent],
    world:       &WorldState,
    results:     &[StepResult],
) {
    let pop    = results.len();
    let food_n = food_agents.len();

    let food_e_avg = if food_n > 0 {
        food_agents.iter().map(|fa| fa.energy).sum::<f32>() / food_n as f32
    } else { 0.0 };

    let food_age_avg = if food_n > 0 {
        food_agents.iter().map(|fa| fa.age as f32).sum::<f32>() / food_n as f32
    } else { 0.0 };

    let q_max    = world.quorum.iter().cloned().fold(0f32, f32::max);
    let q_avg    = world.quorum.iter().sum::<f32>() / world.quorum.len() as f32;
    let f_max    = world.food.iter().cloned().fold(0f32, f32::max);
    let colonias = world.quorum.iter().filter(|&&q| q > crate::world::QUORUM_THRESH).count();

    let (fit_avg, fit_min, fit_max, div_avg, hunger_avg,
         metabolism_avg, curiosity_avg, sociability_avg, selectivity_avg, altruism_avg) = if results.is_empty() {
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    } else {
        let n = results.len() as f32;
        (
            results.iter().map(|r| r.fitness).sum::<f32>() / n,
            results.iter().map(|r| r.fitness).fold(f32::INFINITY,     f32::min),
            results.iter().map(|r| r.fitness).fold(f32::NEG_INFINITY, f32::max),
            results.iter().map(|r| r.diversity).sum::<f32>() / n,
            results.iter().map(|r| r.hunger).sum::<f32>() / n,
            results.iter().map(|r| r.metabolism).sum::<f32>() / n,
            results.iter().map(|r| r.curiosity).sum::<f32>() / n,
            results.iter().map(|r| r.sociability).sum::<f32>() / n,
            results.iter().map(|r| r.selectivity).sum::<f32>() / n,
            results.iter().map(|r| r.altruism).sum::<f32>() / n,
        )
    };

    let Ok(mut f) = OpenOptions::new().append(true).open("telemetry.csv") else { return };
    writeln!(f,
        "{},{},{},{:.2},{:.1},{},{:.3},{:.5},{:.2},{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.6},{:.4},{:.4},{:.4},{:.4}",
        step, pop, food_n, food_e_avg, food_age_avg, food_gen,
        q_max, q_avg, f_max, colonias,
        fit_avg, fit_min, fit_max, div_avg, hunger_avg,
        metabolism_avg, curiosity_avg, sociability_avg, selectivity_avg, altruism_avg
    ).unwrap();
}

pub fn record_food_death(step: u32, food_gen: u32, fa: &FoodAgent) {
    let Ok(mut f) = OpenOptions::new().append(true).open("food_lineage.csv") else { return };
    writeln!(f, "{},{},{},{:.2},{:.4}",
        step, food_gen, fa.age, fa.energy, fa.diversity()
    ).unwrap();
}
