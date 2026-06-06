use std::collections::VecDeque;
use std::fs::File;
use std::io::Write;
use crate::actor::StepResult;
use crate::food::FoodAgent;
use crate::world::*;

pub fn save_history(history: &VecDeque<String>) {
    if let Ok(mut f) = File::create("ultimo_historico.txt") {
        for line in history { let _ = writeln!(f, "{}", line); }
    }
}

fn cell_char_from_results(
    pos:         usize,
    results:     &[StepResult],
    food_agents: &[FoodAgent],
    cv:          &[f32],
    quorum:      &[f32],
    food:        &[f32],
    colored:     bool,
) -> String {
    // Bacteria en esta celda
    if let Some(r) = results.iter().find(|r| r.position == pos) {
        let label = format!("{:<3}", format!("{:X}", r.id % 256));
        return if colored { format!("\x1B[1;92m{}\x1B[0m", label) } else { label };
    }
    // Food agent
    if food_agents.iter().any(|fa| fa.position == pos) {
        return if colored { "\x1B[1;33mF  \x1B[0m".into() } else { "F  ".into() };
    }
    // Quórum
    if quorum[pos] > 8.0 {
        return if colored { "\x1B[1;35m@  \x1B[0m".into() } else { "@  ".into() };
    }
    if quorum[pos] > QUORUM_THRESH {
        return if colored { "\x1B[1;36m*  \x1B[0m".into() } else { "*  ".into() };
    }
    // Señal de comida
    if food[pos] > FOOD_SIGNAL_THR {
        return if colored { "\x1B[1;32mf  \x1B[0m".into() } else { "f  ".into() };
    }
    // Historial de visitas
    if cv[pos] > 100.0 {
        return if colored { "\x1B[1;33m+  \x1B[0m".into() } else { "+  ".into() };
    }
    if cv[pos] > 20.0 {
        return if colored { "\x1B[37m.  \x1B[0m".into() } else { ".  ".into() };
    }
    "   ".into()
}

pub fn print_map(
    results:     &[StepResult],
    food_agents: &[FoodAgent],
    quorum:      &[f32],
    food:        &[f32],
    step:        u32,
    pop:         usize,
) {
    // Visitas combinadas aproximadas (usamos posición actual como proxy)
    let mut cv = vec![0f32; GRID_SIZE];
    for r in results { cv[r.position] += 1.0; }

    let zones = colony_zones(quorum);
    let q_max = quorum.iter().cloned().fold(0f32, f32::max);
    let f_max = food.iter().cloned().fold(0f32, f32::max);

    print!("\x1B[2J\x1B[H");
    println!("  paso={:>9}  pop={}  food={}  colonias={}  q_max={:.1}  f_max={:.1}",
        step, pop, food_agents.len(), zones, q_max, f_max);

    for (i, r) in results.iter().enumerate() {
        print!("  {:X}:{:>4} fit={:+.2}", r.id % 256, r.position, r.fitness);
        if (i + 1) % 4 == 0 { println!(); }
    }
    println!();

    print!("     ");
    for col in 0..GRID_SIDE { print!("{:<3}", format!("{:X}", col)); }
    println!();
    for row in 0..GRID_SIDE {
        print!(" {:2X}  ", row);
        for col in 0..GRID_SIDE {
            let pos = row * GRID_SIDE + col;
            print!("{}", cell_char_from_results(pos, results, food_agents, &cv, quorum, food, true));
        }
        println!();
    }
    println!("\n  0-F.. bacteria  F comida  @ núcleo  * colonia  f señal  + tibio  . frío");
}

pub fn save_snapshot(
    results:     &[StepResult],
    food_agents: &[FoodAgent],
    quorum:      &[f32],
    food:        &[f32],
    step:        u32,
    index:       usize,
    trigger:     &str,
) {
    let mut cv = vec![0f32; GRID_SIZE];
    for r in results { cv[r.position] += 1.0; }

    let zones = colony_zones(quorum);
    let q_max = quorum.iter().cloned().fold(0f32, f32::max);
    let f_max = food.iter().cloned().fold(0f32, f32::max);

    if let Ok(mut f_out) = File::create(format!("snapshot_{:02}.txt", index)) {
        let _ = writeln!(f_out,
            "Snap {:02} | {} | paso={} | pop={} | food={} | colonias={} | q_max={:.1} | f_max={:.1}",
            index, trigger, step, results.len(), food_agents.len(), zones, q_max, f_max);

        for r in results {
            let _ = writeln!(f_out,
                "  {:X}  pos={:>4}  fit={:+.3}  age={}  div={:.0}%",
                r.id % 256, r.position, r.fitness, r.age,
                r.diversity * 100.0);
        }
        for (i, fa) in food_agents.iter().enumerate() {
            let _ = writeln!(f_out,
                "  F{:X}  pos={:>4}  div={:>3.0}%  energy={:>6.1}  age={}",
                i, fa.position, fa.diversity() * 100.0, fa.energy, fa.age);
        }
        let _ = writeln!(f_out);
        let mut header = String::from("      ");
        for col in 0..GRID_SIDE { header.push_str(&format!("{:<3}", format!("{:X}", col))); }
        let _ = writeln!(f_out, "{}", header);
        for row in 0..GRID_SIDE {
            let mut line = format!(" {:2X}   ", row);
            for col in 0..GRID_SIDE {
                let pos = row * GRID_SIDE + col;
                line.push_str(&cell_char_from_results(
                    pos, results, food_agents, &cv, quorum, food, false,
                ));
            }
            let _ = writeln!(f_out, "{}", line);
        }
    }
}
