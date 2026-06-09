// Métricas científicas para análisis publicable — escribe observer.csv cada 1000 pasos
//
// Columnas:
//   A — campo estigmérgico: stigma_entropy, stigma_food_corr, stigma_coverage, stigma_bias
//   B — uso conductual:     stigma_align  (fracción de movimientos hacia mayor stigma)
//   C — selección:          fit_gini, var_meta, var_cur, var_soc, var_sel, var_alt

use std::fs::OpenOptions;
use std::io::Write;
use std::sync::OnceLock;
use crate::actor::StepResult;
use crate::world::{WorldState, GRID_SIZE};

static OBS_PATH: OnceLock<String> = OnceLock::new();

pub fn init(prefix: &str) {
    let path = format!("observer_{}.csv", prefix);
    if let Ok(mut f) = std::fs::File::create(&path) {
        let _ = writeln!(f,
            "step,stigma_entropy,stigma_food_corr,stigma_coverage,stigma_bias,\
             stigma_align,fit_gini,var_meta,var_cur,var_soc,var_sel,var_alt");
    } else { eprintln!("[observer] no se pudo crear {path}"); }
    OBS_PATH.set(path).ok();
}

pub fn rename_with_end(start: &str, end: &str) {
    let old = format!("observer_{}.csv", start);
    let new = format!("observer_{}_{}.csv", start, end);
    if let Err(e) = std::fs::rename(&old, &new) {
        eprintln!("[observer] no se pudo renombrar {old}: {e}");
    }
}

pub fn record(
    step:           u32,
    world:          &WorldState,
    results:        &[StepResult],
    stigma_aligned: u32,   // movimientos hacia mayor stigma en la ventana
    stigma_moved:   u32,   // total movimientos en la ventana
) {
    // ── A: campo estigmérgico ─────────────────────────────────────────────────
    let stigma_entropy   = entropy(&world.stigma);
    let stigma_food_corr = pearson(&world.stigma, &world.food);
    let stigma_coverage  = world.stigma.iter().filter(|&&s| s > 1.0).count() as f32
                           / GRID_SIZE as f32;

    // Stigma bias: ¿están las bacterias donde hay más memoria?
    let mean_all = world.stigma.iter().sum::<f32>() / GRID_SIZE as f32;
    let mean_occupied = if results.is_empty() { mean_all } else {
        results.iter().map(|r| world.stigma[r.new_pos]).sum::<f32>() / results.len() as f32
    };
    let stigma_bias = if mean_all > 1e-10 { mean_occupied / mean_all } else { 1.0 };

    // ── B: uso conductual ─────────────────────────────────────────────────────
    // 0.5 = aleatorio, >0.5 = siguen memoria, <0.5 = la evitan
    let stigma_align = if stigma_moved > 0 {
        stigma_aligned as f32 / stigma_moved as f32
    } else { 0.5 };

    // ── C: selección evolutiva ────────────────────────────────────────────────
    let fitness_vals: Vec<f32> = results.iter().map(|r| r.fitness.max(0.0)).collect();
    let fit_gini = gini(&fitness_vals);

    let var_meta = var(results.iter().map(|r| r.metabolism));
    let var_cur  = var(results.iter().map(|r| r.curiosity));
    let var_soc  = var(results.iter().map(|r| r.sociability));
    let var_sel  = var(results.iter().map(|r| r.selectivity));
    let var_alt  = var(results.iter().map(|r| r.altruism));

    let path = OBS_PATH.get().map(|s| s.as_str()).unwrap_or("observer.csv");
    let Ok(mut f) = OpenOptions::new().append(true).open(path) else { return };
    writeln!(f,
        "{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.6},{:.4},{:.4},{:.4},{:.4}",
        step,
        stigma_entropy, stigma_food_corr, stigma_coverage, stigma_bias,
        stigma_align, fit_gini,
        var_meta, var_cur, var_soc, var_sel, var_alt,
    ).unwrap();
}

// ── Estadísticas ──────────────────────────────────────────────────────────────

fn entropy(field: &[f32]) -> f32 {
    let total: f32 = field.iter().sum();
    if total < 1e-10 { return 0.0; }
    -field.iter()
        .filter(|&&v| v > 0.0)
        .map(|&v| { let p = v / total; p * p.ln() })
        .sum::<f32>()
}

fn pearson(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len() as f32;
    let ma = a.iter().sum::<f32>() / n;
    let mb = b.iter().sum::<f32>() / n;
    let num: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - ma) * (y - mb)).sum();
    let da  = a.iter().map(|x| (x - ma).powi(2)).sum::<f32>().sqrt();
    let db  = b.iter().map(|y| (y - mb).powi(2)).sum::<f32>().sqrt();
    if da * db < 1e-10 { 0.0 } else { num / (da * db) }
}

fn gini(values: &[f32]) -> f32 {
    if values.is_empty() { return 0.0; }
    let mut s = values.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n   = s.len() as f32;
    let sum: f32 = s.iter().sum();
    if sum < 1e-10 { return 0.0; }
    let weighted: f32 = s.iter().enumerate().map(|(i, v)| (2 * (i + 1)) as f32 * v).sum();
    (weighted / (n * sum)) - (n + 1.0) / n
}

fn var(iter: impl Iterator<Item = f32>) -> f32 {
    let v: Vec<f32> = iter.collect();
    let n = v.len() as f32;
    if n == 0.0 { return 0.0; }
    let mean = v.iter().sum::<f32>() / n;
    v.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n
}
