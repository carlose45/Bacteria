// Serialización de población: guarda/carga genomas al final/inicio de cada run.
// Formato binario sin dependencias externas.
//
// Layout por bacteria (406 × f32 = 1624 bytes):
//   w[12][12]=144 | w_in[12][14]=168 | bias[12] | tau_raw[12]
//   w_out[5][12]=60 | bias_out[5]
//   traits: metabolism, curiosity, sociability, selectivity, altruism
//
// Al cargar siempre se expande a MIN_SEED_POP individuos:
//   primeros N  → 70% herencia + 30% random  (conservan lo aprendido)
//   restantes   → 50% herencia + 50% random  (fuerzan diversidad)

use std::io::{Read, Write};
use crate::bacteria::{Bacteria, XorShift32};
use crate::ctrnn::{Ctrnn, crossover_ctrnn};
use crate::world::{GRID_SIZE, MAX_BACTERIA_ENERGY};

const MAGIC:        u32   = 0xBAC7_E21B;  // cambiado vs v2 para invalidar archivos viejos
const VERSION:      u32   = 3;
const MIN_SEED_POP: usize = 8;

pub const POPULATION_FILE: &str = "population.bin";

// ── Tipos internos ────────────────────────────────────────────────────────────

struct SavedGenome {
    ctrnn:       Ctrnn,
    metabolism:  f32,
    curiosity:   f32,
    sociability: f32,
    selectivity: f32,
    altruism:    f32,
}

// ── API pública ───────────────────────────────────────────────────────────────

pub fn save(path: &str, bacteria_list: &[&Bacteria]) {
    if bacteria_list.is_empty() {
        eprintln!("[genome] nada que guardar");
        return;
    }
    let Ok(mut f) = std::fs::File::create(path) else {
        eprintln!("[genome] no se pudo crear {path}");
        return;
    };
    let n = bacteria_list.len() as u32;
    let _ = f.write_all(&MAGIC.to_le_bytes());
    let _ = f.write_all(&VERSION.to_le_bytes());
    let _ = f.write_all(&n.to_le_bytes());
    for b in bacteria_list {
        write_ctrnn(&mut f, &b.ctrnn);
        write_f32(&mut f, b.metabolism);
        write_f32(&mut f, b.curiosity);
        write_f32(&mut f, b.sociability);
        write_f32(&mut f, b.selectivity);
        write_f32(&mut f, b.altruism);
    }
    eprintln!("[genome] guardada población ({n} bacterias) → {path}");
}

/// Carga y expande siempre a MIN_SEED_POP individuos.
/// Devuelve None si el archivo no existe o está corrupto.
pub fn load(path: &str, rng: &mut XorShift32) -> Option<Vec<Bacteria>> {
    let loaded = read_file(path)?;
    let n_loaded = loaded.len();

    let target = MIN_SEED_POP.max(n_loaded);
    let step   = GRID_SIZE / target;
    let mut out = Vec::with_capacity(target);

    let rf = |r: &mut XorShift32| r.next_u32() as f32 / u32::MAX as f32;
    let noise = |v: f32, scale: f32, r: &mut XorShift32| {
        v + (rf(r) - 0.5) * scale
    };

    for i in 0..target {
        let src = &loaded[i % n_loaded];

        // Individuo heredado directo: 70% herencia
        // Individuo extra (más allá de los guardados): 50% herencia con más diversidad
        let pct_heritage: u32 = if i < n_loaded { 70 } else { 50 };
        let fresh  = Ctrnn::new(rng);
        let ctrnn  = crossover_ctrnn(&src.ctrnn, &fresh, pct_heritage, rng);

        // Añadir variación a los rasgos
        let trait_noise = if i < n_loaded { 0.15 } else { 0.4 };
        let metabolism  = (noise(src.metabolism,  trait_noise * 0.01, rng)).max(0.0);
        let curiosity   = (noise(src.curiosity,   trait_noise,        rng)).max(0.1);
        let sociability = (noise(src.sociability, trait_noise,        rng)).max(0.1);
        let selectivity = noise(src.selectivity,  trait_noise,        rng).clamp(0.0, 2.0);
        let altruism    = (noise(src.altruism,    trait_noise,        rng)).max(0.1);

        out.push(Bacteria {
            position:    (i * step) % GRID_SIZE,
            ctrnn,
            visits:      Box::new([0.0; GRID_SIZE]),
            recent:      std::collections::VecDeque::with_capacity(500),
            rewards:     std::collections::VecDeque::with_capacity(200),
            age:         0,
            cooldown:    0,
            rng:         XorShift32::new(rng.next_u32()),
            energy:      MAX_BACTERIA_ENERGY,
            metabolism,
            curiosity,
            sociability,
            selectivity,
            altruism,
        });
    }

    eprintln!("[genome] cargados {n_loaded} genomas → expandidos a {} individuos ← {path}", out.len());
    Some(out)
}

// ── Lectura interna ───────────────────────────────────────────────────────────

fn read_file(path: &str) -> Option<Vec<SavedGenome>> {
    let mut data = Vec::new();
    std::fs::File::open(path).ok()?.read_to_end(&mut data).ok()?;
    let mut cur = 0usize;

    let magic   = read_u32(&data, &mut cur)?;
    let version = read_u32(&data, &mut cur)?;
    let n       = read_u32(&data, &mut cur)? as usize;

    if magic != MAGIC || version != VERSION {
        eprintln!("[genome] formato incompatible (magic={magic:#x} ver={version}), ignorando");
        return None;
    }

    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        out.push(SavedGenome {
            ctrnn:       read_ctrnn(&data, &mut cur)?,
            metabolism:  read_f32(&data, &mut cur)?,
            curiosity:   read_f32(&data, &mut cur)?,
            sociability: read_f32(&data, &mut cur)?,
            selectivity: read_f32(&data, &mut cur)?,
            altruism:    read_f32(&data, &mut cur)?,
        });
    }
    Some(out)
}

// ── Escritura binaria ─────────────────────────────────────────────────────────

fn write_f32(f: &mut std::fs::File, v: f32) {
    let _ = f.write_all(&v.to_le_bytes());
}

fn write_ctrnn(f: &mut std::fs::File, c: &Ctrnn) {
    for row in &c.w       { for &v in row { write_f32(f, v); } }
    for row in &c.w_in    { for &v in row { write_f32(f, v); } }
    for &v in &c.bias     { write_f32(f, v); }
    for &v in &c.tau_raw  { write_f32(f, v); }
    for row in &c.w_out   { for &v in row { write_f32(f, v); } }
    for &v in &c.bias_out { write_f32(f, v); }
}

// ── Lectura binaria ───────────────────────────────────────────────────────────

fn read_u32(data: &[u8], cur: &mut usize) -> Option<u32> {
    let b = data.get(*cur..*cur + 4)?;
    *cur += 4;
    Some(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

fn read_f32(data: &[u8], cur: &mut usize) -> Option<f32> {
    let b = data.get(*cur..*cur + 4)?;
    *cur += 4;
    Some(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

fn read_ctrnn(data: &[u8], cur: &mut usize) -> Option<Ctrnn> {
    let mut c = Ctrnn::zeroed();
    for row in c.w.iter_mut()       { for v in row { *v = read_f32(data, cur)?; } }
    for row in c.w_in.iter_mut()    { for v in row { *v = read_f32(data, cur)?; } }
    for v in c.bias.iter_mut()      { *v = read_f32(data, cur)?; }
    for v in c.tau_raw.iter_mut()   { *v = read_f32(data, cur)?; }
    for row in c.w_out.iter_mut()   { for v in row { *v = read_f32(data, cur)?; } }
    for v in c.bias_out.iter_mut()  { *v = read_f32(data, cur)?; }
    Some(c)
}
