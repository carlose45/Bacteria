# ============================================================
# Sistema B — Observador Científico del Ecosistema Evolutivo
#
# Uso:
#   Rscript analysis/report.R                  # run más reciente
#   Rscript analysis/report.R 2026-06-07_04-25 # prefijo específico
#
# Dependencias: ggplot2, dplyr, scales, patchwork  (sin tidyr/stringr/stringi)
# ============================================================

suppressPackageStartupMessages({
  pkgs <- c("ggplot2", "dplyr", "scales", "patchwork")
  miss <- pkgs[!sapply(pkgs, requireNamespace, quietly = TRUE)]
  if (length(miss)) install.packages(miss, repos = "https://cloud.r-project.org")
  invisible(lapply(pkgs, library, character.only = TRUE))
})

# ── Localizar archivos ──────────────────────────────────────────────────────

find_files <- function(prefix = NULL, dir = ".") {
  if (!is.null(prefix)) {
    tele <- list.files(dir, pattern = paste0("telemetry_.*", prefix, ".*\\.csv"), full.names = TRUE)
  } else {
    tele <- list.files(dir, pattern = "^telemetry_.*\\.csv$", full.names = TRUE)
  }
  if (!length(tele)) stop("No se encontró telemetry_*.csv en: ", dir)
  latest <- tele[which.max(file.mtime(tele))]
  pfx    <- sub("^.*telemetry_(.+)\\.csv$", "\\1", latest)
  list(
    telemetry    = latest,
    food_lineage = file.path(dir, paste0("food_lineage_", pfx, ".csv")),
    observer     = file.path(dir, paste0("observer_",     pfx, ".csv")),
    prefix       = pfx
  )
}

args    <- commandArgs(trailingOnly = TRUE)
run_dir <- "."
pfx_arg <- if (length(args) > 0) args[1] else NULL
files   <- find_files(pfx_arg, run_dir)
cat("Run:", files$prefix, "\n")

# ── Cargar datos ────────────────────────────────────────────────────────────

tele <- read.csv(files$telemetry)
cat(sprintf("Telemetría:    %d filas  |  paso máximo: %s\n", nrow(tele), max(tele$step)))

food <- if (file.exists(files$food_lineage)) {
  df <- read.csv(files$food_lineage)
  cat(sprintf("Food lineage:  %d muertes  |  gen máxima: %s\n", nrow(df), max(df$food_gen)))
  df
} else { cat("food_lineage no encontrado\n"); NULL }

obs <- if (file.exists(files$observer)) {
  df <- read.csv(files$observer)
  cat(sprintf("Observer:      %d filas\n", nrow(df)))
  df
} else { cat("observer no encontrado\n"); NULL }

# ── Tema y paleta ───────────────────────────────────────────────────────────

TRAITS     <- c("metabolism", "curiosity", "sociability", "selectivity", "altruism")
TRAIT_COLS <- paste0(TRAITS, "_avg")
COLORS     <- c(metabolism  = "#E64B35", curiosity   = "#4DBBD5",
                sociability = "#00A087", selectivity = "#3C5488", altruism = "#F39B7F")

th <- function(legend = "bottom") {
  theme_bw(base_size = 10) +
  theme(panel.grid.minor = element_blank(),
        strip.background  = element_rect(fill = "grey95"),
        legend.position   = legend,
        plot.title        = element_text(size = 11, face = "bold"))
}

# Pivot largo sin tidyr — base R stack()
pivot_traits <- function(df, cols, names_to = "trait", values_to = "value") {
  long <- do.call(rbind, lapply(cols, function(col) {
    data.frame(step = df$step, trait = col, value = df[[col]], stringsAsFactors = FALSE)
  }))
  colnames(long)[2:3] <- c(names_to, values_to)
  long
}

# ── FIGURA 1: Ecosistema — dinámica general ─────────────────────────────────

p1a <- ggplot(tele, aes(step)) +
  geom_ribbon(aes(ymin = fit_min, ymax = fit_max), alpha = .15, fill = "#3C5488") +
  geom_line(aes(y = fit_avg), color = "#3C5488", linewidth = .7) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey60") +
  labs(title = "Fitness bacteria", x = NULL, y = "fitness") + th("none")

p1b <- ggplot(tele, aes(step, food_n)) +
  geom_step(color = "#E64B35", linewidth = .7) +
  scale_y_continuous(breaks = 0:6) +
  labs(title = "Agentes de comida (food_n)", x = NULL, y = "n") + th("none")

p1c <- ggplot(tele, aes(step, hunger_avg)) +
  geom_line(color = "#F39B7F", linewidth = .7) +
  scale_y_continuous(labels = percent_format()) +
  labs(title = "Hambre media bacteria", x = NULL, y = "hunger") + th("none")

traits_long <- pivot_traits(tele, TRAIT_COLS)
traits_long$trait <- sub("_avg$", "", traits_long$trait)

p1d <- ggplot(traits_long, aes(step, value, color = trait)) +
  geom_line(linewidth = .6, alpha = .85) +
  scale_color_manual(values = COLORS) +
  labs(title = "Rasgos hereditarios", x = "paso", y = "valor medio", color = NULL) +
  th()

fig1 <- (p1a / p1b / p1c / p1d) +
  plot_annotation(title = "Figura 1 — Dinámica del ecosistema",
                  theme = theme(plot.title = element_text(size = 13, face = "bold")))

# ── FIGURA 2: Red Queen — longevidad de comida ──────────────────────────────

fig2 <- NULL
if (!is.null(food) && nrow(food) > 10) {

  food$epoch <- paste0("gen ", floor(food$food_gen / 200) * 200)

  p2a <- ggplot(food, aes(food_age, fill = epoch, color = epoch)) +
    geom_density(alpha = .25, linewidth = .5) +
    scale_x_log10(labels = comma_format()) +
    labs(title = "Distribución de longevidad por época (log)",
         x = "edad al morir (pasos)", y = "densidad", fill = "época", color = "época") +
    th()

  # Supervivencia empírica sin dplyr group_by para evitar deps
  epochs  <- unique(food$epoch)
  surv_list <- lapply(epochs, function(ep) {
    sub_df  <- food[food$epoch == ep, ]
    ages    <- sort(sub_df$food_age)
    n       <- length(ages)
    data.frame(epoch = ep, food_age = ages, surv = 1 - seq_len(n) / n)
  })
  surv_data <- do.call(rbind, surv_list)

  p2b <- ggplot(surv_data, aes(food_age, surv, color = epoch)) +
    geom_step(linewidth = .6) +
    scale_x_log10(labels = comma_format()) +
    scale_y_continuous(labels = percent_format()) +
    labs(title = "Supervivencia empírica por época",
         x = "edad (pasos)", y = "P(vivir > t)", color = "época") +
    th()

  food$gen_bin <- floor(food$food_gen / 50) * 50
  food_roll <- aggregate(food_age ~ gen_bin, food,
                         FUN = function(x) c(max = max(x), avg = mean(x), med = median(x)))
  food_roll <- data.frame(gen_bin  = food_roll$gen_bin,
                          max_age  = food_roll$food_age[, "max"],
                          avg_age  = food_roll$food_age[, "avg"],
                          med_age  = food_roll$food_age[, "med"])

  p2c <- ggplot(food_roll, aes(gen_bin)) +
    geom_ribbon(aes(ymin = med_age, ymax = max_age), alpha = .2, fill = "#E64B35") +
    geom_line(aes(y = avg_age), color = "#E64B35", linewidth = .7) +
    geom_line(aes(y = max_age), color = "#E64B35", linewidth = .4, linetype = "dashed") +
    scale_y_log10(labels = comma_format()) +
    labs(title = "Longevidad por generación",
         x = "generación", y = "edad (pasos, log)") + th("none")

  p2d <- ggplot(food, aes(food_gen, food_diversity)) +
    geom_point(alpha = .2, size = .6, color = "#4DBBD5") +
    geom_smooth(method = "loess", span = .15, color = "#3C5488",
                se = TRUE, linewidth = .8) +
    scale_y_continuous(limits = c(0, 1)) +
    labs(title = "Diversidad exploratoria al morir",
         x = "generación", y = "diversidad") + th("none")

  fig2 <- (p2a + p2b) / (p2c + p2d) +
    plot_annotation(title = "Figura 2 — Dinámica Red Queen (comida)",
                    theme = theme(plot.title = element_text(size = 13, face = "bold")))
}

# ── FIGURA 3: Estigmergia ───────────────────────────────────────────────────

fig3 <- NULL
if (!is.null(obs) && nrow(obs) > 10) {

  p3a <- ggplot(obs, aes(step, stigma_entropy)) +
    geom_line(color = "#8491B4", linewidth = .6) +
    labs(title = "Entropía del campo stigma",
         subtitle = "Alta = difuso | Baja = concentrado",
         x = NULL, y = "entropía (nats)") + th("none")

  p3b <- ggplot(obs, aes(step, stigma_food_corr)) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "grey60") +
    geom_line(color = "#00A087", linewidth = .6) +
    geom_smooth(method = "loess", span = .15, color = "#3C5488",
                se = TRUE, linewidth = .8) +
    scale_y_continuous(limits = c(-1, 1)) +
    labs(title = "Correlación stigma ↔ comida",
         subtitle = "Positivo = la memoria sigue a la comida",
         x = NULL, y = "Pearson r") + th("none")

  p3c <- ggplot(obs, aes(step, stigma_align)) +
    geom_hline(yintercept = .5, linetype = "dashed", color = "grey60", linewidth = .8) +
    geom_ribbon(aes(ymin = .5, ymax = stigma_align), alpha = .2, fill = "#F39B7F") +
    geom_line(color = "#E64B35", linewidth = .7) +
    scale_y_continuous(labels = percent_format(), limits = c(0, 1)) +
    labs(title = "Alineación conductual con stigma",
         subtitle = "50% = aleatorio | >50% = siguen la memoria",
         x = "paso", y = "% hacia mayor stigma") + th("none")

  # Cobertura y bias — pivot manual
  cov_df <- rbind(
    data.frame(step = obs$step, metric = "cobertura", value = obs$stigma_coverage),
    data.frame(step = obs$step, metric = "sesgo",     value = obs$stigma_bias)
  )

  p3d <- ggplot(cov_df, aes(step, value, color = metric)) +
    geom_line(linewidth = .6) +
    geom_hline(yintercept = 1, linetype = "dashed", color = "grey70") +
    scale_color_manual(values = c(cobertura = "#4DBBD5", sesgo = "#E64B35")) +
    labs(title = "Cobertura y sesgo del campo stigma",
         subtitle = "Sesgo > 1 = bacterias en zonas de alta memoria",
         x = "paso", y = "valor", color = NULL) + th()

  fig3 <- (p3a + p3b) / (p3c + p3d) +
    plot_annotation(title = "Figura 3 — Análisis estigmérgico",
                    theme = theme(plot.title = element_text(size = 13, face = "bold")))
}

# ── FIGURA 4: Presión selectiva ─────────────────────────────────────────────

fig4 <- NULL
if (!is.null(obs) && nrow(obs) > 10) {

  var_map <- c(var_meta = "metabolism", var_cur = "curiosity",
               var_soc  = "sociability", var_sel = "selectivity", var_alt = "altruism")
  var_cols <- names(var_map)

  vars_long <- do.call(rbind, lapply(var_cols, function(col) {
    data.frame(step = obs$step, trait = var_map[[col]], variance = obs[[col]],
               stringsAsFactors = FALSE)
  }))

  p4a <- ggplot(vars_long, aes(step, variance, color = trait)) +
    geom_line(linewidth = .6, alpha = .8) +
    scale_color_manual(values = COLORS) +
    labs(title = "Varianza de rasgos",
         subtitle = "Baja = selección convergió",
         x = NULL, y = "varianza", color = NULL) + th()

  p4b <- ggplot(obs, aes(step, fit_gini)) +
    geom_line(color = "#3C5488", linewidth = .7) +
    geom_smooth(method = "loess", span = .15, se = TRUE,
                color = "#E64B35", fill = "#E64B35", alpha = .15, linewidth = .5) +
    scale_y_continuous(limits = c(0, 1)) +
    labs(title = "Gini del fitness",
         subtitle = "0 = equidad | 1 = dominante único",
         x = "paso", y = "Gini") + th("none")

  n_tail  <- max(1, round(nrow(tele) * .1))
  final   <- tail(tele, n_tail)
  cor_mat <- cor(final[, TRAIT_COLS], use = "complete.obs")
  colnames(cor_mat) <- rownames(cor_mat) <- TRAITS

  cor_df <- as.data.frame(as.table(cor_mat))
  colnames(cor_df) <- c("trait1", "trait2", "r")
  cor_df$trait1 <- factor(cor_df$trait1, TRAITS)
  cor_df$trait2 <- factor(cor_df$trait2, TRAITS)

  p4c <- ggplot(cor_df, aes(trait1, trait2, fill = r)) +
    geom_tile(color = "white") +
    geom_text(aes(label = round(r, 2)), size = 3) +
    scale_fill_gradient2(low = "#E64B35", mid = "white", high = "#3C5488",
                         midpoint = 0, limits = c(-1, 1)) +
    labs(title = "Correlación entre rasgos (último 10%)",
         x = NULL, y = NULL, fill = "r") +
    th("none") + theme(axis.text.x = element_text(angle = 30, hjust = 1))

  fig4 <- (p4a / p4b) | p4c +
    plot_annotation(title = "Figura 4 — Presión selectiva",
                    theme = theme(plot.title = element_text(size = 13, face = "bold")))
}

# ── TABLA DE RESUMEN ────────────────────────────────────────────────────────

cols_sum <- c("fit_avg", "hunger_avg", "metabolism_avg", "curiosity_avg",
              "sociability_avg", "selectivity_avg", "altruism_avg")
n10 <- max(1, round(nrow(tele) * .1))
sum_tbl <- rbind(
  cbind(periodo = "inicio (10%)", as.data.frame(t(colMeans(head(tele, n10)[, cols_sum])))),
  cbind(periodo = "final (10%)",  as.data.frame(t(colMeans(tail(tele, n10)[, cols_sum]))))
)
cat("\n── Resumen estadístico ─────────────────────────────\n")
print(sum_tbl, digits = 4)

# ── GENERAR PDF ─────────────────────────────────────────────────────────────

out_dir  <- "analysis"
if (!dir.exists(out_dir)) dir.create(out_dir)
out_file <- file.path(out_dir, paste0("report_", files$prefix, ".pdf"))
pdf(out_file, width = 12, height = 9)

print(fig1)
if (!is.null(fig2)) print(fig2)
if (!is.null(fig3)) print(fig3)
if (!is.null(fig4)) print(fig4)

dev.off()
cat("\nReporte guardado en:", out_file, "\n")
