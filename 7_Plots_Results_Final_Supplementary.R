# =============================================================================
# 7_Plots_Results_Final_Supplementary.R
# Supplementary figures for the Results section.
#
# Figures produced:
#   S1. Barcelona weather summary (2019-2024): weekly rainfall + daily mean temperature
#   S2. Driver correlation matrix: Spearman correlation among the 5 final predictors
#   S3. Active drains yearly time series per zone (reform-driven stepwise drops)
#   S4. Pairwise inter-drain distance distributions per zone, initial (2019) vs final (2024)
#   S5. COMBINED NETWORK-STRUCTURE FIGURE (S3 center + S4 satellites)
#
# Data source: Figures/Simulations_Results.csv, Figures/Predictors.csv,
#              Dades_netes/Meteo_BCN_2019_2024.xlsx, Dades_netes/Unique_Items.csv,
#              Parameters/ZR_Distance_Matrix.txt
# =============================================================================

rm(list = ls())

library(dplyr)
library(ggplot2)
library(tidyverse)
library(patchwork)
library(minpack.lm)

Sys.setlocale("LC_TIME", "C")


# =============================================================================
# PARAMETERS
# =============================================================================

SAVE_PLOTS <- FALSE
SEASON_DAYS <- 244     # April 1 to Nov 30

alpha_min  <- 0.1
alpha_best <- 0.5
alpha_even <- 0.8

RANGE_FRAC <- 0.15
MIN_K <- 3

script_dir <- "."
save_dir <- file.path(script_dir, "Figures/Optimization/Supplementary")
temp_dir <- tempdir()
dir.create(save_dir, showWarnings = FALSE, recursive = TRUE)

# Zone short names
zone_short <- c(
  "Jardins Mossèn Cinto Verdaguer"       = "Cinto Verdaguer",
  "Jardins de Vil\u00b7la Amèlia - Cecília"   = "Vil·la Amèlia",
  "Jardins del Teatre Grec"              = "Teatre Grec",
  "Jardins del Turó del Putxet"          = "Turó Putxet",
  "Parc de la Ciutadella"                = "Ciutadella",
  "Parc de la Guineueta"                 = "Guineueta",
  "Parc del Turó de la Peira"            = "Turó Peira"
)

# Zone colors - using both short and full names for compatibility
zone_colors_short <- c(
  "Ciutadella"      = "#FDB462",
  "Turó Peira"      = "#FCCDE5",
  "Cinto Verdaguer" = "#80B1D3",
  "Teatre Grec"     = "#BEBADA",
  "Turó Putxet"     = "#FB8072",
  "Vil·la Amèlia"   = "#8DD3C7",
  "Guineueta"       = "#B3DE69"
)

zone_colors <- c(
  "Parc de la Ciutadella"                = "#FDB462",
  "Parc del Turó de la Peira"            = "#FCCDE5",
  "Jardins Mossèn Cinto Verdaguer"       = "#80B1D3",
  "Jardins del Teatre Grec"              = "#BEBADA",
  "Jardins del Turó del Putxet"          = "#FB8072",
  "Jardins de Vil·la Amèlia - Cecília"   = "#8DD3C7",
  "Parc de la Guineueta"                 = "#B3DE69"
)


# =============================================================================
# HELPER: Save and display
# =============================================================================

save_and_display <- function(p, filename, w = 10, h = 8) {
  temp_path <- file.path(temp_dir, filename)
  save_path <- file.path(save_dir, filename)

  # cairo_pdf keeps the Catalan and Greek characters in the labels; NULL lets
  # ggsave pick the device from the extension for any non-pdf output
  dev <- if (grepl("\\.pdf$", filename)) cairo_pdf else NULL

  ggsave(temp_path, plot = p, width = w, height = h, dpi = 300, bg = "white", device = dev)
  system2("xdg-open", temp_path, wait = FALSE)
  Sys.sleep(0.5)

  if (SAVE_PLOTS) {
    ggsave(save_path, plot = p, width = w, height = h, dpi = 300, bg = "white", device = dev)
  }
}


# =============================================================================
# 1. LOAD & FILTER DATA
# =============================================================================

cat("\n========================================\n")
cat("LOADING AND FILTERING DATA\n")
cat("========================================\n\n")

all_data <- read.csv(file.path(script_dir, "Figures/Simulations_Results.csv"))
all_data <- all_data %>% mutate(Beta = as.numeric(Beta) / 10)
all_data <- all_data %>% mutate(Periodicity = as.numeric(Periodicity))

Results <- all_data %>% filter(!ID %in% c("ASPB", "Untreated"))
ASPB_simulations <- all_data %>% filter(ID == "ASPB")
Untreated_simulations <- all_data %>% filter(ID == "Untreated")
rm(all_data)

# Keep simulations within RANGE_FRAC of the best score. See main script for rationale.
top_sims <- Results %>%
  group_by(Year, Beta, ID) %>%
  summarise(J = first(J), INT = sum(INT), .groups = "drop") %>%
  group_by(Year, Beta) %>%
  mutate(
    score = if_else(Beta == 0.0, INT, J),
    threshold = min(score) + RANGE_FRAC * (max(score) - min(score)),
    in_range = score <= threshold,
    rank_score = row_number(score),
    keep = ifelse(rep(sum(in_range) >= MIN_K, n()), in_range, rank_score <= MIN_K)
  ) %>%
  filter(keep) %>%
  ungroup() %>%
  select(Year, Beta, ID)

Results <- Results %>% semi_join(top_sims, by = c("Year", "Beta", "ID"))

cat(sprintf("Optimized rows: %d | ASPB rows: %d | Untreated rows: %d\n",
            nrow(Results), nrow(ASPB_simulations), nrow(Untreated_simulations)))

# Untreated lookup
untreated_peryear <- Untreated_simulations %>%
  select(Zone, Year, Zone_INT) %>%
  rename(Zone_INT_Untreated = Zone_INT)

# Predictors
predictors <- read.csv(file.path(script_dir, "Figures/Predictors.csv"),
                       stringsAsFactors = FALSE)
cat(sprintf("Loaded %d rows from Predictors.csv\n", nrow(predictors)))

N_ZONES <- 7

# Canonical zone order for figures
zone_order <- c("Ciutadella", "Turó Peira", "Cinto Verdaguer",
                "Teatre Grec", "Turó Putxet", "Vil·la Amèlia", "Guineueta")


# =============================================================================
# S1. BARCELONA WEATHER SUMMARY (2019–2024)
# =============================================================================
# Average the 5 MeteoCAT stations by date to produce a single city-wide series.
# Weekly rainfall bars + daily mean temperature line on a continuous x-axis.

cat("\n========================================\n")
cat("S1. BARCELONA WEATHER SUMMARY\n")
cat("========================================\n\n")

library(readxl)
library(lubridate)

meteo_raw <- read_excel(file.path(script_dir, "Dades_netes/Meteo_BCN_2019_2024.xlsx"))

meteo_city <- meteo_raw %>%
  mutate(
    Fecha    = as.Date(Fecha),
    Mean_T   = as.numeric(Mean_T),
    Rainfall = as.numeric(Rainfall)
  ) %>%
  filter(!is.na(Mean_T) | !is.na(Rainfall)) %>%
  group_by(Fecha) %>%
  summarise(
    Mean_T   = mean(Mean_T,   na.rm = TRUE),
    Rainfall = mean(Rainfall, na.rm = TRUE),
    .groups  = "drop"
  ) %>%
  mutate(Week = floor_date(Fecha, "week", week_start = 1))

# Weekly rainfall sums
rainfall_weekly <- meteo_city %>%
  group_by(Week) %>%
  summarise(Weekly_Rainfall = sum(Rainfall, na.rm = TRUE), .groups = "drop")

# Global scaling ratio so temperature fits on the rainfall axis
global_max_rain <- max(rainfall_weekly$Weekly_Rainfall, na.rm = TRUE)
global_max_temp <- max(meteo_city$Mean_T, na.rm = TRUE)
ratio <- global_max_rain / global_max_temp

meteo_city <- meteo_city %>% mutate(Scaled_Temp = Mean_T * ratio)

# Year-start date-breaks for x-axis labels
year_starts <- as.Date(paste0(2019:2024, "-01-01"))

p_weather <- ggplot() +
  geom_col(data = rainfall_weekly,
           aes(x = Week, y = Weekly_Rainfall),
           fill = "#53A1C7", alpha = 0.5, width = 7) +
  geom_line(data = meteo_city,
            aes(x = Fecha, y = Scaled_Temp),
            color = "firebrick", linewidth = 0.7, alpha = 0.85) +
  # Shaded seasons (April–November) for each year
  geom_rect(data = data.frame(
    xmin = as.Date(paste0(2019:2024, "-04-01")),
    xmax = as.Date(paste0(2019:2024, "-11-30"))
  ),
  aes(xmin = xmin, xmax = xmax, ymin = -Inf, ymax = Inf),
  fill = "grey90", alpha = 0.3, inherit.aes = FALSE) +
  scale_x_date(
    breaks       = year_starts,
    date_labels  = "%Y",
    expand       = expansion(add = c(10, 10))
  ) +
  scale_y_continuous(
    name     = expression("Weekly rainfall (mm)"),
    limits   = c(0, global_max_rain * 1.10),
    expand   = c(0, 0),
    sec.axis = sec_axis(~ . / ratio,
                        name   = "Mean temperature (\u00b0C)",
                        breaks = scales::pretty_breaks(n = 5))
  ) +
  labs(x = NULL) +
  theme_minimal() +
  theme(
    axis.title.y.left  = element_text(size = 14, color = "black"),
    axis.text.y.left   = element_text(size = 12, color = "black"),
    axis.title.y.right = element_text(size = 14, color = "black"),
    axis.text.y.right  = element_text(size = 12, color = "black"),
    axis.text.x        = element_text(size = 13, color = "black"),
    panel.grid.minor   = element_blank(),
    panel.grid.major.x = element_blank()
  )

save_and_display(p_weather, "FigS_Weather_Barcelona.pdf", w = 12, h = 5)

# Summary table printed to console for LaTeX
# Uses meteo_city (per-day station averages) so station dropouts don't distort
# the annual totals — each day is weighted equally regardless of how many
# stations reported that day.
cat("\n--- Annual weather summary (city average) ---\n")
meteo_city %>%
  mutate(Year = year(Fecha)) %>%
  group_by(Year) %>%
  summarise(
    Annual_Rainfall_mm = sprintf("%.1f", round(sum(Rainfall, na.rm = TRUE), 1)),
    Mean_Temperature_C = sprintf("%.1f", round(mean(Mean_T,  na.rm = TRUE), 1)),
    .groups = "drop"
  ) %>%
  filter(Year >= 2019) %>%
  print()

cat("Weather figure saved.\n")


# =============================================================================
# S2. DRIVER CORRELATION MATRIX (final 5 predictors only)
# =============================================================================
# Full Spearman correlation among the five predictors entering the main lollipop
# figure. No outputs included — this shows collinearity among predictors only.

cat("\n========================================\n")
cat("S2. DRIVER CORRELATION MATRIX\n")
cat("========================================\n\n")

# Final manuscript predictors (structural + climatic; same as lollipop in main text)
final_drivers <- c("N_active_drains", "Mean_Intra_Dist",
                   "Mean_Temperature", "Dry_days_frac", "N_flush_events")

driver_labels <- c(
  "N_active_drains"  = "Number of water drains",
  "Mean_Intra_Dist"  = "Mean inter-drain distance",
  "Mean_Temperature" = "Mean temperature",
  "Dry_days_frac"    = "Fraction of dry days",
  "N_flush_events"   = "Flushing events"
)

corr_data <- predictors %>% select(all_of(final_drivers))
corr_mat  <- cor(corr_data, method = "spearman", use = "complete.obs")

cat(sprintf("Spearman correlation matrix (n = %d zone-year observations)\n",
            nrow(corr_data)))
print(round(corr_mat, 3))

# Significance p-values (for reference; not shown in plot)
n_obs <- nrow(corr_data)
t_stat <- corr_mat * sqrt((n_obs - 2) / (1 - corr_mat^2))
p_mat  <- 2 * pt(-abs(t_stat), df = n_obs - 2)
cat("\n--- p-values ---\n"); print(round(p_mat, 4))

# Long format for ggplot
corr_long <- as.data.frame(as.table(corr_mat)) %>%
  rename(Var1 = Var1, Var2 = Var2, Correlation = Freq) %>%
  mutate(
    Var1 = factor(driver_labels[as.character(Var1)],
                  levels = driver_labels[final_drivers]),
    Var2 = factor(driver_labels[as.character(Var2)],
                  levels = rev(driver_labels[final_drivers]))
  )

p_corr <- ggplot(corr_long, aes(x = Var1, y = Var2, fill = Correlation)) +
  geom_tile(color = "white", linewidth = 0.6) +
  geom_text(aes(label = sprintf("%.2f", Correlation)),
            size = 4, fontface = ifelse(abs(corr_long$Correlation) > 0.4 & corr_long$Correlation != 1 , "bold", "plain")) +
  scale_fill_gradient2(
    low      = "#4393C3",
    mid      = "white",
    high     = "#B2182B",
    midpoint = 0,
    limits   = c(-1, 1),
    name     = expression("Spearman " * rho)
  ) +
  labs(
    x        = NULL, y = NULL
  ) +
  theme_minimal() +
  theme(
    axis.text.x      = element_text(angle = 40, hjust = 1, size = 13, color = "black"),
    axis.text.y      = element_text(size = 13, color = "black"),
    plot.title       = element_text(size = 14, hjust = 0.5, color = "black"),
    plot.subtitle    = element_text(size = 11, hjust = 0.5, color = "black"),
    panel.grid       = element_blank(),
    legend.title     = element_text(size = 12),
    legend.text      = element_text(size = 11)
  ) +
  coord_fixed()

save_and_display(p_corr, "FigS_Driver_Corr_Matrix.pdf", w = 7, h = 6)
cat("Driver correlation matrix saved.\n")


# =============================================================================
# S3. ACTIVE DRAINS YEARLY TIME SERIES PER ZONE
# =============================================================================
# One colored line per zone showing how the count of unreformed (active) drains
# evolves from 2019 to 2024. Reform events appear as stepwise drops.

cat("\n========================================\n")
cat("S3. ACTIVE DRAINS TIME SERIES\n")
cat("========================================\n\n")

drain_ts <- predictors %>%
  mutate(Zone_full = Zone) %>%
  filter(!is.na(Zone_full)) %>%
  mutate(Zone_full = factor(Zone_full, levels = c(
    "Parc de la Ciutadella",
    "Parc del Turó de la Peira",
    "Jardins Mossèn Cinto Verdaguer",
    "Jardins del Teatre Grec",
    "Jardins del Turó del Putxet",
    "Jardins de Vil·la Amèlia - Cecília",
    "Parc de la Guineueta"
  )))

p_drains <- ggplot(drain_ts, aes(x = Year, y = N_active_drains,
                                  color = Zone_full, group = Zone_full)) +
  geom_line(linewidth = 1.1) +
  geom_point(size = 2.8, shape = 16) +
  geom_text(
    data = drain_ts %>% filter(Year == max(Year)),
    aes(label = N_active_drains),
    hjust = -0.6, size = 3.2, show.legend = FALSE
  ) +
  scale_color_manual(values = zone_colors, name = NULL) +
  scale_x_continuous(breaks = 2019:2024,
                     expand = expansion(add = c(0.1, 1.1))) +
  scale_y_continuous(limits = c(0, NA), expand = expansion(mult = c(0, 0.10))) +
  labs(
    x = NULL,
    y = "Number of water drains"
  ) +
  theme_minimal() +
  theme(
    legend.position         = "inside",
    legend.position.inside  = c(0.98, 0.97),
    legend.justification    = c(1, 1),
    legend.text             = element_text(size = 10, color = "black"),
    legend.background       = element_rect(fill = "white", color = "black", linewidth = 0.3),
    axis.text               = element_text(size = 11, color = "black"),
    axis.title              = element_text(size = 12, color = "black"),
    panel.grid.minor        = element_blank()
  )

save_and_display(p_drains, "FigS_Active_Drains_TimeSeries.png", w = 9, h = 5)
cat("Active drains time series saved.\n")


# =============================================================================
# S4. PAIRWISE INTER-DRAIN DISTANCE DISTRIBUTIONS (initial vs final)
# =============================================================================


cat("\n========================================\n")
cat("S4. PAIRWISE DISTANCE DISTRIBUTIONS\n")
cat("========================================\n\n")

dist_matrix <- as.matrix(read.table(
  file.path(script_dir, "Parameters/ZR_Distance_Matrix.txt"), sep = "\t"
))

drains <- read.csv(file.path(script_dir, "Dades_netes/Unique_Items.csv"),
                   stringsAsFactors = FALSE)
drains$reform_year <- as.integer(format(as.Date(drains$data_reforma), "%Y"))
drains$reform_year[is.na(drains$reform_year)] <- 2030  # never reformed -> always active

PERIODS <- c(2019, 2024)
period_labels <- c("2019" = "Initial (2019)", "2024" = "Final (2024)")

# Collect the full pairwise-distance vector per zone/period (not just the mean)
pairwise_dist <- data.frame()
zone_means <- data.frame()

for (z in names(zone_colors)) {
  zone_idx <- which(drains$nom_zr == z)
  for (yr in PERIODS) {
    active_idx <- zone_idx[drains$reform_year[zone_idx] > yr]
    if (length(active_idx) < 2) next
    sub_D <- dist_matrix[active_idx, active_idx]
    d <- sub_D[upper.tri(sub_D)]
    pairwise_dist <- rbind(pairwise_dist, data.frame(Zone = z, Year = yr, Dist = d))
    zone_means <- rbind(zone_means,
                        data.frame(Zone = z, Year = yr, Mean_Dist = mean(d), N_pairs = length(d)))
  }
}

zone_means$Period <- factor(period_labels[as.character(zone_means$Year)],
                            levels = period_labels)
zone_means$Zone_short <- factor(zone_short[zone_means$Zone], levels = zone_order)

cat("--- Mean pairwise distance per zone/period (must match Predictors.csv Mean_Intra_Dist) ---\n")
print(zone_means %>% select(Zone, Year, N_pairs, Mean_Dist) %>% arrange(Zone, Year))

# Bin each zone on shared breaks spanning its 2019 (largest) range, so the
# final histogram falls into the exact same bins as the initial one.
N_BINS <- 20

compute_zone_hist <- function(zone_name) {
  d2019 <- pairwise_dist$Dist[pairwise_dist$Zone == zone_name & pairwise_dist$Year == 2019]
  if (length(d2019) == 0) return(NULL)
  breaks <- seq(0, max(d2019) * 1.02, length.out = N_BINS + 1)
  binwidth <- diff(breaks)[1]

  bind_rows(lapply(PERIODS, function(yr) {
    d <- pairwise_dist$Dist[pairwise_dist$Zone == zone_name & pairwise_dist$Year == yr]
    if (length(d) == 0) return(NULL)
    h <- hist(d, breaks = breaks, plot = FALSE)
    data.frame(Zone = zone_name, Year = yr, mid = h$mids, count = h$counts, bw = binwidth)
  }))
}

hist_df <- bind_rows(lapply(names(zone_colors), compute_zone_hist))
hist_df$Zone_short <- factor(zone_short[hist_df$Zone], levels = zone_order)
hist_df$Period <- factor(period_labels[as.character(hist_df$Year)], levels = period_labels)

# Draw Initial (pale) before Final (opaque) so the smaller final bars nest
# visibly inside the taller initial ones rather than being drawn over.
hist_df <- hist_df %>% arrange(Period, Zone_short)

p_pairdist <- ggplot(hist_df, aes(x = mid, y = count, fill = Zone, alpha = Period, width = bw)) +
  geom_col(position = "identity", color = NA) +
  geom_vline(data = zone_means,
             aes(xintercept = Mean_Dist, linetype = Period, color = Period),
             linewidth = 0.6) +
  scale_fill_manual(values = zone_colors, guide = "none") +
  # All four scales key on Period, merged into one legend that shows both the
  # bar shade and the mean-line style (dashed/solid) per key via override.aes.
  scale_alpha_manual(
    values = c("Initial (2019)" = 0.30, "Final (2024)" = 0.85),
    name   = NULL,
    guide  = guide_legend(override.aes = list(
      fill      = c("grey75", "grey30"),
      color     = c("grey40", "black"),
      linetype  = c("dashed", "solid"),
      linewidth = 0.6
    ))
  ) +
  scale_color_manual(values = c("Initial (2019)" = "grey40", "Final (2024)" = "black"), guide = "none") +
  scale_linetype_manual(values = c("Initial (2019)" = "dashed", "Final (2024)" = "solid"), guide = "none") +
  facet_wrap(~ Zone_short, scales = "free", ncol = 4) +
  labs(x = "Pairwise inter-drain distance (m)", y = "Number of drain pairs") +
  theme_minimal() +
  theme(
    legend.position  = "bottom",
    strip.text       = element_text(size = 12, face = "bold", color = "black"),
    axis.text        = element_text(size = 10, color = "black"),
    axis.title       = element_text(size = 12, color = "black"),
    panel.grid.minor = element_blank()
  )

save_and_display(p_pairdist, "FigS_PairwiseDist_Distribution_Panel.png", w = 14, h = 7)
cat("Pairwise distance distribution panel saved.\n")


# =============================================================================
# S5. COMBINED NETWORK-STRUCTURE FIGURE (S3 center + S4 satellites)
# =============================================================================

cat("\n========================================\n")
cat("S5. COMBINED NETWORK-STRUCTURE FIGURE\n")
cat("========================================\n\n")


pad_to <- function(n) {
  function(x) {
    s <- formatC(x, format = "d")
    out <- paste0(strrep("\u2007", pmax(0, n - nchar(s))), s)
    out[is.na(x)] <- ""
    out
  }
}

make_satellite <- function(zname, xlab = NULL, ylab = NULL, ndig = NULL) {
  hd   <- hist_df    %>% filter(Zone == zname)
  zm   <- zone_means %>% filter(Zone == zname)
  zcol <- zone_colors[[zname]]

  ggplot(hd, aes(x = mid, y = count, alpha = Period, width = bw)) +
    geom_col(position = "identity", fill = zcol, color = NA) +
    geom_vline(data = zm,
               aes(xintercept = Mean_Dist, linetype = Period),
               color = "grey20", linewidth = 0.5, show.legend = FALSE) +
    scale_alpha_manual(
      values = c("Initial (2019)" = 0.30, "Final (2024)" = 0.85),
      guide  = "none"
    ) +
    scale_linetype_manual(
      values = c("Initial (2019)" = "dashed", "Final (2024)" = "solid"),
      guide  = "none"
    ) +
    scale_y_continuous(expand = expansion(mult = c(0, 0.08)),
                       labels = if (is.null(ndig)) waiver() else pad_to(ndig)) +
    labs(x = xlab, y = ylab) +
    theme_minimal(base_size = 12) +
    theme(
      axis.text        = element_text(size = 11, color = "black"),
      axis.title       = element_text(size = 12.5, color = "black"),
      panel.grid.minor = element_blank(),
      legend.position  = "bottom",
      plot.margin      = margin(2, 3, 2, 3)
    )
}

XLAB <- "Inter-drain distance (m)"
YLAB <- "Number of water drain pairs"

# 3-row x 4-col grid. The center (the S3 trajectory) spans a 2x2 block at rows
# 2-3, cols 2-3; the top row holds three satellites plus the legend in the corner;
# two satellites flank each side below. 

sat_ciutadella <- make_satellite("Parc de la Ciutadella",              ylab = YLAB, ndig = 3)              # (1,1)
sat_peira      <- make_satellite("Parc del Turó de la Peira",          xlab = XLAB)                        # (1,2)
sat_cinto      <- make_satellite("Jardins Mossèn Cinto Verdaguer",     xlab = XLAB)                        # (1,3)
sat_guineueta  <- make_satellite("Parc de la Guineueta",               ylab = YLAB, ndig = 3)              # (2,1)
sat_teatre     <- make_satellite("Jardins del Teatre Grec",            ylab = YLAB, ndig = 2)              # (2,4)
sat_amelia     <- make_satellite("Jardins de Vil·la Amèlia - Cecília", xlab = XLAB, ylab = YLAB, ndig = 3) # (3,1)
sat_putxet     <- make_satellite("Jardins del Turó del Putxet",        xlab = XLAB, ylab = YLAB, ndig = 2) # (3,4)


period_key <- data.frame(
  Year = 2019, N_active_drains = 0,
  Period = factor(c("Initial", "Final"), levels = c("Initial", "Final"))
)

p_center <- p_drains +
  geom_col(data = period_key,
           aes(x = Year, y = N_active_drains, alpha = Period),
           fill = "grey40", inherit.aes = FALSE) +
  labs(title = NULL, x = "Year") +
  scale_x_continuous(breaks = 2019:2024, expand = expansion(add = c(0.1, 0.9))) +
  scale_y_continuous(limits = c(0, NA), expand = expansion(mult = c(0, 0.05))) +
  scale_color_manual(values = zone_colors, labels = zone_short, name = NULL,
                     guide = guide_legend(order = 1)) +
  scale_alpha_manual(
    values = c("Initial" = 0.35, "Final" = 0.9), name = NULL,
    guide  = guide_legend(order = 2, nrow = 1, override.aes = list(
      fill      = c("grey75", "grey30"),
      color     = c("grey40", "black"),
      linetype  = c("dashed", "solid"),
      linewidth = 0.6
    ))
  ) +
  theme(
    legend.position   = "right",
    legend.background = element_blank(),
    axis.text         = element_text(size = 13, color = "black"),
    axis.title        = element_text(size = 14, color = "black"),
    axis.title.x      = element_text(margin = margin(t = 8)),
    axis.title.y      = element_text(margin = margin(r = 8)),
    plot.margin       = margin(2, 2, 2, 2)
  )

# 3x4 design. E = center (cols 2-3, rows 2-3); D = legend (top-right corner).
#   A B C D     Ciutadella  Turó Peira  Cinto Verd.  legend
#   F E E G     Guineueta   [    S3    center    ]    Teatre Grec
#   H E E I     Vil·la Am.  [    (2x2 block)     ]    Turó Putxet
s5_design <- "ABCD\nFEEG\nHEEI"

p_S5 <- sat_ciutadella + sat_peira + sat_cinto + guide_area() +
  p_center +
  sat_guineueta + sat_teatre + sat_amelia + sat_putxet +
  plot_layout(design = s5_design, guides = "collect") &
  theme(
    aspect.ratio          = 1,
    legend.box            = "vertical",
    legend.box.just       = "left",
    legend.justification  = "center",
    legend.background     = element_blank(),
    legend.box.background = element_rect(color = "grey30", fill = NA, linewidth = 0.4),
    legend.box.margin     = margin(8, 8, 8, 8),
    legend.margin         = margin(4, 8, 4, 8),
    legend.key.size       = grid::unit(0.7, "cm"),
    legend.text           = element_text(size = 14),
    legend.spacing        = grid::unit(8, "pt")
  )

save_and_display(p_S5, "FigS_Network_Structure_Combined.pdf", w = 12, h = 9)
cat("Combined network-structure figure (S5) saved.\n")