# Install packages
#install.packages(packages_to_install)

#Empty workspace
rm(list = ls())

#Import packages

library(readxl)
library(dplyr)
library(openxlsx)
library(tidyverse) 
library(lubridate)
library(zoo)


# LOAD AND PROCESS DATAFRAME
script_dir <- "Dades_crues"
Meteo_Cat <- read.xlsx(file.path(script_dir, "Meteo_Cat_2018_2024.xlsx")) #Read data

#Select only stations from Barcelona
Meteo_BCN <- Meteo_Cat[grepl("Barcelona", Meteo_Cat$station_name), ]

Meteo_BCN <- Meteo_BCN %>%
  select(timestamp, station_name, mean_temperature, min_temperature, max_temperature, precipitation, long_station, lat_station) %>% # Select relevant columns
  rename(
    Fecha = timestamp,
    station = station_name,
    Mean_T = mean_temperature,
    Min_T = min_temperature,
    Max_T = max_temperature,
    Rainfall = precipitation,
    longitude = long_station,
    latitude = lat_station
  )

Meteo_BCN <- Meteo_BCN %>%
  group_by(station) %>% # Group by station
  mutate(
    Rainfall_21 = rollapply(
      data = Rainfall,
      width = 21,
      FUN = sum,
      align = "right",
      fill = NA
    )
  ) %>%
  ungroup() # Remove grouping after computation

# Filter dates before 2019-01-01
Meteo_BCN <- Meteo_BCN %>%
  filter(as.Date(Fecha) >= as.Date("2019-01-01"))

#Write dataframe
write.xlsx(Meteo_BCN, "Dades_netes/Meteo_BCN_2018_2024.xlsx", rowNames = FALSE)
