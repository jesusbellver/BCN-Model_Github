# Install packages
#install.packages("xlsx")

#Empty workspace
rm(list = ls())

#Import packages
library(lme4)
library(MuMIn)
library(readxl)
library(dplyr)
library(tidyr)
library(geosphere)
library(oce) 
library(openxlsx)



# LOAD AND PROCESS DATAFRAME
script_dir <- "Dades_crues"
Revisions <- read_excel(file.path(script_dir, "rev_items_8zones_2019_2024_final.xlsx")) #Cargar base de datos de Excel. Ajustar n_max si cambia el numero de filas.

#Process dataframe
Revisions <- Revisions %>% #select relevant variables
  select(data, nom_zr, tipus_entrada, id_item, x, y, tipus_entitat, tipologia, Modificat, prof_sorrer_cm, aigua, activitat, num_larves_aedes, num_larves_culex, tractament) %>% # Select relevant columns
  rename( #rename variables
    Fecha = data,
    visita = tipus_entrada,
    modificat = Modificat,
    prof_sorrer = prof_sorrer_cm,
    activitat_both = activitat,
    activitat_aedes = num_larves_aedes,
    activitat_culex = num_larves_culex) %>% # format variables. Change prof_sorrer to mm
    mutate(Fecha = as.Date(Fecha), modificat = (modificat=="Modificat"), aigua = as.logical(aigua), prof_sorrer = 10 * prof_sorrer,
           activitat_aedes = (activitat_aedes!="0"), activitat_culex = (activitat_culex!="0"), tractament = as.logical(tractament)) %>%
    mutate(activitat_both = (activitat_aedes | activitat_culex), tractament = replace_na(tractament, FALSE))


# Sort dataframe by id_item and data
Revisions <- Revisions %>% arrange(id_item, Fecha)


#Some statistics
num_unique_items <- n_distinct(Revisions$id_item) #Number of unique items studied
unique_zone_date <- nrow(Revisions %>% distinct(nom_zr, Fecha)) #Number of visits to different zones


#Filtrar lo que no son imbornales
Revisions <- Revisions %>%
  filter(tipus_entitat == "Embornal" | tipus_entitat == "Reixa")

# Remove id_item groups where tipologia is always "Directe"
Revisions <- Revisions %>%
  group_by(id_item) %>%
  filter(!(modificat == FALSE & tipologia == "Directe")) %>%
  ungroup()

# Set data_reforma
Revisions <- Revisions %>%
  group_by(id_item) %>%
  mutate(data_reforma = if_else(modificat == TRUE, last(Fecha[tipologia == "Sorrenc"]), NA)) %>%
  ungroup()

#Fusionar Zona Vil.la Amèlia i Vil.la Cecília
Revisions <- Revisions %>%
  mutate(nom_zr = case_when(
    nom_zr == "Jardins de Vil·la Amèlia" ~ "Jardins de Vil·la Amèlia - Cecília",
    nom_zr == "Jardins de Vil·la Cecília" ~ "Jardins de Vil·la Amèlia - Cecília",
    TRUE ~ nom_zr # Keep other values of nom_zr unchanged
  ))

#SOLUCIONAR Sorrencs con prof_sorrer=0 or missing. Cambiar prof_sorrer=0 por la media en la zona de los sorrencs
Revisions <- Revisions %>%
  group_by(nom_zr) %>%
  mutate(mean_prof_sorrer = round(mean(prof_sorrer[prof_sorrer != 0], na.rm = TRUE)),
         prof_sorrer = if_else(prof_sorrer == 0 | is.nan(prof_sorrer), mean_prof_sorrer, prof_sorrer)) %>%
  select(-mean_prof_sorrer) %>%
  ungroup()

#Gestionar Posición Imbornales
# Convert UTM coordinates to longitude and latitude using utm2lonlat function
lonlat <- utm2lonlat(as.numeric(Revisions$x), as.numeric(Revisions$y), zone = 31, hemisphere = "N", km = FALSE)

# Add longitude and latitude to the dataframe
Revisions$x_lon <- lonlat$longitude #round to the 6th decimal place
Revisions$y_lat <- lonlat$latitude
#Remove x,y
Revisions <- subset(Revisions, select = -c(x, y))

#Save Revisions net
write.csv(Revisions, "Dades_netes/Revisions.csv", row.names = FALSE)
#write.xlsx(Revisions_Aedes, "Dades_netes/Revisions_net_Albopictus.xlsx", rowNames = FALSE)
#write.xlsx(Revisions_Culex, "Dades_netes/Revisions_net_Culex.xlsx", rowNames = FALSE)



#CREACION DISTANCE MATRIX i UNIQUE_ITEMS
#Dataframe with unique elements.
Unique_items <- Revisions %>%
  distinct(nom_zr, id_item, x_lon, y_lat, tipus_entitat, modificat, prof_sorrer, data_reforma)

#Save unique items
write.csv(Unique_items, "Dades_netes/Unique_Items.csv", row.names = FALSE)


#COMPUTE DISTANCE MATRIX
# Create a matrix of longitude and latitude
coords <- Unique_items[, c("x_lon", "y_lat")]

#Create distance matrix
distance_matrix <- distm(coords, fun = distHaversine)
rownames(distance_matrix) <- Unique_items$id_item
colnames(distance_matrix) <- Unique_items$id_item

#Save processed data
write.table(distance_matrix, file = "Parameters/ZR_Distance_Matrix.txt", sep = "\t", quote = FALSE, row.names = FALSE, col.names = FALSE)








# 
# 
# #With zones disconnected
# #Disconnect breeding sites from different risk zones
# 
# for (i in Unique_items$id_item) {
#   for (j in Unique_items$id_item) {
#     nom_zr_i <- as.character(Unique_items[Unique_items$id_item == i, "nom_zr"])
#     nom_zr_j <- as.character(Unique_items[Unique_items$id_item == j, "nom_zr"])
#     
#     if (nom_zr_i != nom_zr_j) {
#       distance_matrix[which(rownames(distance_matrix) == i), which(colnames(distance_matrix) == j)] <- 0
#       distance_matrix[which(rownames(distance_matrix) == j), which(colnames(distance_matrix) == i)] <- 0
#     }
#   }
# }
# 
# 
# #Save processed data
# write.table(distance_matrix, file = "Parameters/ZR_Distance_Matrix_Disconnected.txt", sep = "\t", quote = FALSE, row.names = FALSE, col.names = FALSE)
