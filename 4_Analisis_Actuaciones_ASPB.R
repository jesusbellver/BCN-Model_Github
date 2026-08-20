#install.packages("leaflet", dependencies = TRUE, INSTALL_opts = '--no-lock')

# Clear workspace
rm(list = ls())

# Load required libraries
library(readxl)
library(dplyr)
library(ggmap)
library(sf)
library(ggplot2)
library(leaflet)
library(lubridate)
library(RColorBrewer)
library(tidyr)

# LOAD AND PROCESS DATAFRAME
# Define the path to your Excel file
script_dir <- "Dades_netes"
Revisions <- read.csv(file.path(script_dir, "Revisions.csv"))  # Load data
Embornals <- read.csv(file.path(script_dir, "Unique_Items.csv"))  # Load data

#Some statistics
# num_unique_items <- n_distinct(Revisions$id_item) #Number of unique items studied
# num_unique_items <- nrow(Revisions) #Number of unique items studied
# unique_zone_date <- nrow(Revisions %>% distinct(nom_zr, Fecha)) #Number of visits to different zones


#FILTER REVISIONS HABITUALS ÚNICAMENT
Revisions <- Revisions %>%
  mutate(Fecha=as.Date(Fecha)) %>%
  filter(visita=="Zona risc")

#PLOT MAP OF ZONES
plot_map <- function(){
  # List of coordinates for each zone
  zones <- list(
    "Jardins de Vil·la Amèlia - Cecília" = list(
      c(2.1241078, 41.39216),
      c(2.1225937, 41.3914256),
      c(2.1214, 41.3927),
      c(2.1212, 41.39335),
      c(2.1211085, 41.39395),
      c(2.122089, 41.39446),
      c(2.122875, 41.393585)

    ),
    "Jardins del Teatre Grec" = list(
      c(2.16, 41.369776),
      c(2.1599, 41.369),
      c(2.15815, 41.36915),
      c(2.15854, 41.3703)
    ),
    "Jardins del Turó del Putxet" = list(
      c(2.14284, 41.4107),
      c(2.144705, 41.408852),
      c(2.1437748, 41.408352),
      c(2.1427, 41.407),
      c(2.14115, 41.4095)
    ),
    "Jardins Mossèn Cinto Verdaguer" = list(
      c(2.16313, 41.367355),
      c(2.1630902, 41.36873),
      c(2.16465, 41.36825),
      c(2.1655, 41.36745),
      c(2.1654725, 41.3665)
    ),
    "Parc de la Ciutadella" = list(
      c(2.1827121481728224, 41.38816781046831),
      c(2.1864, 41.3909),
      c(2.18745, 41.39),
      c(2.189472, 41.3873),
      c(2.18624,41.3855)
    ),
    "Parc de la Guineueta" = list(
      c(2.1699, 41.44255),
      c(2.1718, 41.443291),
      c(2.172738, 41.441956),
      c(2.17307, 41.44106),
      c(2.17513987, 41.4390808),
      c(2.1742223, 41.437976)
    ),
    "Parc del Turó de la Peira" = list(
      c(2.1636, 41.4345),
      c(2.16870, 41.43333),
      c(2.16732467, 41.431562),
      c(2.16510485, 41.4317331),
      c(2.16327, 41.433)
    )
  )
  
  zone_names <- names(zones)
  # Define the colors for each zone
  zone_colors <- c(
    "Parc de la Ciutadella" = "#FDB462",
    "Parc del Turó de la Peira" = "#FCCDE5",
    "Jardins Mossèn Cinto Verdaguer" = "#80B1D3",
    "Jardins del Teatre Grec" = "#BEBADA",
    "Jardins del Turó del Putxet" = "#FB8072",
    "Jardins de Vil·la Amèlia - Cecília" = "#8DD3C7",
    "Parc de la Guineueta" = "#B3DE69"
  )
  
  
  #PLOT MAP
  # Helper function to add polygons
  add_zone_polygon <- function(map, zone_name, coordinates, color) {
    map %>%
      addPolygons(
        lng = sapply(coordinates, "[[", 1),  # Extract longitudes
        lat = sapply(coordinates, "[[", 2),  # Extract latitudes
        color = color,                       # Border color of the polygon
        fillColor = color,                   # Fill color of the polygon
        fillOpacity = 0.9,                   # Transparency of the polygon fill (0 = transparent, 1 = opaque)
        weight = 2                           # Border thickness
      )
  }
  
  # Initialize the leaflet map
  # mapa_barcelona <- leaflet() %>%
  #   addProviderTiles(providers$CartoDB.Positron) %>%
  #   setView(lng = 2.1631, lat = 41.397, zoom = 12.45) %>%
  #   addScaleBar(position = "bottomleft", options = scaleBarOptions(imperial = FALSE))
  
  mapa_barcelona <- leaflet() %>%
    addProviderTiles(providers$CartoDB.Positron) %>%
    setView(lng = 2.123006, lat = 41.393496, zoom = 13) %>%
    addScaleBar(position = "topleft", options = scaleBarOptions(imperial = FALSE))
  
  # Loop through each zone and add its polygon to the map
  for (zone_name in zone_names) {
    mapa_barcelona <- add_zone_polygon(mapa_barcelona, zone_name, zones[[zone_name]], zone_colors[[zone_name]])
  }
  
  # Add a legend
  mapa_barcelona <- mapa_barcelona %>%
    addLegend(
      position = "topleft",
      colors = zone_colors[zone_names],  # Ensure colors match the order of zone_names
      labels = zone_names,
      title = "Zones"
    )
  
  # Display the map
  mapa_barcelona
}


plot_map()



#PLOT NUMBER OF ZONES VISITED PER DAY
zones_per_day <- function(df){
  
  # Extracting the number of uniquely different "nom_zr" visited for any given date
  unique_nom_zr_per_date <- df %>%
    group_by(Fecha) %>%
    summarise(unique_nom_zr = n_distinct(nom_zr))
  
  # Display the result
  print(median(unique_nom_zr_per_date$unique_nom_zr))
  
  # Histogram for Zones Different Visit in One Day
  p <- ggplot(unique_nom_zr_per_date, aes(x = unique_nom_zr)) +
    geom_histogram(binwidth = 1, fill = "blue", color = "black", alpha = 0.8) +
    labs(title = "", x = "Number of risk zones visited per day", y = "Counts") +
    theme(plot.title = element_text(hjust = 0.5, size = 20),
          axis.title = element_text(size = 20),
          axis.text = element_text(size = 18),
          plot.margin = margin(10, 10, 10, 10),
          text = element_text(size = 18))
  
  print(p)
  #ggsave("Figures/Analisis_Actuaciones_ASPB/Zones_per_day.png", width = 10, height = 5, plot=p)

}

#PLOT NUMBER OF VISITS PER WEEK
days_per_week <- function(df){
  
  
  visits_per_week <- df %>%
    mutate(week = strftime(Fecha, "%Y-%U")) %>%
    group_by(week) %>%
    summarise(distinct_dates_per_week = n_distinct(Fecha))
  
  # Histogram for Distinct Dates per Week
  p <- ggplot(visits_per_week, aes(x = distinct_dates_per_week)) +
    geom_histogram(binwidth = 1, fill = "red", color = "black", alpha = 0.8) +
    labs(title = "", x = "Number of days of visit per week", y = "Counts") +
    theme(plot.title = element_text(hjust = 0.5, size = 20),
          axis.title = element_text(size = 20),
          axis.text = element_text(size = 18),
          plot.margin = margin(10, 10, 10, 10),
          text = element_text(size = 18))
  
  print(p)
  #ggsave("Figures/Analisis_Actuaciones_ASPB/Days_per_week.png", width = 10, height = 5, plot=p)
  
}



#PLOT NUBER OF DAYS BETWEEN VISITS TO SAME ZONE
weeks_between_visits <- function(df){
  # Calculating the number of days between any two distinct visits to a given "nom_zr", excluding the year changes
  Revisions_distinct_year <- df %>%
    arrange(nom_zr, Fecha) %>%
    distinct(nom_zr, Fecha) %>%
    group_by(nom_zr) %>%
    mutate(
      previous_year = lag(year(Fecha), order_by = Fecha),
      current_year = year(Fecha),
      days_between_visits = ifelse(current_year != previous_year, NA, as.integer(Fecha - lag(Fecha, order_by = Fecha))/7)
    ) %>%
    select(nom_zr, Fecha, days_between_visits)
  
  # Display the result
  print(mean(Revisions_distinct_year$days_between_visits,na.rm=TRUE))
  print(median(Revisions_distinct_year$days_between_visits,na.rm=TRUE))
  
  
  # Histogram for the number of days between any two distinct visits to a given Risk Zone
  p <- ggplot(Revisions_distinct_year, aes(x = days_between_visits, fill = nom_zr)) +
    geom_histogram(binwidth = 1, color = "black", alpha = 0.8, position = "stack", show.legend = FALSE) +  # Enable legend
    labs(title = "",
         x = "Days between visits to the same risk zone",
         y = "Counts") +
    scale_fill_brewer(palette = "Set3", name = "Zona de Risc") +  # Use a color palette suitable for categorical data
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5, size = 20),   # Center the title and increase size
          axis.title = element_text(size = 20),                 # Axis labels size
          axis.text = element_text(size = 18),                  # Axis text size
          text = element_text(size = 18),                        # General text size
          plot.margin = margin(10, 10, 10, 10))                  # Adjust plot margins
  
  print(p)
  #ggsave("Figures/Analisis_Actuaciones_ASPB/Weeks_between_visits.png", width = 10, height = 5, plot=p)

}


#PLOT FIRST DAY OF VISIT
first_visits <- function(df){

  # Compute first visits and assign a fake year
  First_Visits_summary <- df %>%
    mutate(Year = year(Fecha)) %>%  # Extract year
    group_by(Year, nom_zr) %>%
    filter(Fecha == min(Fecha)) %>%  # Keep only the first visit date per zone for each year
    ungroup() %>%
    distinct(Fecha, nom_zr, .keep_all = TRUE) %>%
    mutate(Fecha = as.Date(paste("2000", format(Fecha, "%m-%d"), sep = "-")), visits=1)  # Assign fake year
  
  p <- ggplot(First_Visits_summary, aes(x = Fecha, y = visits, fill = nom_zr)) +
    geom_bar(stat = "identity", width = 1, color = "black") +  # Fixed width and black border
    scale_fill_brewer(palette = "Set3", name = "Zona de Risc") +  # Use a color palette suitable for categorical data
    scale_x_date(date_labels = "%b", date_breaks = "1 month") +  # Show only months
    labs(x = "Month",
         y = "First visits to zones each year",
         fill = "Zone") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
    
  # Print the plot for the current year
  print(p)
  
  # Save the plot with the year in the filename
  #ggsave(filename = "Figures/Analisis_Actuaciones_ASPB/First_Day_of_Visit.png", plot = p, width = 10, height = 5)

}


last_visits <- function(df){
  
  # Compute first visits and assign a fake year
  Last_Visits_summary <- df %>%
    mutate(Year = year(Fecha)) %>%  # Extract year
    group_by(Year, nom_zr) %>%
    filter(Fecha == max(Fecha)) %>%  # Keep only the first visit date per zone for each year
    ungroup() %>%
    distinct(Fecha, nom_zr, .keep_all = TRUE) %>%
    mutate(Fecha = as.Date(paste("2000", format(Fecha, "%m-%d"), sep = "-")), visits=1)  # Assign fake year
  
  p <- ggplot(Last_Visits_summary, aes(x = Fecha, y = visits, fill = nom_zr)) +
    geom_bar(stat = "identity", width = 1, color = "black") +  # Fixed width and black border
    scale_fill_brewer(palette = "Set3", name = "Zona de Risc") +  # Use a color palette suitable for categorical data
    scale_x_date(date_labels = "%b", date_breaks = "1 month") +  # Show only months
    labs(x = "Month",
         y = "Last visits to zones each year",
         fill = "Zone") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
  # Print the plot for the current year
  print(p)
  
  # Save the plot with the year in the filename
  #ggsave(filename = "Figures/Analisis_Actuaciones_ASPB/Last_Day_of_Visit.png", plot = p, width = 10, height = 5)
  
}



#PLOTS
#plot_map()  
#zones_per_day(Revisions)
#days_per_week(Revisions)
#weeks_between_visits(Revisions)
first_visits(Revisions)
#last_visits(Revisions)

# Compute first visits and assign a fake year
First_Visits_summary <- Revisions %>%
  mutate(Year = year(Fecha)) %>%  # Extract year
  group_by(Year, nom_zr) %>%
  filter(Fecha == min(Fecha)) %>%  # Keep only the first visit date per zone for each year
  ungroup() %>%
  distinct(Fecha, nom_zr, .keep_all = TRUE) %>%
  mutate(Fecha = as.Date(paste("2000", format(Fecha, "%m-%d"), sep = "-")), visits=1)  # Assign fake year

df_temp <- First_Visits_summary %>%
  group_by(nom_zr) %>%
  summarise(fv=mean(Fecha)) %>%
  ungroup()

# Calculating the number of days between any two distinct visits to a given "nom_zr", excluding the year changes
Revisions_distinct_year <- Revisions %>%
  arrange(nom_zr, Fecha) %>%
  distinct(nom_zr, Fecha) %>%
  group_by(nom_zr) %>%
  mutate(
    previous_year = lag(year(Fecha), order_by = Fecha),
    current_year = year(Fecha),
    days_between_visits = ifelse(current_year != previous_year, NA, as.integer(Fecha - lag(Fecha, order_by = Fecha))/7)
  ) %>%
  select(nom_zr, Fecha, days_between_visits)

df_temp_2 <- Revisions_distinct_year %>%
  group_by(nom_zr) %>%
  summarise(fv = mean(days_between_visits, na.rm = TRUE)) %>%
  ungroup()



#PLOT SCHEDULE
# Create a new dataframe summarizing visits by unique date and zone
Revisions_summary <- Revisions %>%
  mutate(data = as.Date(data)) %>%  # Ensure the 'data' column is in Date format
  distinct(data, nom_zr, .keep_all = TRUE) %>%  # Keep only unique date-zone pairs
  group_by(data, nom_zr) %>%
  summarize(visits = 1, .groups = 'drop')  # Count each date-zone as a single visit

# Create bar plot with stacked colors for each zone
for (year in 2019:2023) {
  
  # Filter data for the current year
  Revisions_year <- Revisions_summary %>%
    filter(year(data) == year)
  
  # Create the plot
  plot <- ggplot(Revisions_year, aes(x = data, y = visits, fill = nom_zr)) +
    geom_bar(stat = "identity", width = 2) +
    scale_fill_manual(values = zone_colors) +
    scale_x_date(limits = as.Date(c(paste(year, "01-01", sep = "-"), paste(year, "12-31", sep = "-"))), date_labels = "%b") +
    labs(title = paste("Visit Schedule by Zone -", year),
         x = "Date",
         y = "Visits",
         fill = "Zone") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
  # Print the plot for the current year
  print(plot)
  #ggsave(filename = paste0("Figures/Real_Data_Analysis/Visit_Schedule_", year, ".png"),
         #plot = plot, width = 10, height = 5)
}



#STACKED AREA PLOT OF CASES IN ZONES
# Aggregate data to sum 'activitat' for each 'data' and 'nom_zr'
aggregated_data <- Revisions %>%
  group_by(data, nom_zr) %>%
  summarise(total_activity = sum(activitat, na.rm = TRUE)) %>%
  ungroup()

# Loop over each year and create a plot
for (Year in (2019:2023)) {
  
  # Filter the data for the current year
  yearly_data <- aggregated_data %>% 
    filter(lubridate::year(data) == Year)
  
  # Create the stacked area plot for the current year
  plot <- ggplot(yearly_data, aes(x = data, y = total_activity, fill = nom_zr)) +
    geom_area(alpha = 0.8, color = "black", size = 0.2) +  # Use geom_area for stacked area plot
    scale_fill_manual(values = color_palette) +  # Apply color palette for fill
    labs(title = paste("Breeding Site Activity in", Year, "by Zone"),
         x = "Date",
         y = "Total Activity (Sum of 1s)",
         fill = "Zone") +
    theme_minimal()

  print(plot)
  #ggsave(filename = paste0("Figures/Real_Data_Analysis/Stacked_Dynamics_", Year, ".png"),
         #plot = plot, width = 10, height = 5)
}



# Loop over each year to create a pie chart
for (Year in 2019:2023) {
  
  # Aggregate data to sum 'activitat' for each 'data' and 'nom_zr'
  aggregated_data <- Revisions %>%
    group_by(data, nom_zr) %>%
    summarise(total_activity = sum(activitat, na.rm = TRUE)) %>%
    ungroup()
  
  # Filter the data for the current year
  yearly_data <- aggregated_data %>% 
    filter(lubridate::year(data) == Year)
  
  # Create the pie chart for the current year
  plot <- ggplot(yearly_data, aes(x = "", y = total_activity, fill = nom_zr)) +
    geom_bar(stat = "identity", width = 1) +
    coord_polar("y") +  # Convert bar chart to pie chart
    scale_fill_manual(values = color_palette) +  # Use the color palette for zones
    labs(title = paste("Percentage of Positives by Zone in", Year),
         fill = "Zone") +
    theme_void() +  # Remove unnecessary background
    theme(plot.title = element_text(hjust = 0.5))
  
  # Display and save the plot
  print(plot)
  #ggsave(filename = paste0("Figures/Real_Data_Analysis/PieChart_Cases_", Year, ".png"),
         #plot = plot, width = 8, height = 8)
}


#BOXPLOT DISPERSION OF MEAN RETURN TIMES BY ZONES
Revisions_distinct_year <- drop_na(Revisions_distinct_year,days_between_visits)
Revisions_distinct_year$data <- year(Revisions_distinct_year$data)
  
plot <- ggplot(Revisions_distinct_year, aes(x = nom_zr, y = days_between_visits, fill = nom_zr)) +
  scale_fill_manual(values = zone_colors) +
  geom_boxplot() +
  labs(
    title = "Days Between Visits per Zone",
    x = "Zone",
    y = "Days Between Visits"
  ) +
  theme_minimal(base_size = 14) +
  theme(legend.position = "none") +
  theme(axis.text.x = element_text(angle = 35, hjust = 1))
print(plot)

#ggsave(filename = "Figures/Real_Data_Analysis/BoxPlot_Time_between_visits_Zone.png",
       #plot = plot, width = 12, height = 8)

medians <- Revisions_distinct_year %>%
  filter(data == 2020) %>%
  group_by(nom_zr) %>%
  summarise(median_days_between_visits = round(mean(days_between_visits, na.rm = TRUE)))

cat("Median return time for each zone:")
print(medians)

medians <- First_Visits_summary %>%
  mutate(month_day = make_date(2023, month(data), day(data))) %>%  # Set all dates to the same year, e.g., 2023
  group_by(nom_zr) %>%
  summarise(median = mean(month_day, na.rm = TRUE)) %>%
  mutate(median = format(median, "%m-%d"))  # Format result to only show month and day

cat("Median first visit for each zone:")
print(medians)

plot <- ggplot(Revisions_distinct_year, aes(x = factor(data), y = days_between_visits)) +
  geom_boxplot() +
  scale_fill_manual(values = zone_colors) +
  labs(
    title = "Days Between Visits per Zone",
    x = "Year",
    y = "Days Between Visits"
  ) +
  theme_minimal() +
  theme(legend.position = "none")
print(plot)

# ggsave(filename = "Figures/Real_Data_Analysis/BoxPlot_Time_between_visits_Year.png",
#        plot = plot, width = 8, height = 8)

#Grouping by year
# Calculate the mean days between visits for each year per zone
yearly_mean_by_zone <- Revisions_distinct_year %>%
  group_by(nom_zr, data) %>%
  summarise(mean_days_between_visits = mean(days_between_visits, na.rm = TRUE)) %>%
  ungroup()

yearly_mean_by_zone <- yearly_mean_by_zone %>%
  filter(data != 2023)

plot <- ggplot(yearly_mean_by_zone, aes(x = nom_zr, y = mean_days_between_visits, fill = nom_zr)) +
  geom_boxplot() +
  scale_fill_manual(values = zone_colors) +
  labs(
    title = "Mean Days Between Visits per Zone (Yearly Averages)",
    x = "Zone",
    y = "Mean Days Between Visits (Yearly)"
  ) +
  theme_minimal() +
  theme(legend.position = "none")
print(plot)

#ggsave(filename = "Figures/Real_Data_Analysis/BoxPlot_Time_between_visits_Zone_Mean.png",
#plot = plot, width = 8, height = 8)


#COMPUTE AND WRITE STATS
# Extract year from the "data" column in Revisions
Revisions <- Revisions %>%
  mutate(year = year(data))

# Calculate cases_zone by summing 'activitat' for each zone and year
cases_zone_df <- Revisions %>%
  group_by(nom_zr, year) %>%
  summarize(cases_zone = sum(activitat, na.rm = TRUE), .groups = 'drop')

# Calculate items_zone by counting distinct 'id_item' in each 'nom_zr' in Embornals
items_zone_df <- Embornals %>%
  group_by(nom_zr) %>%
  summarize(items_zone = n_distinct(id_item), .groups = 'drop')

# Calculate visits by grouping by nom_zr, year, and data, then counting unique dates
visits_df <- Revisions %>%
  group_by(nom_zr, year, data) %>%
  summarize(visit_count = 1, .groups = 'drop') %>%  # Each unique (nom_zr, year, data) combination counts as one
  group_by(nom_zr, year) %>%
  summarize(visits = sum(visit_count), .groups = 'drop')

# Combine these dataframes into a final dataframe
final_df <- cases_zone_df %>%
  left_join(items_zone_df, by = "nom_zr") %>%
  left_join(visits_df, by = c("nom_zr", "year"))
  
  for (Year in 2019:2023) {
    
    # Filter data for the current year
    year_data <- final_df %>% filter(year == Year)
    
    # Zones ordered by number of cases
    zones_by_cases <- year_data %>%
      arrange(desc(cases_zone)) %>%
      select(nom_zr, cases_zone)
    
    total_cases <- sum(zones_by_cases$cases_zone, na.rm = TRUE)
    
    # Zones ordered by density of cases per item (cases_zone / items_zone)
    zones_by_cases_density <- year_data %>%
      mutate(cases_density = cases_zone / items_zone) %>%
      arrange(desc(cases_density)) %>%
      select(nom_zr, cases_density)
    
    # Zones ordered by number of visits
    zones_by_visits <- year_data %>%
      arrange(desc(visits)) %>%
      select(nom_zr, visits)
    
    total_visits <- sum(zones_by_visits$visits, na.rm = TRUE)
    
    # Zones ordered by density of cases per visit (cases_zone / visits)
    zones_by_cases_per_visit <- year_data %>%
      mutate(cases_per_visit = cases_zone / visits) %>%
      arrange(desc(cases_per_visit)) %>%
      select(nom_zr, cases_per_visit)
    
    # Constructing the report text
    output_text <- paste0(
      "Real Data, Year: ", Year, "\n\n",
      "Zones ordered by number of cases:\n",
      paste0(apply(zones_by_cases, 1, function(row) paste(row["nom_zr"], ": ", row["cases_zone"])), collapse = "\n"), "\n\n",
      "Total cases: ", total_cases, "\n\n",
      "Zones ordered by density of cases/id_item:\n",
      paste0(apply(zones_by_cases_density, 1, function(row) paste(row["nom_zr"], ": ", row["cases_density"])), collapse = "\n"), "\n\n",
      "Zones ordered by number of visits:\n",
      paste0(apply(zones_by_visits, 1, function(row) paste(row["nom_zr"], ": ", row["visits"])), collapse = "\n"), "\n\n",
      "Total Visits: ", total_visits, "\n\n",
      "Zones ordered by density of cases/visit:\n",
      paste0(apply(zones_by_cases_per_visit, 1, function(row) paste(row["nom_zr"], ": ", row["cases_per_visit"])), collapse = "\n")
    )
  # Write to text file
  #writeLines(output_text, paste0("Figures/Real_Data_Analysis/Real_Stats_", Year, ".txt"))
}
