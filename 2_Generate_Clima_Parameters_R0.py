import pandas as pd
import numpy as np
import xarray as xr
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os


#SOME FUNCTIONS
def smooth_data(data, window_size):
    smoothed_data = np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    return smoothed_data


# Put this near the top of the script (after imports)
script_dir = os.path.dirname(os.path.abspath(__file__))   # folder where the .py is


#LOAD DATA 
# Open the Excel file for reading
Meteo = pd.read_excel(os.path.join(script_dir, "Dades_netes", "Meteo_BCN_2018_2024.xlsx"))
Meteo['Fecha'] = pd.to_datetime(Meteo['Fecha'])  #Convert timestamp to dates


#CREATE RASTER GRID FROM POINT DATA USING IDW INTERPOLATION

#Create a function to interpolate spatially using Inverse Distance Weighting (IDW)
#The function needs coordinates (lon, lat) and a dataframe with df.longitude, df.latitude, df.values
def IDW_Interp(long, lat, df_data):
    # Get unique dates
    dates = df_data['Fecha'].unique()
    
    # Create a dictionary to store daily rasters
    daily_rasters = {}
    
    # Use the grid passed as arguments
    lon_grid = long
    lat_grid = lat
    
    for date in dates:
        df_day = df_data[df_data['Fecha'] == date].copy()
        df_day = df_day.dropna(subset=['values'])
        
        if len(df_day) == 0:
            continue
        
        # Create grid points
        LON, LAT = np.meshgrid(lon_grid, lat_grid)
        
        # Points to interpolate
        grid_points = np.column_stack([LON.ravel(), LAT.ravel()])
        data_points = df_day[['longitude', 'latitude']].values
        data_values = df_day['values'].values
        
        # Calculate distances
        distances = cdist(grid_points, data_points)
        
        # IDW interpolation
        weights = 1.0 / (distances + 1e-10)
        weights_sum = weights.sum(axis=1, keepdims=True)
        interpolated = (weights * data_values).sum(axis=1) / weights_sum.ravel()
        
        daily_rasters[date] = interpolated.reshape(LON.shape)
    
    return daily_rasters, lon_grid, lat_grid


#Define grid bounds (big enough to contain Barcelona)
lon_grid = np.arange(2.0, 2.21, 0.005)
lat_grid = np.arange(41.31, 41.47, 0.005)  # Extended to 41.47 to cover all zones including Guineueta (41.4427) and Turó de la Peira (41.4340)

#Rainfall
df_rainfall = Meteo[['Fecha', 'longitude', 'latitude', 'Rainfall']].copy()
df_rainfall.rename(columns={'Rainfall': 'values'}, inplace=True)
rainfall_rasters, lon_grid, lat_grid = IDW_Interp(lon_grid, lat_grid, df_rainfall)

#21Days Cumulative Rainfall
df_rainfall_21 = Meteo[['Fecha', 'longitude', 'latitude', 'Rainfall_21']].copy()
df_rainfall_21.rename(columns={'Rainfall_21': 'values'}, inplace=True)
rainfall_21_rasters, _, _ = IDW_Interp(lon_grid, lat_grid, df_rainfall_21)

#Mean Temperature
df_mean_temp = Meteo[['Fecha', 'longitude', 'latitude', 'Mean_T']].copy()
df_mean_temp.rename(columns={'Mean_T': 'values'}, inplace=True)
mean_temp_rasters, _, _ = IDW_Interp(lon_grid, lat_grid, df_mean_temp)


#SAVE AS NETCDF FILES
def save_raster_to_netcdf(rasters_dict, lon_grid, lat_grid, filename):
    dates = sorted(rasters_dict.keys())
    n_time = len(dates)
    n_lat = len(lat_grid)
    n_lon = len(lon_grid)
    
    # Create data array
    data = np.zeros((n_time, n_lat, n_lon))
    for t, date in enumerate(dates):
        data[t, :, :] = rasters_dict[date]
    
    # Create xarray Dataset
    ds = xr.Dataset(
        {
            'data': (['time', 'latitude', 'longitude'], data),
        },
        coords={
            'longitude': lon_grid,
            'latitude': lat_grid,
            'time': dates,
        }
    )
    
    ds.to_netcdf(f"Parameters/{filename}.nc")
    print(f"Saved: Parameters/{filename}.nc")


# Create Parameters directory if it doesn't exist
os.makedirs("Parameters", exist_ok=True)

save_raster_to_netcdf(rainfall_rasters, lon_grid, lat_grid, "Rainfall")
save_raster_to_netcdf(rainfall_21_rasters, lon_grid, lat_grid, "Rainfall_21")
save_raster_to_netcdf(mean_temp_rasters, lon_grid, lat_grid, "Mean_Temperature")


#COMPUTE AND SAVE MOSQUITO PARAMETERS
## Thermal responses Aedes Albopictus from Mordecai 2017:
#Basic Functions. Params: c, T0, Tm or Tmin, Tmax
def Briere_func(cte, tmin, tmax, temp):
    if tmax >= temp >= tmin:
        outp = temp * cte * (temp - tmin) * np.sqrt(tmax - temp)
    else:
        return 0.00001
    return outp

def Quad_func(cte, tmin, tmax, temp): #in Mordecai 2019 -q(temp-tmin)(temp-tmax), here q(temp-tmin)(tmax-temp)
    outp = cte * (temp - tmin) * (tmax - temp)
    if temp <= tmin or temp >= tmax:
        return 0.00001
    return outp


def Lin_func(m, z, temp):
    outp = -m * temp + z
    if outp < 0.00001:
        outp = 0.00001
    return outp



#Parameters Albopictus -> 2019 Mordecai
def bit_rate_alb(temp): return Briere_func(0.00019, 10.4, 38.1, temp)  # Biting rate = 1/gonotrophic cycle in Mordecai 2019
def fecundity_alb(temp): return Briere_func(0.0477, 7.9, 35.6, temp)  # Fecundity (eggs/female/gonotrophic cycle) -> Called TFD in Mordecai 2019
def lf_alb(temp): return Quad_func(1.39, 13.5, 31.4, temp)  #lifespan, 1/death rate
def pEA_alb(temp): return Quad_func(0.00356, 9.1, 36.2, temp)  #egg to adult survival


#Parameters Culex -> 2019 Mordecai
def bit_rate_cul(temp): return Briere_func(0.00017, 9.4, 39.6, temp)  # Biting rate, Mordecai 2019
def fecundity_cul(temp): return Quad_func(0.598, 5.3, 38.9, temp)  # Fecundity (eggs/female/gonotrophic cycle), Mordecai 2019 -> Called EFGC in Mordecai 2019
def lf_cul(temp): return Lin_func(4.86, 169.8, temp)  # Adult life span, Mordecai 2019
def pEL_cul(temp): return Quad_func(0.00211, 3.2, 42.6, temp)  # Probability of egg to larval survival, called Egg Viability in Mordecai 2019
def pLA_cul(temp): return Quad_func(0.0036, 7.8, 38.4, temp)  # Probability of larval to adult survival, Mordecai 2019


#R0 Alb
def R0_alb(temp):
    br = bit_rate_alb(temp)
    f = fecundity_alb(temp)
    ls = lf_alb(temp)
    p_ea = pEA_alb(temp)
    
    R0 = f * br * ls * p_ea
    return R0

#R0 Cul
def R0_cul(temp):
    br = bit_rate_cul(temp)
    f = fecundity_cul(temp)
    ls = lf_cul(temp)
    p_el = pEL_cul(temp)
    p_la = pLA_cul(temp)

    R0 = f * br * ls * p_la * p_el
    return R0


#Compute R0 values for each grid cell and time step
def compute_R0_rasters(mean_temp_rasters, lon_grid, lat_grid):
    dates = sorted(mean_temp_rasters.keys())
    
    R0_alb_rasters = {}
    R0_cul_rasters = {}
    R0_max_rasters = {}
    
    for date in dates:
        temp_data = mean_temp_rasters[date]
        
        # Vectorized R0 computation
        R0_alb_data = np.vectorize(R0_alb)(temp_data)
        R0_cul_data = np.vectorize(R0_cul)(temp_data)
        
        R0_alb_rasters[date] = R0_alb_data
        R0_cul_rasters[date] = R0_cul_data
        R0_max_rasters[date] = np.maximum(R0_alb_data, R0_cul_data)
    
    return R0_alb_rasters, R0_cul_rasters, R0_max_rasters


R0_alb_rasters, R0_cul_rasters, R0_max_rasters = compute_R0_rasters(mean_temp_rasters, lon_grid, lat_grid)

#Save
save_raster_to_netcdf(R0_alb_rasters, lon_grid, lat_grid, "R0_aedes")  # Aedes albopictus
save_raster_to_netcdf(R0_cul_rasters, lon_grid, lat_grid, "R0_culex")  # Culex pipiens
save_raster_to_netcdf(R0_max_rasters, lon_grid, lat_grid, "R0_both")  # Maximum R0


#PLOTS
# Sample a location near Barcelona (2.3, 41.3)
def get_timeseries_at_location(rasters_dict, lon_grid, lat_grid, target_lon, target_lat):
    # Find closest grid point
    lon_idx = np.argmin(np.abs(lon_grid - target_lon))
    lat_idx = np.argmin(np.abs(lat_grid - target_lat))
    
    dates = sorted(rasters_dict.keys())
    values = [rasters_dict[date][lat_idx, lon_idx] for date in dates]
    return dates, values

dates_plot, rainfall_21_values = get_timeseries_at_location(rainfall_21_rasters, lon_grid, lat_grid, 2.3, 41.3)
_, r0_alb_values = get_timeseries_at_location(R0_alb_rasters, lon_grid, lat_grid, 2.3, 41.3)
_, r0_cul_values = get_timeseries_at_location(R0_cul_rasters, lon_grid, lat_grid, 2.3, 41.3)
_, r0_max_values = get_timeseries_at_location(R0_max_rasters, lon_grid, lat_grid, 2.3, 41.3)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].plot(dates_plot, rainfall_21_values)
axes[0, 0].set_title('Rainfall 21-day cumulative')
axes[0, 0].set_xlabel('Date')
axes[0, 0].set_ylabel('Rainfall (mm)')
axes[0, 0].tick_params(axis='x', rotation=45)

axes[0, 1].plot(dates_plot, r0_alb_values, label='Aedes albopictus', color='red')
axes[0, 1].plot(dates_plot, r0_cul_values, label='Culex pipiens', color='blue')
axes[0, 1].set_title('R0 Computations')
axes[0, 1].set_xlabel('Date')
axes[0, 1].set_ylabel('R0')
axes[0, 1].legend()
axes[0, 1].tick_params(axis='x', rotation=45)

axes[1, 0].plot(dates_plot, r0_max_values)
axes[1, 0].set_title('R0 max(Aedes, Culex)')
axes[1, 0].set_xlabel('Date')
axes[1, 0].set_ylabel('R0')
axes[1, 0].tick_params(axis='x', rotation=45)
 
difference = np.maximum(r0_cul_values, r0_alb_values) - np.array(r0_cul_values)
axes[1, 1].plot(dates_plot, difference)
axes[1, 1].set_title('max(R0_Culex, R0_Aedes) - R0_Culex')
axes[1, 1].set_xlabel('Date')
axes[1, 1].set_ylabel('Difference')
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('Parameters/R0_Analysis.png', dpi=100, bbox_inches='tight')
print("Saved plot: Parameters/R0_Analysis.png")
plt.show()
plt.close()

print("\nAll processing completed successfully!")

