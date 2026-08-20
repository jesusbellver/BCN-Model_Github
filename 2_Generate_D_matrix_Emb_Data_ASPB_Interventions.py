import os
import pandas as pd
import numpy as np
from datetime import datetime
import pickle


#SAVE D MATRIX AS PARAMETER
D = np.loadtxt(os.path.join(os.path.dirname(__file__), "Parameters/ZR_Distance_Matrix.txt"), delimiter='\t')

with open(os.path.join(os.path.dirname(__file__), "Parameters/D_Matrix"), 'wb') as f:
    pickle.dump(D, f)


#SAVE COORDINATES AND PROFUNDITAT AS PARAMETERS
Unique_BS = pd.read_csv(os.path.join(os.path.dirname(__file__), "Dades_netes/Unique_Items.csv"))
XY = np.column_stack((Unique_BS['x_lon'], Unique_BS['y_lat']))  # Create a matrix with XY info
X_emb = {}
for i, value in enumerate(Unique_BS['x_lon']):
    X_emb[i] = value
Y_emb = {}
for i, value in enumerate(Unique_BS['y_lat']):
    Y_emb[i] = value

#Store Prof_Sorers as Dictionary
Prof_Sorrers = {}
for i, value in enumerate(Unique_BS['prof_sorrer']):
    Prof_Sorrers[i] = float(value)

with open(os.path.join(os.path.dirname(__file__), "Parameters/X_coords"), 'wb') as f:
    pickle.dump(X_emb, f)
with open(os.path.join(os.path.dirname(__file__), "Parameters/Y_coords"), 'wb') as f:
    pickle.dump(Y_emb, f)
with open(os.path.join(os.path.dirname(__file__), "Parameters/Profunditat_Sorrers"), 'wb') as f:
    pickle.dump(Prof_Sorrers, f)
 



# ============================================================================
# MODEL ASPB CONTROL INTERVENTIONS
# ============================================================================

BS_Data = pd.read_csv(os.path.join(os.path.dirname(__file__), "Dades_netes/Revisions.csv"))

# Convert dates to datetime
BS_Data['Fecha'] = pd.to_datetime(BS_Data['Fecha'])
d0 = datetime(2019, 1, 1)

# Map id_item to drain index (0-based, matching Unique_Items.csv order)
# Note: groupby('id_item') sorts by id_item, which matches Unique_Items.csv order
id_to_index = {id_item: i for i, (id_item, _) in enumerate(BS_Data.groupby('id_item'))}
BS_Data['drain_index'] = BS_Data['id_item'].map(id_to_index)

# Build intervention dictionary: {drain_index: [day0, day1, ...]}
# where day is 0-based index (0 = Jan 1, 2019)
Interventions = {}
for _, row in BS_Data.iterrows():
    if row['tractament'] == True:
        drain_idx = row['drain_index']
        day = (row['Fecha'] - d0).days  # 0-based day index
        
        if drain_idx not in Interventions:
            Interventions[drain_idx] = []
        Interventions[drain_idx].append(day)

# Remove duplicates and sort
for drain_idx in Interventions:
    Interventions[drain_idx] = sorted(set(Interventions[drain_idx]))

print(f"Total drains with interventions: {len(Interventions)}")
print(f"Total intervention events: {sum(len(v) for v in Interventions.values())}")

with open(os.path.join(os.path.dirname(__file__), "Parameters/ASPB_Interventions"), 'wb') as f:
    pickle.dump(Interventions, f)


# ============================================================================
# ZONA RISC INTERVENTIONS (subset of ASPB interventions)
# ============================================================================

# Filter only "Zona risc" treatments
Interventions_Zona_Risc = {}
for _, row in BS_Data.iterrows():
    if row['tractament'] == True and row['visita'] == 'Zona risc':
        drain_idx = row['drain_index']
        day = (row['Fecha'] - d0).days
        
        if drain_idx not in Interventions_Zona_Risc:
            Interventions_Zona_Risc[drain_idx] = []
        Interventions_Zona_Risc[drain_idx].append(day)

# Remove duplicates and sort
for drain_idx in Interventions_Zona_Risc:
    Interventions_Zona_Risc[drain_idx] = sorted(set(Interventions_Zona_Risc[drain_idx]))

print(f"\nZona risc drains with interventions: {len(Interventions_Zona_Risc)}")
print(f"Zona risc intervention events: {sum(len(v) for v in Interventions_Zona_Risc.values())}")

with open(os.path.join(os.path.dirname(__file__), "Parameters/ASPB_Interventions_Zona_risc"), 'wb') as f:
    pickle.dump(Interventions_Zona_Risc, f)


# ============================================================================
# ZONA RISC ALL VISITS — zone-level (Option 3)
# ============================================================================


# Map drain index to zone name
drain_zone_map = {}
for i, (_, row) in enumerate(Unique_BS.iterrows()):
    drain_zone_map[i] = row['nom_zr']

# Collect unique (zone, day) pairs from ALL zona-risc visits
zone_visit_days_raw = {}  # zone_name -> set of days
for _, row in BS_Data.iterrows():
    if row['visita'] == 'Zona risc':
        zone_name = row['nom_zr']
        day = (row['Fecha'] - d0).days
        if zone_name not in zone_visit_days_raw:
            zone_visit_days_raw[zone_name] = set()
        zone_visit_days_raw[zone_name].add(day)

# Merge back-to-back visits (gap <= 2 days → keep first date of cluster)
# Chaining: if days 3,5,6 occur, 5 merges with 3 (gap=2), then 6 merges
# with 5 (gap=1), so all three form one cluster applied on day 3.
zone_visit_days = {}
for zone_name in zone_visit_days_raw:
    raw_sorted = sorted(zone_visit_days_raw[zone_name])
    if len(raw_sorted) == 0:
        zone_visit_days[zone_name] = []
        continue
    merged = [raw_sorted[0]]  # first date of first cluster
    last_in_cluster = raw_sorted[0]
    for d in raw_sorted[1:]:
        if d - last_in_cluster > 2:
            merged.append(d)  # new cluster starts
        last_in_cluster = d
    zone_visit_days[zone_name] = merged

# Build drain-level dictionary: each drain gets ALL visit days of its zone
Visits_Zona_Risc = {}
for drain_idx in range(len(Unique_BS)):
    zone_name = drain_zone_map[drain_idx]
    if zone_name in zone_visit_days and len(zone_visit_days[zone_name]) > 0:
        Visits_Zona_Risc[drain_idx] = zone_visit_days[zone_name].copy()

print(f"\nZona risc ALL visits (zone-level, merged <=2d):")
print(f"  Drains with visit days: {len(Visits_Zona_Risc)}")
print(f"  Total drain-visit entries: {sum(len(v) for v in Visits_Zona_Risc.values())}")
for zone_name in sorted(zone_visit_days_raw.keys()):
    n_raw = len(zone_visit_days_raw[zone_name])
    n_merged = len(zone_visit_days[zone_name])
    print(f"  {zone_name}: {n_raw} raw → {n_merged} merged visit days")

with open(os.path.join(os.path.dirname(__file__), "Parameters/ASPB_Visits_Zona_risc"), 'wb') as f:
    pickle.dump(Visits_Zona_Risc, f)

