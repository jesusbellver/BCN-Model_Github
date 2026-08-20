import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import geopandas as gpd
import contextily as ctx
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
from collections import OrderedDict

script_dir = os.path.dirname(os.path.abspath(__file__))
save_dir   = os.path.join(script_dir, "Figures")

# ==============================================================================
# ZONE POLYGONS (lon, lat — WGS84)
#
# Amèlia and Cecília are treated as one zone throughout the pipeline but occupy
# two nearby polygons that do not share an edge (gap ≈ 15 m in the provided
# coordinates). We fuse them with a buffer-union-unbuffer operation: expand
# each polygon by just enough to bridge the gap, take the union, then shrink
# back. The result is a single valid Polygon that covers both gardens.
# ==============================================================================

_amelia_raw  = [(2.123006,41.393496),(2.1241078,41.39216),
                (2.1225937,41.3914256),(2.12139,41.392725)]
_cecilia_raw = [(2.122875,41.393585),(2.12139,41.39288),(2.1212,41.39335),
                (2.1211085,41.39395),(2.122089,41.39446)]
_buf = 0.00015   # slightly above the 0.000137° gap — recovers original shape after shrink
_amelia_cecilia = unary_union(
    [Polygon(_amelia_raw).buffer(_buf), Polygon(_cecilia_raw).buffer(_buf)]
).buffer(-_buf)

# Canonical zone order (must match the rest of the pipeline)
zone_geometries = OrderedDict([
    ("Parc de la Ciutadella",
        Polygon([(2.1827121481728224,41.38816781046831),(2.1864,41.3909),
                 (2.18745,41.39),(2.189472,41.3873),(2.18624,41.3855)])),
    ("Parc del Turó de la Peira",
        Polygon([(2.1636,41.4345),(2.16870,41.43333),(2.16732467,41.431562),
                 (2.16510485,41.4317331),(2.16327,41.433)])),
    ("Jardins Mossèn Cinto Verdaguer",
        Polygon([(2.16313,41.367355),(2.1630902,41.36873),(2.16465,41.36825),
                 (2.1655,41.36745),(2.1654725,41.3665)])),
    ("Jardins del Teatre Grec",
        Polygon([(2.16,41.369776),(2.1599,41.369),(2.15815,41.36915),
                 (2.15854,41.3703)])),
    ("Jardins del Turó del Putxet",
        Polygon([(2.14284,41.4107),(2.144705,41.408852),(2.1437748,41.408352),
                 (2.1427,41.407),(2.14115,41.4095)])),
    ("Jardins de Vil·la Amèlia - Cecília", _amelia_cecilia),
    ("Parc de la Guineueta",
        Polygon([(2.1699,41.44255),(2.1718,41.443291),(2.172738,41.441956),
                 (2.17307,41.44106),(2.17513987,41.4390808),(2.1742223,41.437976)])),
])

zone_colors = OrderedDict([
    ("Parc de la Ciutadella",              "#FDB462"),
    ("Parc del Turó de la Peira",          "#FCCDE5"),
    ("Jardins Mossèn Cinto Verdaguer",     "#80B1D3"),
    ("Jardins del Teatre Grec",            "#BEBADA"),
    ("Jardins del Turó del Putxet",        "#FB8072"),
    ("Jardins de Vil·la Amèlia - Cecília", "#8DD3C7"),
    ("Parc de la Guineueta",               "#B3DE69"),
])

# ==============================================================================
# METEOCAT STATIONS (from Dades_netes/Meteo_BCN_2018_2024.xlsx)
# ZAL Prat (41.319°N) is excluded — too far south, it forces a zoom-out that
# shrinks all Barcelona zones. Its coordinates are reported in the Methods text.
# ==============================================================================

meteo_stations = {
    "Zoo":                (2.18847, 41.38943),
    "El Raval":           (2.16775, 41.38390),
    "Zona Universitària": (2.10540, 41.37919),
    "Obs. Fabra":         (2.12379, 41.41864),
}

# ==============================================================================
# LOAD DRAIN COORDINATES — Ciutadella, active in 2019
# Active = not reformed before 2019-01-01
# ==============================================================================

drains = pd.read_csv(os.path.join(script_dir, "Dades_netes", "Unique_Items.csv"))
drains["data_reforma"] = pd.to_datetime(drains["data_reforma"], errors="coerce")
drains_2019 = drains[
    drains["data_reforma"].isna() | (drains["data_reforma"] > pd.Timestamp("2019-01-01"))
]
cit_drains = drains_2019[drains_2019["nom_zr"] == "Parc de la Ciutadella"]
print(f"Ciutadella drains active in 2019: {len(cit_drains)}")

# ==============================================================================
# BUILD GeoDataFrames AND PROJECT TO WEB MERCATOR (EPSG:3857)
# ==============================================================================

records = [
    {"zone": name, "color": zone_colors[name], "geometry": geom}
    for name, geom in zone_geometries.items()
]
gdf = gpd.GeoDataFrame(records, crs="EPSG:4326").to_crs("EPSG:3857")

gdf_stations = gpd.GeoDataFrame(
    [{"name": n, "geometry": Point(lon, lat)} for n, (lon, lat) in meteo_stations.items()],
    crs="EPSG:4326",
).to_crs("EPSG:3857")

gdf_cit_drains = gpd.GeoDataFrame(
    [{"geometry": Point(r.x_lon, r.y_lat)} for _, r in cit_drains.iterrows()],
    crs="EPSG:4326",
).to_crs("EPSG:3857")

print("Geometries projected. Fetching basemap tiles...")

# ==============================================================================
# FIGURE — two-panel layout
#   Left : full Barcelona map (zones filled + meteo stations)
#   Right : zoom on Ciutadella (outline only + drain dots)
# ==============================================================================

fig, (ax_main, ax_zoom) = plt.subplots(1, 2, figsize=(14.5, 8),
                                       gridspec_kw={"width_ratios": [0.82, 0.94],
                                                    "wspace": 0.02})

# ---- LEFT PANEL --------------------------------------------------------------

for _, row in gdf.iterrows():
    gpd.GeoDataFrame([row], crs="EPSG:3857").plot(
        ax=ax_main,
        color=row["color"],
        edgecolor="black",
        linewidth=1.2,
        alpha=0.50,   # transparent enough to see street context through
        zorder=2,
    )

gdf_stations.plot(
    ax=ax_main, color="red", edgecolor="black",
    markersize=60, marker="o", alpha=0.85, linewidth=0.8, zorder=4,
)
for _, row in gdf_stations.iterrows():
    ax_main.annotate(
        row["name"],
        xy=(row.geometry.x, row.geometry.y),
        xytext=(-1, 6), textcoords="offset points",
        fontsize=10, color="darkred", fontweight="bold", zorder=5,
    )

ctx.add_basemap(ax_main, source=ctx.providers.CartoDB.Positron, zoom=13, zorder=1, attribution="")

# Set extent for left panel: union of zone bounds AND station bounds + tight margin
import numpy as np
zone_bounds = gdf.total_bounds
sta_bounds  = gdf_stations.total_bounds
all_bounds = np.array([
    min(zone_bounds[0], sta_bounds[0]),
    min(zone_bounds[1], sta_bounds[1]),
    max(zone_bounds[2], sta_bounds[2]),
    max(zone_bounds[3], sta_bounds[3]),
])
mx = (all_bounds[2] - all_bounds[0]) * 0.04
my = (all_bounds[3] - all_bounds[1]) * 0.04
ax_main.set_xlim(all_bounds[0] - mx, all_bounds[2] + mx)
ax_main.set_ylim(all_bounds[1] - my, all_bounds[3] + my)

legend_handles = [
    mpatches.Patch(facecolor=c, edgecolor="black", linewidth=0.8, label=z)
    for z, c in zone_colors.items()
]
legend_handles.append(
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="red",
               markeredgecolor="black", markeredgewidth=0.8,
               markersize=8, alpha=0.85, label="MeteoCAT station")
)
ax_main.legend(handles=legend_handles, loc="upper left",
               fontsize=13, framealpha=0.92)
ax_main.set_axis_off()

# ---- RIGHT PANEL: Ciutadella zoom -------------------------------------------

cit_gdf = gdf[gdf["zone"] == "Parc de la Ciutadella"]

# Outline only — no fill, so the aerial context remains fully readable
cit_gdf.boundary.plot(
    ax=ax_zoom,
    edgecolor=zone_colors["Parc de la Ciutadella"],
    linewidth=2.5,
    zorder=3,
)

gdf_cit_drains.plot(
    ax=ax_zoom, color="black", edgecolor="black",
    markersize=25, marker="o", alpha=0.85, linewidth=0.8, zorder=4,
)

ctx.add_basemap(ax_zoom, source=ctx.providers.CartoDB.Positron, zoom=16, zorder=1, attribution="")

# Tight extent around the polygon so park edges are visible
bounds = cit_gdf.total_bounds   # (minx, miny, maxx, maxy) in EPSG:3857
mx = (bounds[2] - bounds[0]) * 0.04
my = (bounds[3] - bounds[1]) * 0.04
ax_zoom.set_xlim(bounds[0] - mx, bounds[2] + mx)
ax_zoom.set_ylim(bounds[1] - my, bounds[3] + my)

# Legend for right panel
ax_zoom.legend(
    handles=[plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="black",
                        markeredgecolor="black", markeredgewidth=0.8,
                        markersize=7, alpha=0.85, label="Sandbox water drain")],
    loc="upper left", fontsize=13, framealpha=0.92,
)

ax_zoom.set_axis_off()

fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0, wspace=0.02)

# Attribution on the figure (bottom-right corner of each panel)
_attr = "Map tiles by CARTO · © OpenStreetMap contributors"
for ax in (ax_main, ax_zoom):
    ax.text(0.99, 0.01, _attr, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, color="dimgray", fontstyle="italic",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1.5))

# ==============================================================================
# SAVE
# ==============================================================================

out_path = os.path.join(save_dir, "Map_BCN_Zones.pdf")
fig.savefig(out_path, dpi=300, facecolor="white")
print(f"Figure saved → {out_path}")
print("\n" + "="*80)
print("CAPTION FOR MANUSCRIPT:")
print("="*80)
print("""
Figure X. Geographic distribution of risk zones and drains in Barcelona.
(a) The seven risk zones studied (coloured polygons) and the four MeteoCAT 
meteorological stations providing temperature and rainfall data (red dots).
(b) Zoom on Parc de la Ciutadella showing the zone boundary (outline) and 
the 50 active drain locations monitored in 2019 (black dots). Map tiles by 
CARTO; map data © OpenStreetMap contributors.
""")
print("="*80)

plt.show()
