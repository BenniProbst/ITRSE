# Put the data loading code here
# After loading, print out a sample of the raw data as it was loaded
# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import threading

import geopandas
import kagglehub
import numpy as np
import pandas as pd
import os
import glob
import os.path
import openpyxl
from datetime import datetime
import matplotlib.pyplot as plt
from shapely.geometry import Point
import matplotlib.colors as mathcolors
import contextily as ctx

from tqdm import tqdm
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

# Eat healthy - stay alive

# Compare the probability of gun violence within the proximity of fast food restaurants against the propability of
# being far away from fast food restaurants.
# https://www.kaggle.com/code/hainescity/fast-food-restaurants-eda

# https://www.kaggle.com/code/shivamb/deep-exploration-of-gun-violence-in-us/input?select=gun-violence-data_01-2013_03-2018.csv

def fill_primary_categories(row):
    if pd.isna(row['primaryCategories']) or str(row['primaryCategories']).strip() == '':
        if row['_merge'] == 'right_only':
            return row['primaryCategories_y']
        else:  # left_only oder both
            return row['primaryCategories_x']
    else:
        return row['primaryCategories']  # vorhandener Wert bleibt

print("-------------------Download Datasets-------------------")

# Download the latest version fast food restaurants
path = kagglehub.dataset_download("imtkaggleteam/fast-food-restaurants-across-america")

print("Path to dataset files for fast food restaurants:", path)

# Download latest version us gun violence dataset
path2 = kagglehub.dataset_download('jameslko/gun-violence-data') #"sobhanmoosavi/us-accidents"

print("Path to dataset files for gun violence:", path2)

print("-------------------Clean Datasets-------------------")

# Alle CSV-Dateien im Verzeichnis finden
csv_files = glob.glob(os.path.join(path, "*.csv"))

# Alle Dateien einlesen und in einer Liste speichern
# 1. Beide CSV-Dateien einlesen
df1 = pd.read_csv(path+"\\Datafiniti_Fast_Food_Restaurants.csv")
df2 = pd.read_csv(path+"\\Datafiniti_Fast_Food_Restaurants_Jun19.csv")

# 2. Falls 'primaryCategories' in df1 fehlt, fügen wir sie hinzu
if 'primaryCategories' not in df1.columns:
    df1['primaryCategories'] = ""

# 3. Spaltenreihenfolge anpassen, damit beide DataFrames gleich aussehen
df1 = df1[df2.columns]  # Reihung so wie df2

# 4. Duplikate finden: alle Zeilen aus df1, die *nicht* in df2 enthalten sind
# Dafür vergleichen wir alle Spalten außer 'primaryCategories'
compare_cols = [col for col in df2.columns if col != 'primaryCategories']

# Duplikate entfernen: Zeilen aus df1, die nicht schon in df2 sind
new_rows = df1.merge(df2, on=compare_cols, how='right', indicator=True)
# Wir holen nur Zeilen aus df1, die nicht auch in df2 sind
new_unique = new_rows[new_rows['_merge'] == 'right_only']
# 5. Neue und alte Daten zusammenfügen
merged_df = pd.concat([df2, new_unique], ignore_index=True)

# Neue Spalte erstellen: Wenn _merge == 'right_only', nimm primaryCategories_y, sonst primaryCategories_x
# Spalte aktualisieren
merged_df['primaryCategories'] = merged_df.apply(fill_primary_categories, axis=1)
merged_df.drop(columns=['primaryCategories_x', 'primaryCategories_y', '_merge'], inplace=True)
df_unique = merged_df.drop_duplicates(subset=['id', 'address', 'categories', 'city', 'country', 'latitude', 'longitude', 'name', 'postalCode', 'country', 'province', 'websites', 'primaryCategories', 'sourceURLs'], keep='first')

# Ausgabe der Spaltennamen (zur Kontrolle)
print("Neue Zeilen aus Datei 1:", len(df_unique))
print("Gesamtergebnis:", len(df_unique))
print(df_unique.head())

# Ersetze die Staaten-Kürzel
us_states = {
    'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas',
    'CA': 'California', 'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware',
    'FL': 'Florida', 'GA': 'Georgia', 'HI': 'Hawaii', 'ID': 'Idaho',
    'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa', 'KS': 'Kansas',
    'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland',
    'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi',
    'MO': 'Missouri', 'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada',
    'NH': 'New Hampshire', 'NJ': 'New Jersey', 'NM': 'New Mexico', 'NY': 'New York',
    'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio', 'OK': 'Oklahoma',
    'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina',
    'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah',
    'VT': 'Vermont', 'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia',
    'WI': 'Wisconsin', 'WY': 'Wyoming', 'DC': 'District of Columbia'
}

df_unique = df_unique.copy()
df_unique['province'] = df_unique['province'].map(us_states).fillna(df_unique['province'])

# Zeitformate anpassen
df_unique['dateAdded'] = pd.to_datetime(df_unique['dateAdded'], format='%Y-%m-%dT%H:%M:%SZ', errors='coerce')
df_unique['dateUpdated'] = pd.to_datetime(df_unique['dateUpdated'], format='%Y-%m-%dT%H:%M:%SZ', errors='coerce')

# Dann ins gewünschte Format konvertieren
df_unique['dateAdded'] = df_unique['dateAdded'].dt.strftime('%Y-%m-%d %H:%M:%S')
df_unique['dateUpdated'] = df_unique['dateUpdated'].dt.strftime('%Y-%m-%d %H:%M:%S')

# Name normalization
def normalize_name(name):
    if pd.isna(name):
        return name
    name = name.strip()
    name = name.lower()
    return name.capitalize()  # Nur erster Buchstabe groß

df_unique['name'] = df_unique['name'].astype(str).map(normalize_name)

# 4. Optional: Ergebnis speichern
df_unique.to_csv(path+"\\fast_food_vereint_ohne_duplikate.csv", index=False)

# Beispiel: Nur Restaurants in Kalifornien (province == "CA")
california_restaurants = merged_df[merged_df["province"] == "CA"]

# Ausgabe von ein paar Beispielen
print(california_restaurants.head())

#Konvertiere csv Datei in eine durchsuchbare xlsx Datei
df_fast = pd.read_csv(path + "\\fast_food_vereint_ohne_duplikate.csv")
df_fast.to_excel(path + "\\fast_food_vereint_ohne_duplikate.xlsx", index=False)


print("Path to dataset files for gun violence:", path2)

# Alle CSV-Dateien im Verzeichnis finden
csv_files2 = glob.glob(os.path.join(path2, "*.csv"))

# 1. Riesige CSV-Dateien einlesen und Auswertung mit Fastfood restaurants vorbereiten
df_gun = pd.read_csv(path2 + "\\gun-violence-data_01-2013_03-2018.csv")

# Entferne Zeilen mit fehlenden Koordinaten
df_fast = df_fast.dropna(subset=["latitude", "longitude"])
df_gun = df_gun.dropna(subset=["latitude", "longitude"])

# in excel schreiben

chunk_size = 1_000_000
if not os.path.exists(path2 + "\\gun-violence-data_01-2013_03-2018_Part1.xlsx"):
    for i in range(0, len(df_gun), chunk_size):
        chunk = df_gun.iloc[i:i + chunk_size]
        file_name = path2 + f"\\gun-violence-data_01-2013_03-2018_Part{i//chunk_size + 1}.xlsx"
        chunk.to_excel(file_name, index=False)
        print(f"Gespeichert: {file_name}")


print("-------------------Fundamental Exploration of Datasets-------------------")

# Spaltennamen ausgeben
# Spaltennamen ausgeben
print(df_fast.columns.tolist())
print(df_gun.columns.tolist())

# === Statistikfunktionen ===
def explore_numerical(df, col):
    data = df[col].dropna()
    return {
        "count": len(data),
        "min": data.min(),
        "max": data.max(),
        "mean": data.mean(),
        "std": data.std(),
        "min_indices": data[data == data.min()].index.tolist() or "None found",
        "max_indices": data[data == data.max()].index.tolist() or "None found",
        "mean_indices": data[np.isclose(data, data.mean())].index.tolist() or "None found"
    }

def explore_categorical(df, col, max_categories=20):
    vc = df[col].value_counts(dropna=False)
    top = vc.head(max_categories).to_dict()
    if len(vc) > max_categories:
        top["__other__"] = vc.iloc[max_categories:].sum()
    return top

def explore_datetime(df, col):
    try:
        data = pd.to_datetime(df[col], errors="coerce").dropna()
    except Exception:
        data = pd.to_datetime(df[col], errors="coerce").dropna()
    if data.empty:
        return {"valid_dates": 0}
    delta = (data.max() - data.min()).days
    return {
        "start": str(data.min()),
        "end": str(data.max()),
        "days": delta,
        "total_entries": len(data),
        "avg_entries_per_day": round(len(data)/delta, 2) if delta > 0 else np.nan
    }

def explore_text(df, col):
    data = df[col].dropna().astype(str)
    lengths = data.str.len()
    avg_len = lengths.mean()
    min_len = lengths.min()
    max_len = lengths.max()
    return {
        "count": len(data),
        "avg_length": avg_len,
        "max_length": max_len,
        "min_length": min_len,
        "max_indices": lengths[lengths == max_len].index.tolist(),
        "min_indices": lengths[lengths == min_len].index.tolist(),
        "mean_indices": lengths[np.isclose(lengths, avg_len)].index.tolist(),
        "max_values": data[lengths == max_len].tolist(),
        "min_values": data[lengths == min_len].tolist(),
        "mean_values": data[np.isclose(lengths, avg_len)].tolist()
    }

# === Spalten, die wir ausschließen wollen ===
excluded_columns = {
    'fastfood': ['id', 'keys'],
    'accidents': ['incident_id']
}

# === Analyse ===

# === Bericht anzeigen (kompakt) ===
import pprint
pp = pprint.PrettyPrinter(depth=3, sort_dicts=False, compact=True)

# 1. Fast Food Dataset
print("Fast Food Dataset Analysis:")
report = {'fastfood': {}}

for col in df_fast.columns:
    if col in excluded_columns['fastfood']:
        continue
    if df_fast[col].dtype in ['float64', 'int64']:
        report['fastfood'][col] = explore_numerical(df_fast, col)
    elif 'date' in col.lower():
        report['fastfood'][col] = explore_datetime(df_fast, col)
    elif df_fast[col].dtype == 'object':
        if col in ['sourceURLs', 'websites']:
            report['fastfood'][col] = explore_text(df_fast, col)
        else:
            report['fastfood'][col] = explore_categorical(df_fast, col, max_categories=55)

pp.pprint(report)
print("-------------------Analyse of Fastfood Dataset-------------------")

# === Analyse der Waffengewalt ===
print("Analyse Gun Violence Dataset:")
report = {"gun_violence": {}}

for col in df_gun.columns:
    if col in ["incident_id"]:
        continue
    if df_gun[col].dtype in ["float64", "int64"]:
        report["gun_violence"][col] = explore_numerical(df_gun, col)
    elif "date" in col.lower():
        report["gun_violence"][col] = explore_datetime(df_gun, col)
    elif df_gun[col].dtype == "object":
        report["gun_violence"][col] = explore_categorical(df_gun, col, max_categories=55)

pp = pprint.PrettyPrinter(depth=3, sort_dicts=False, compact=True)
pp.pprint(report)

print("-------------------Analyse of Gun Violence Dataset-------------------")

# berechnet die nächste Entfernung eines Unfalls zu einem Fastfood-Restaurant und stelle die top 10 der
# nächsten Unfälle in einer Liste dar

# GeoDataFrames erstellen
# In metrisches Koordinatensystem konvertieren
gdf_fast = geopandas.GeoDataFrame(
    df_fast,
    geometry=geopandas.points_from_xy(df_fast["longitude"], df_fast["latitude"]),
    crs="EPSG:4326"
).to_crs(epsg=3857)

gdf_gun = geopandas.GeoDataFrame(
    df_gun,
    geometry=geopandas.points_from_xy(df_gun["longitude"], df_gun["latitude"]),
    crs="EPSG:4326"
).to_crs(epsg=3857)

# === Zielverzeichnis vorbereiten ===
output_dir = path2 + "\\accident_reports_at_restaurants"
if os.path.exists(output_dir):
    for f in os.listdir(output_dir):
        os.remove(os.path.join(output_dir, f))
else:
    os.makedirs(output_dir)

# === Pro Restaurant nahe Vorfälle finden und Gesamtergebnis auswerten ===

# Funktion zur Verarbeitung eines einzelnen Restaurants
# Zähler und Sperre für Threads
completed_count = [0]  # mutable counter
count_lock = threading.Lock()
def process_restaurant(restaurant, count_lock, completed_count_list, total_count):
    distance_threshold_m = 300

    try:
        name = str(restaurant["name"]).replace(" ", "_").replace("/", "_")
        state = str(restaurant.get("province", "Unknown")).replace(" ", "_")
        plz = str(restaurant.get("postalCode", "Unknown")).replace(" ", "_")
        city = str(restaurant.get("city", "Unknown")).replace(" ", "_")
        street = str(restaurant.get("address", "Unknown")).replace(" ", "_")

        filename_search = f"{name}, at {state}_{plz}_{city}_{street}"
        print(f"Searching for gun incident near {filename_search}...")

        x, y = restaurant.geometry.x, restaurant.geometry.y
        candidates = gdf_gun[
            (gdf_gun.geometry.x >= x - distance_threshold_m) & (gdf_gun.geometry.x <= x + distance_threshold_m) &
            (gdf_gun.geometry.y >= y - distance_threshold_m) & (gdf_gun.geometry.y <= y + distance_threshold_m)
        ]
        nearby = candidates[candidates.distance(restaurant.geometry) <= distance_threshold_m].copy()

        # Füge die Distanzspalte hinzu
        nearby["incident distance"] = nearby.geometry.distance(restaurant.geometry)

        # Füge Spalte für gestohlene Waffen hinzu (NaN ersetzen)
        nearby["gun_stolen"] = nearby["gun_stolen"].fillna("Unknown")

        if len(nearby) == 0:
            return None

        if "geometry" in nearby.columns:
            nearby.drop(columns=["geometry"], inplace=True)

        first_date = restaurant.get("dateAdded", None)
        if pd.notna(first_date):
            try:
                parsed_date = pd.to_datetime(first_date)
                date_str = parsed_date.strftime("%Y%m%d")
            except Exception:
                date_str = datetime.now().strftime("%Y%m%d")
        else:
            date_str = datetime.now().strftime("%Y%m%d")

        filename = f"{date_str}, {name}, at {state}_{plz}_{city}_{street}.xlsx"
        filepath = os.path.join(output_dir, filename)

        nearby.to_excel(filepath, index=False)

        with count_lock:
            completed_count_list[0] += 1
            print(f"Created gun incident report for {filename} ({completed_count_list[0]} of {total_count})")

        return {
            "name": restaurant["name"],
            "address": restaurant["address"],
            "state": state,
            "city": city,
            "postalCode": plz,
            "num_incidents": len(nearby),
            "total_killed": nearby["n_killed"].sum(),
            "total_injured": nearby["n_injured"].sum(),
            "gun_stolen_counts": nearby["gun_stolen"].value_counts().to_dict(),
            "restaurant_latitude": restaurant["latitude"],
            "restaurant_longitude": restaurant["longitude"]
        }
    except Exception as e:
        with count_lock:
            completed_count_list[0] += 1
            print(f"Error processing restaurant ({completed_count_list[0]} of {total_count}): {e}")
        return None

# Nutze alle verfügbaren Threads
summary_data = []
num_threads = multiprocessing.cpu_count()
with ThreadPoolExecutor(max_workers=num_threads) as executor:
    futures = [
        executor.submit(process_restaurant, row, count_lock, completed_count, len(gdf_fast))
        for _, row in gdf_fast.iterrows()
    ]
    for future in as_completed(futures):
        result = future.result()
        if result:
            summary_data.append(result)

# Waffengewalt auswerten

# DataFrame für Übersichtstabelle
summary_path = path2 + "\\gun_violence_at_restaurant_results.xlsx"
summary_df = pd.DataFrame(summary_data)

# Neue Spalte: Anzahl gestohlener Waffen (außer "Unknown")
def count_stolen_weapons(gun_dict):
    if not isinstance(gun_dict, dict):
        return 0
    return sum(v for k, v in gun_dict.items() if k.lower() not in ["unknown", ""])

summary_df["num_stolen_weapons"] = summary_df["gun_stolen_counts"].apply(count_stolen_weapons)

# Gesamtergebnis zwischenspeichern
summary_df.to_excel(summary_path, index=False)

# Sortiere nach den gewünschten Kriterien
top10 = summary_df.sort_values(
    by=["num_incidents", "total_killed", "total_injured", "num_stolen_weapons"],
    ascending=[False, False, False, False]
).head(10)

# Ausgabe mit pretty print
pp = pprint.PrettyPrinter(depth=3, sort_dicts=False)
pp.pprint(top10[["name", "address", "num_incidents", "total_killed", "total_injured"]].to_dict(orient="records"))

# Plotting Charts and Heat maps

# Barplot 1: Anzahl der Vorfälle
plt.figure(figsize=(12, 10))
plt.bar(top10["name"] + " @ " + top10["city"] + " - " + top10["state"] + ", " + top10["address"], top10["num_incidents"])
plt.title("Top 10 gefährlichste Restaurants in den USA nach Anzahl der Vorfälle")
plt.ylabel("Anzahl der Vorfälle")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Sortiere nach den gewünschten Kriterien
top10 = summary_df.sort_values(
    by=["total_killed", "total_injured", "num_stolen_weapons", "num_incidents"],
    ascending=[False, False, False, False]
).head(10)

# Barplot 2: Anzahl der Toten
plt.figure(figsize=(12, 10))
plt.bar(top10["name"] + " @ " + top10["city"] + " - " + top10["state"] + ", " + top10["address"], top10["total_killed"])
plt.title("Top 10 gefährlichste Restaurants nach Anzahl der Toten")
plt.ylabel("Anzahl der Toten")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Sortiere nach den gewünschten Kriterien
top10 = summary_df.sort_values(
    by=["total_injured", "num_stolen_weapons", "num_incidents", "total_killed"],
    ascending=[False, False, False, False]
).head(10)

# Barplot 3: Anzahl der Verletzten
plt.figure(figsize=(12, 10))
plt.bar(top10["name"] + " @ " + top10["city"] + " - " + top10["state"] + ", " + top10["address"], top10["total_injured"])
plt.title("Top 10 gefährlichste Restaurants nach Anzahl der Verletzten")
plt.ylabel("Anzahl der Verletzten")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Piechart: Waffenstatus

# Entferne Zeilen mit fehlenden Werten in den interessierenden Spalten
df_gun_clean = df_gun.dropna(subset=["gun_type", "gun_stolen"])

# Hilfsfunktion
def extract_entries(entry):
    return [item.split("::")[1].strip() if "::" in item else item.strip()
            for item in str(entry).split("||")]

# 🔍 Mapping von konkreten Waffentypen zu Kategorien
def classify_gun_type(gun):
    gun = gun.lower()
    if "handgun" in gun or "pistol" in gun or "revolver" in gun:
        return "Handgun"
    elif "rifle" in gun or "ak" in gun or "ar-" in gun or "7.62" in gun or "carbine" in gun:
        return "Rifle"
    elif "shotgun" in gun:
        return "Shotgun"
    elif "bb gun" in gun or "air" in gun:
        return "Airgun"
    elif "unknown" in gun:
        return "Unknown"
    else:
        return "Other"

# === Alle Kombinationen extrahieren ===
type_list, stolen_list, classified_list = [], [], []

for _, row in df_gun_clean.iterrows():
    types = extract_entries(row["gun_type"])
    stolen = extract_entries(row["gun_stolen"])
    for t, s in zip(types, stolen):
        type_list.append(t)
        stolen_list.append(s)
        classified_list.append(classify_gun_type(t))

# In DataFrame umwandeln
df_comb = pd.DataFrame({
    "gun_type": type_list,
    "gun_stolen": stolen_list,
    "gun_class": classified_list
})

# === Klassifikationstabelle ===
classification_table = pd.crosstab(df_comb["gun_class"], df_comb["gun_stolen"])
classification_table.to_excel(path2 + "\\gun_classification.xlsx")
print("✔ gun_classification.xlsx wurde gespeichert.")

# === Kuchendiagramm 1: Waffenstatus ===
plt.figure(figsize=(10, 10))
df_comb["gun_stolen"].value_counts().plot.pie(autopct='%1.1f%%', startangle=140)
plt.title("Verteilung Waffenstatus (gun_stolen)")
plt.ylabel("")
plt.tight_layout()
plt.show()

# === Kuchendiagramm 2: Klassifizierte Waffentypen ===
plt.figure(figsize=(10, 10))
df_comb["gun_class"].value_counts().plot.pie(autopct='%1.1f%%', startangle=140)
plt.title("Verteilung klassifizierter Waffentypen")
plt.ylabel("")
plt.tight_layout()
plt.show()

# Piechart für gun_stolen (nur diese Spalte)
stolen_all = []
for entry in df_gun_clean["gun_stolen"]:
    stolen_all.extend(extract_entries(entry))

df_stolen = pd.Series(stolen_all).value_counts()
plt.figure(figsize=(18, 18))
plt.pie(df_stolen, labels=df_stolen.index, autopct='%1.1f%%', startangle=140)
plt.title("Verteilung des Waffenstatus (gun_stolen)")
plt.tight_layout()
plt.show()

# Piechart für gun_type (nur diese Spalte)
type_all = []
for entry in df_gun_clean["gun_type"]:
    type_all.extend(extract_entries(entry))

df_types = pd.Series(type_all).value_counts()
plt.figure(figsize=(18, 18))
plt.pie(df_types, labels=df_types.index, autopct='%1.1f%%', startangle=140)
plt.title("Verteilung der Waffentypen (gun_type)")
plt.tight_layout()
plt.show()

# Heatmap USA mit Restaurantvorfällen (Farben grün-gelb-rot je nach Anzahl)
# Erzeuge GeoDataFrame mit korrekten Koordinaten
geometry = [Point(xy) for xy in zip(summary_df["restaurant_longitude"], summary_df["restaurant_latitude"])]
gdf = geopandas.GeoDataFrame(summary_df, geometry=geometry, crs="EPSG:4326").to_crs(epsg=3857)

# Farbskala und Normalisierung
incident_counts = gdf["num_incidents"]
max_incidents = incident_counts.max()
cmap = mathcolors.LinearSegmentedColormap.from_list("custom_green_red", ["green", "yellow", "red"])
norm = mathcolors.Normalize(vmin=0, vmax=max_incidents)
colors = [cmap(norm(x)) for x in incident_counts]

# Plot mit Basiskarte (z. B. OpenStreetMap)
fig, ax = plt.subplots(figsize=(20, 15))
gdf.plot(ax=ax, color=colors, markersize=20, alpha=0.7)
ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

ax.set_title("US Map of Gun Violence at Restaurants (Color = #Incidents)")
ax.axis('off')

# Colorbar hinzufügen
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)
cbar.set_label("Number of Incidents")

plt.tight_layout()
plt.show()

# Anzeigen
plt.show()