# Put the data loading code here
# After loading, print out a sample of the raw data as it was loaded

# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import geopandas
import kagglehub
import numpy as np
import pandas as pd
import os
import glob
import os.path
import openpyxl

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
df_acc = pd.read_csv(path2 + "\\gun-violence-data_01-2013_03-2018.csv")

# Entferne Zeilen mit fehlenden Koordinaten
df_fast = df_fast.dropna(subset=["latitude", "longitude"])
df_acc = df_acc.dropna(subset=["latitude", "longitude"])

# in excel schreiben

chunk_size = 1_000_000
if not os.path.exists(path2 + "\\gun-violence-data_01-2013_03-2018_Part1.xlsx"):
    for i in range(0, len(df_acc), chunk_size):
        chunk = df_acc.iloc[i:i+chunk_size]
        file_name = path2 + f"\\gun-violence-data_01-2013_03-2018_Part{i//chunk_size + 1}.xlsx"
        chunk.to_excel(file_name, index=False)
        print(f"Gespeichert: {file_name}")


print("-------------------Fundamental Exploration of Datasets-------------------")

# Spaltennamen ausgeben
# Spaltennamen ausgeben
print(df_fast.columns.tolist())
print(df_acc.columns.tolist())

# === Hilfsfunktionen ===
def explore_numerical(df, col):
    data = df[col].dropna()
    min_val = data.min()
    max_val = data.max()
    mean_val = data.mean()
    std_val = data.std()

    # Indizes für exakte oder numerisch ähnliche Werte
    min_indices = data[data == min_val].index.tolist()
    max_indices = data[data == max_val].index.tolist()
    mean_indices = data[np.isclose(data, mean_val)].index.tolist()

    return {
        "count": len(data),
        "min": min_val,
        "max": max_val,
        "mean": mean_val,
        "std": std_val,
        "min_indices": min_indices or "None found",
        "max_indices": max_indices or "None found",
        "mean_indices": mean_indices or "None found"
    }


def explore_categorical(df, col, max_categories=20):
    vc = df[col].value_counts(dropna=False)
    top = vc.head(max_categories).to_dict()
    if len(vc) > max_categories:
        top['__other__'] = vc.iloc[max_categories:].sum()
    return top

def explore_datetime(df, col):
    try:
        if 'T' in df[col].dropna().astype(str).iloc[0]:
            data = pd.to_datetime(df[col], format='%Y-%m-%dT%H:%M:%SZ', errors='coerce').dropna()
        else:
            data = pd.to_datetime(df[col], format='%Y-%m-%d %H:%M:%S', errors='coerce').dropna()
    except Exception:
        data = pd.to_datetime(df[col], errors='coerce').dropna()

    if data.empty:
        return {"valid_dates": 0}
    delta = (data.max() - data.min()).days
    result = {
        "start": str(data.min()),
        "end": str(data.max()),
        "days": delta,
        "total_entries": len(data),
        "avg_entries_per_day": round(len(data)/delta, 2) if delta > 0 else np.nan
    }
    if 'Start_Time' in df.columns and 'End_Time' in df.columns:
        start_times = pd.to_datetime(df['Start_Time'], errors='coerce')
        end_times = pd.to_datetime(df['End_Time'], errors='coerce')
        durations = (end_times - start_times).dt.total_seconds() / 3600  # in Stunden
        durations = durations.dropna()
        result["avg_duration_hours"] = durations.mean()
        result["std_duration_hours"] = durations.std()
    return result

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
    'accidents': ['ID']
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

# 2. Accident Dataset
print("Accidents Dataset Analysis:")
report = {'accidents': {}}

for col in df_acc.columns:
    if col in excluded_columns['accidents']:
        continue
    if df_acc[col].dtype in ['float64', 'int64']:
        report['accidents'][col] = explore_numerical(df_acc, col)
    elif 'time' in col.lower() or 'timestamp' in col.lower():
        report['accidents'][col] = explore_datetime(df_acc, col)
    elif df_acc[col].dtype == 'object':
        if col in ['Description']:
            report['accidents'][col] = explore_text(df_acc, col)
        else:
            report['accidents'][col] = explore_categorical(df_acc, col, max_categories=55)

pp.pprint(report)
print("-------------------Analyse of Accidents Dataset-------------------")

# berechnet die nächste Entfernung eines Unfalls zu einem Fastfood-Restaurant und stelle die top 10 der
# nächsten Unfälle in einer Liste dar

# GeoDataFrames erstellen
# In metrisches Koordinatensystem konvertieren
gdf_fast = geopandas.GeoDataFrame(
    df_fast,
    geometry=geopandas.points_from_xy(df_fast["longitude"], df_fast["latitude"]),
    crs="EPSG:4326"
).to_crs(epsg=3857)

gdf_acc = geopandas.GeoDataFrame(
    df_acc,
    geometry=geopandas.points_from_xy(df_acc["longitude"], df_acc["latitude"]),
    crs="EPSG:4326"
).to_crs(epsg=3857)

# === Zielverzeichnis vorbereiten ===
output_dir = path2 + "\\accident_reports_at_restaurants"
if os.path.exists(output_dir):
    for f in os.listdir(output_dir):
        os.remove(os.path.join(output_dir, f))
else:
    os.makedirs(output_dir)

distance_threshold_m = 300

# === Zuordnung und Speicherung ===
for idx, restaurant in gdf_fast.iterrows():
    restaurant_geom = restaurant.geometry
    nearby_accidents = gdf_acc[gdf_acc.distance(restaurant_geom) <= distance_threshold_m].copy()

    state = restaurant['province'].replace(" ", "_") if pd.notna(restaurant['province']) else "Unknown"
    plz = str(restaurant['postalCode']).replace(" ", "_") if pd.notna(restaurant['postalCode']) else "Unknown"
    city = restaurant['city'].replace(" ", "_") if pd.notna(restaurant['city']) else "Unknown"
    street = restaurant['address'].replace(" ", "_") if pd.notna(restaurant['address']) else "Unknown"
    name = restaurant['name'].replace(" ", "_").replace("/", "_")
    date = pd.Timestamp.now().strftime("%Y%m%d")

    filename = f"{date} (Aktualisiert), {name}, Adresse_{state}_{plz}_{city}_{street}.xlsx"
    filepath = os.path.join(output_dir, filename)

    columns_to_save = [
        'ID', 'Source', 'Severity', 'Start_Time', 'End_Time', 'Start_Lat', 'Start_Lng', 'End_Lat', 'End_Lng',
        'Distance(mi)', 'Description', 'Street', 'City', 'County', 'State', 'Zipcode', 'Country', 'Timezone',
        'Airport_Code', 'Weather_Timestamp', 'Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)', 'Pressure(in)',
        'Visibility(mi)', 'Wind_Direction', 'Wind_Speed(mph)', 'Precipitation(in)', 'Weather_Condition',
        'Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction', 'No_Exit', 'Railway', 'Roundabout', 'Station',
        'Stop', 'Traffic_Calming', 'Traffic_Signal', 'Turning_Loop', 'Sunrise_Sunset', 'Civil_Twilight',
        'Nautical_Twilight', 'Astronomical_Twilight', 'mid_lat', 'mid_lng'
    ]
    # Sicherstellen, dass Spalten existieren
    existing_columns = [col for col in columns_to_save if col in nearby_accidents.columns]
    nearby_accidents[existing_columns].to_excel(filepath, index=False)

