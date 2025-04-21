# Put the data loading code here
# After loading, print out a sample of the raw data as it was loaded

# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import kagglehub
import pandas as pd
import os
import glob

from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import time

# Funktion zur sicheren Reverse-Geocodierung
def get_zip_code(lat, lon):
    try:
        location = geolocator.reverse((lat, lon), language='en', timeout=10)
        if location and "postcode" in location.raw["address"]:
            return location.raw["address"]["postcode"]
    except GeocoderTimedOut:
        print(f"Timeout bei: {lat}, {lon}")
        return None
    return None

# Geolocator initialisieren
geolocator = Nominatim(user_agent="zip_finder")

def fill_primary_categories(row):
    if pd.isna(row['primaryCategories']) or str(row['primaryCategories']).strip() == '':
        if row['_merge'] == 'right_only':
            return row['primaryCategories_y']
        else:  # left_only oder both
            return row['primaryCategories_x']
    else:
        return row['primaryCategories']  # vorhandener Wert bleibt

# Download latest version
path = kagglehub.dataset_download("imtkaggleteam/fast-food-restaurants-across-america")

print("Path to dataset files:", path)

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

# Anzahl gefundener Postleitzahlen
count_filled = 0

# Durchlaufe nur Zeilen mit leerem postalCode
for idx, row in merged_df[merged_df['postalCode'].isna()].iterrows():
    lat, lon = row['latitude'], row['longitude']
    if pd.notna(lat) and pd.notna(lon):
        zip_code = get_zip_code(lat, lon)
        if zip_code:
            merged_df.at[idx, 'postalCode'] = zip_code
            count_filled += 1
            print(f"Ergänzt: {zip_code} für Zeile {idx}")
        time.sleep(1)  # Nominatim: max 1 Anfrage/Sekunde

print(f"Fertig – {count_filled} ZIP-Codes ergänzt.")

# Ausgabe der Spaltennamen (zur Kontrolle)
print("Neue Zeilen aus Datei 1:", len(new_unique))
print("Gesamtergebnis:", len(merged_df))
print(merged_df.head())

# 4. Optional: Ergebnis speichern
merged_df.to_csv(path+"\\fast_food_vereint_ohne_duplikate.csv", index=False)

# Beispiel: Nur Restaurants in Kalifornien (province == "CA")
california_restaurants = merged_df[merged_df["province"] == "CA"]

# Ausgabe von ein paar Beispielen
print(california_restaurants.head())

