# Put the data loading code here
# After loading, print out a sample of the raw data as it was loaded

# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import kagglehub
import openpyxl
import pandas as pd
import os
import glob

# Eat healthy - stay alive

# Compare the probability of gun violence with the concentration of fast food restaurants and back life habits
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

# Download latest version fast food restaurants
path = kagglehub.dataset_download("imtkaggleteam/fast-food-restaurants-across-america")

print("Path to dataset files for fast food restaurants:", path)

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

# 4. Optional: Ergebnis speichern
df_unique.to_csv(path+"\\fast_food_vereint_ohne_duplikate.csv", index=False)

# Beispiel: Nur Restaurants in Kalifornien (province == "CA")
california_restaurants = merged_df[merged_df["province"] == "CA"]

# Ausgabe von ein paar Beispielen
print(california_restaurants.head())

#Konvertiere csv Datei in eine durchsuchbare xlsx Datei
df = pd.read_csv(path+"\\fast_food_vereint_ohne_duplikate.csv")
df.to_excel(path+"\\fast_food_vereint_ohne_duplikate.xlsx", index=False)

# Download latest version fast food restaurants
path2 = kagglehub.dataset_download("sobhanmoosavi/us-accidents")

print("Path to dataset files for gun violence:", path2)

# Alle CSV-Dateien im Verzeichnis finden
csv_files2 = glob.glob(os.path.join(path2, "*.csv"))

# 1. Riesige CSV-Dateien einlesen und Auswertung mit Fastfood restaurants vorbereiten
df2_1 = pd.read_csv(path2 + "\\US_Accidents_March23.csv")

# in excel schreiben
"""
chunk_size = 1_000_000
for i in range(0, len(df2_1), chunk_size):
    chunk = df2_1.iloc[i:i+chunk_size]
    file_name = path2 + f"\\US_Accidents_March23_Part{i//chunk_size + 1}.xlsx"
    chunk.to_excel(file_name, index=False)
    print(f"Gespeichert: {file_name}")
"""
# Spaltennamen ausgeben
print(df2_1.columns.tolist())