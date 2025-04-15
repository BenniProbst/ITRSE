# Put the data loading code here
# After loading, print out a sample of the raw data as it was loaded

# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import kagglehub
import pandas as pd
import os
import glob

# Download latest version
path = kagglehub.dataset_download("imtkaggleteam/fast-food-restaurants-across-america")

print("Path to dataset files:", path)

# Alle CSV-Dateien im Verzeichnis finden
csv_files = glob.glob(os.path.join(path, "*.csv"))

# Alle Dateien einlesen und in einer Liste speichern
# 1. CSV-Dateien einlesen
df1 = pd.read_csv(path+"\\Datafiniti_Fast_Food_Restaurants.csv")
df2 = pd.read_csv(path+"\\Datafiniti_Fast_Food_Restaurants_Jun19.csv")

# In ein gemeinsames DataFrame zusammenführen
combined_df = pd.concat([df1, df2], ignore_index=True)
combined_unique = combined_df.drop_duplicates()

# Ausgabe der Spaltennamen (zur Kontrolle)
print("Vor dem Entfernen:", len(combined_df))
print("Nach dem Entfernen:", len(combined_unique))

# 4. Optional: Ergebnis speichern
combined_unique.to_csv(path+"\\fast_food_vereint_ohne_duplikate.csv", index=False)

# Beispiel: Nur Restaurants in Kalifornien (province == "CA")
california_restaurants = combined_df[combined_df["province"] == "CA"]

# Ausgabe von ein paar Beispielen
print(california_restaurants.head())

