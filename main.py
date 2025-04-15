# Put the data loading code here
# After loading, print out a sample of the raw data as it was loaded

# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import kagglehub

# Download latest version
path = kagglehub.dataset_download("imtkaggleteam/fast-food-restaurants-across-america")

print("Path to dataset files:", path)