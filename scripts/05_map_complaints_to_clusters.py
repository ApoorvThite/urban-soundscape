# scripts/04_map_complaints_to_clusters.py

import pandas as pd
from sklearn.neighbors import NearestCentroid
import os

# Paths
LABELED_FEATURES_PATH = "data/labeled_clusters_fold1_100.csv"
COMPLAINTS_PATH = "data/cleaned_311_noise_complaints.csv"
OUTPUT_PATH = "data/311_mapped_clusters.csv"

# Load labeled audio features
df_features = pd.read_csv(LABELED_FEATURES_PATH)
if "cluster" not in df_features.columns:
    raise ValueError("Cluster labels not found in features file.")

# Load complaint data
df_complaints = pd.read_csv(COMPLAINTS_PATH)
if not {"latitude", "longitude"}.issubset(df_complaints.columns):
    raise ValueError("Latitude and longitude are required in complaints data.")

# Drop NaNs in complaints
df_complaints = df_complaints.dropna(subset=["latitude", "longitude"])

# Train NearestCentroid using audio cluster centroids
X = df_features[["mfcc_1", "mfcc_2"]]  # Or however many MFCCs you saved
y = df_features["cluster"]

clf = NearestCentroid()
clf.fit(X, y)

# Simulate MFCC-like coordinates for complaints (just an example)
# In reality, you'd need audio-to-feature mapping for complaint locations.
# Here, we'll fake it with lat/lon just for prototype linking
X_fake = df_complaints[["latitude", "longitude"]].copy()
X_fake.columns = ["mfcc_1", "mfcc_2"]

# Predict closest cluster for each complaint (for now, just illustrative)
df_complaints["predicted_cluster"] = clf.predict(X_fake)

# Optionally map human-readable labels
cluster_labels = {
    0: "Engine Idling",
    1: "Traffic Horns",
    2: "Drilling Noise",
    3: "Children Playing"
}
df_complaints["predicted_label"] = df_complaints["predicted_cluster"].map(cluster_labels)

# Save to disk
os.makedirs("data", exist_ok=True)
df_complaints.to_csv(OUTPUT_PATH, index=False)
print(f"✅ Saved complaints with predicted clusters to {OUTPUT_PATH}")