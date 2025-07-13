import pandas as pd
import folium
from folium.plugins import MarkerCluster
import plotly.express as px
import os

# Paths
CLUSTER_DATA = "data/labeled_clusters_fold1_100.csv"
COMPLAINTS_DATA = "data/311_mapped_clusters.csv"
MAP_OUTPUT = "outputs/noise_complaints_map.html"

# Load Data
df_clusters = pd.read_csv(CLUSTER_DATA)
df_complaints = pd.read_csv(COMPLAINTS_DATA)

# Clean
df_complaints = df_complaints.dropna(subset=["latitude", "longitude", "predicted_label"])

# 1️⃣ Create Map
city_center = [40.75, -73.98]
m = folium.Map(location=city_center, zoom_start=11, tiles="CartoDB positron")
marker_cluster = MarkerCluster().add_to(m)

# Color map for clusters
colors = {
    "Engine Idling": "blue",
    "Traffic Horns": "orange",
    "Drilling Noise": "green",
    "Children Playing": "red"
}

# Add points to map
for _, row in df_complaints.iterrows():
    label = row["predicted_label"]
    folium.Marker(
        location=[row["latitude"], row["longitude"]],
        popup=f"{row['complaint_type']}<br>{label}",
        icon=folium.Icon(color=colors.get(label, "gray"), icon="volume-up", prefix='fa')
    ).add_to(marker_cluster)

# Save map
os.makedirs("outputs", exist_ok=True)
m.save(MAP_OUTPUT)
print(f"✅ Interactive map saved to: {MAP_OUTPUT}")

# 2️⃣ Plotly Cluster Count Bar Chart
bar_df = df_complaints["predicted_label"].value_counts().reset_index()
bar_df.columns = ["predicted_label", "Count"]
fig = px.bar(bar_df, x="predicted_label", y="Count", color="predicted_label", title="Noise Complaint Distribution by Cluster")
fig.show()
