import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random
import librosa
import sounddevice as sd

def load_features(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Feature file not found: {path}")
    df = pd.read_csv(path)
    return df

def reduce_dimensions(df, n_components=2):
    features = df.drop(columns=["file", "classID"])
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(features)
    print(f"Explained variance: {pca.explained_variance_ratio_}")
    return reduced

def cluster_profiles(reduced_data, n_clusters=4):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(reduced_data)
    return labels

def visualize_clusters(reduced_data, labels):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=reduced_data[:, 0], y=reduced_data[:, 1],
                    hue=labels, palette="Set2", s=60)
    plt.title("Soundscape Clusters (PCA + KMeans)")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend(title="Cluster")
    plt.tight_layout()
    plt.show()

def play_sample_from_clsuter(df, cluster_id, fold ="fold1", sample_count = 3):
    samples = df[df["cluster"] == cluster_id]["file"].tolist()
    print(f"\nPlaying {sample_count} sample(s) from Cluster {cluster_id}:")
    for fname in random.sample(samples, min(len(samples), sample_count)):
        path = f"UrbanSound8K/audio/{fold}/{fname}"
        print(f"Playing: {fname}")
        y, sr = librosa.load(path, sr=None)
        sd.play(y, sr)
        sd.wait()

def describe_clusters_by_class(df, metadata_path):
    meta = pd.read_csv(metadata_path)
    merged = df.merge(meta[['slice_file_name', 'class']], left_on="file", right_on="slice_file_name")
    grouped = merged.groupby("cluster")["class"].value_counts().unstack().fillna(0)
    print("\n🎧 Class Distribution by Cluster:\n")
    print(grouped.astype(int))

if __name__ == "__main__":
    csv_path = "data/features_fold1_100.csv"
    df = load_features(csv_path)

    reduced = reduce_dimensions(df)
    labels = cluster_profiles(reduced, n_clusters=4)

    # Assign numeric cluster labels
    df["cluster"] = labels

    # ✅ OPTIONAL: Assign descriptive labels
    cluster_labels = {
        0: "Engine Idling",
        1: "Traffic Horns",
        2: "Drilling Noise",
        3: "Children Playing"
    }

    df["cluster_label"] = df["cluster"].map(cluster_labels)

    # Save labeled result
    output_path = "data/labeled_clusters_fold1_100.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved labeled features to {output_path}")

    # Play samples from a specific cluster (optional)
    for i in range(4):
        play_sample_from_clsuter(df, i, fold="fold1", sample_count=2)

    describe_clusters_by_class(df, "UrbanSound8K/metadata/UrbanSound8K.csv")

    # Visualize numeric clusters only (optional to update for label text)
    visualize_clusters(reduced, labels)