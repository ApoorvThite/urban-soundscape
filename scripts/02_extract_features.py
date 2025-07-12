import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

# Paths
METADATA_PATH = "UrbanSound8K/metadata/UrbanSound8K.csv"
AUDIO_DIR = "UrbanSound8K/audio"

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)

        rms = np.mean(librosa.feature.rms(y=y))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        spec_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_mean = mfccs.mean(axis=1)

        features = {
            "rms": rms,
            "zcr": zcr,
            "spectral_centroid": spec_centroid
        }

        for i, coeff in enumerate(mfccs_mean):
            features[f"mfcc_{i+1}"] = coeff

        return features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def process_subset(n=100, fold="fold1"):
    metadata = pd.read_csv(METADATA_PATH)
    subset = metadata[metadata['fold'] == int(fold.replace("fold", ""))].head(n)

    results = []

    print(f"Extracting features from {n} files in {fold}...")

    for _, row in tqdm(subset.iterrows(), total=len(subset)):
        file_name = row["slice_file_name"]
        class_id = row["classID"]
        file_path = os.path.join(AUDIO_DIR, fold, file_name)

        features = extract_features(file_path)
        if features:
            features["file"] = file_name
            features["classID"] = class_id
            results.append(features)

    df = pd.DataFrame(results)
    out_path = f"data/features_{fold}_{n}.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved features to {out_path}")

if __name__ == "__main__":
    process_subset(n=100, fold="fold1")