import requests
from tqdm import tqdm
import os
import pandas as pd

# NYC 311 Noise Complaints API (limit: 100000 rows)
url = "https://data.cityofnewyork.us/resource/erm2-nwe9.csv?$limit=100000"

# File path to save
output_path = "data/nyc_311_noise.csv"

def download_with_progress(url, output_path):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(output_path, "wb") as file, tqdm(
        desc="Downloading NYC 311 Noise Complaints",
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                file.write(chunk)
                bar.update(len(chunk))

def main():
    if not os.path.exists(output_path):
        download_with_progress(url, output_path)
    else:
        print("✔️ File already exists locally. Skipping download.")

    # Optional: Load a few rows to confirm
    df = pd.read_csv(output_path)
    print("✅ Sample data:")
    print(df.head())

    noise_df = df[df["complaint_type"].str.contains("Noise", na=False)]
    noise_df = noise_df.dropna(subset=["latitude", "longitude"])
    print("🎯 Filtered noise-related complaints:", noise_df.shape)

    noise_df.to_csv("data/cleaned_311_noise_complaints.csv", index=False)
    print("✅ Saved to data/cleaned_311_noise_complaints.csv")

    # Optional: inspect common descriptors
    print("\n🧭 Top descriptors:")
    print(noise_df["descriptor"].value_counts().head(10))

if __name__ == "__main__":
    main()
