import pandas as pd

def download_311_data(limit=100000):
    url = f"https://data.cityofnewyork.us/resource/erm2-nwe9.csv?$limit={limit}"
    df = pd.read_csv(url)
    print(f"Downloaded {len(df)} records")
    return df

def filter_noise_complaints(df):
    noise_df = df[df["complaint_type"].str.contains("Noise", na=False)]
    print(f"Filtered to {len(noise_df)} noise complaints")
    return noise_df

if __name__ == "__main__":
    df = download_311_data()
    noise_df = filter_noise_complaints(df)
    noise_df.to_csv("data/nyc_311_noise.csv", index=False)
    print("Saved to data/nyc_311_noise.csv")