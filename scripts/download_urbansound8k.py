import os
import tarfile
import urllib.request

def download_file(url, dest_path):
    if not os.path.exists(dest_path):
        print(f"Downloading from {url}...")
        urllib.request.urlretrieve(url, dest_path)
        print("Download complete.")
    else:
        print("File already downloaded.")

def extract_tar_gz(file_path, extract_path):
    print("Extracting dataset...")
    with tarfile.open(file_path, "r:gz") as tar:
        tar.extractall(path=extract_path)
    print("Extraction complete.")

if __name__ == "__main__":
    url = "https://zenodo.org/record/1203745/files/UrbanSound8K.tar.gz"
    archive_path = "UrbanSound8K.tar.gz"
    extract_to = "UrbanSound8K"

    # Download dataset
    download_file(url, archive_path)

    # Extract dataset
    if not os.path.exists(os.path.join(extract_to, "audio")):
        extract_tar_gz(archive_path, ".")  # Extracts to current dir
    else:
        print("Dataset already extracted.")
