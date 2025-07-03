import os
import requests

GITHUB_API_URL = "https://api.github.com/repos/ufal/SimulStreaming/contents/simul_whisper/whisper"
RAW_BASE_URL = "https://raw.githubusercontent.com/ufal/SimulStreaming/main/simul_whisper/whisper"
TARGET_DIR = os.path.join("whisperlivekit", "simul_whisper", "whisper")

def download_files_from_github(api_url, local_dir):
    os.makedirs(local_dir, exist_ok=True)
    response = requests.get(api_url)
    response.raise_for_status()
    items = response.json()

    for item in items:
        if item['type'] == 'file':
            download_url = item['download_url']
            file_name = item['name']
            print(f"Downloading {file_name} ...")
            file_response = requests.get(download_url)
            file_response.raise_for_status()
            with open(os.path.join(local_dir, file_name), 'wb') as f:
                f.write(file_response.content)
        elif item['type'] == 'dir':
            # Recursive call for subdirectories
            download_files_from_github(item['url'], os.path.join(local_dir, item['name']))

def main():
    print(f"Downloading files into {TARGET_DIR} ...")
    download_files_from_github(GITHUB_API_URL, TARGET_DIR)
    print("✅ Download completed successfully.")

if __name__ == "__main__":
    main()
