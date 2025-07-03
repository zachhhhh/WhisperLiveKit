import os
import requests
import inspect

def get_module_path():
    return os.path.dirname(inspect.getfile(inspect.currentframe()))

GITHUB_API_URL = "https://api.github.com/repos/ufal/SimulStreaming/contents/simul_whisper/whisper"
RAW_BASE_URL = "https://raw.githubusercontent.com/ufal/SimulStreaming/main/simul_whisper/whisper"
TARGET_DIR = os.path.join(get_module_path(), "simul_whisper", "whisper")

def download_files_from_github(api_url, local_dir):
    os.makedirs(local_dir, exist_ok=True)
    response = requests.get(api_url)
    response.raise_for_status()
    items = response.json()
    for item in items:
        if item['type'] == 'file':
            download_url = item['download_url']
            file_name = item['name']
            file_response = requests.get(download_url)
            file_response.raise_for_status()
            with open(os.path.join(local_dir, file_name), 'wb') as f:
                f.write(file_response.content)
        elif item['type'] == 'dir':
            # Recursive call for subdirectories
            download_files_from_github(item['url'], os.path.join(local_dir, item['name']))

def download_simulstreaming_backend():
    print(f"Downloading files into {TARGET_DIR} ...")
    download_files_from_github(GITHUB_API_URL, TARGET_DIR)
    print("✅ Download of SimulStreaming backend files completed successfully.")