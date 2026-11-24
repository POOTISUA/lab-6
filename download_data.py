"""Завантаження датасету варіанта (Д6: OCTMNIST) із Zenodo (MedMNIST v3.0)."""
import os
import shutil
import subprocess
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
TARGET = os.path.join(DATA_DIR, "octmnist.npz")
URL = "https://zenodo.org/records/10519652/files/octmnist.npz?download=1"


def ensure_data():
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(TARGET) and os.path.getsize(TARGET) > 10_000_000:
        print(f"Dataset already present: {TARGET} ({os.path.getsize(TARGET)} bytes)")
        return TARGET
    print(f"Downloading {URL} -> {TARGET} ...")
    curl_bin = shutil.which("curl") or shutil.which("curl.exe")
    if curl_bin:
        res = subprocess.run([curl_bin, "-fL", "-o", TARGET, URL], check=False)
        if res.returncode == 0 and os.path.exists(TARGET) and os.path.getsize(TARGET) > 10_000_000:
            print(f"Downloaded via curl: {TARGET} ({os.path.getsize(TARGET)} bytes)")
            return TARGET
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=180) as r, open(TARGET, "wb") as f:
        while True:
            chunk = r.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    print(f"Downloaded: {TARGET} ({os.path.getsize(TARGET)} bytes)")
    return TARGET


if __name__ == "__main__":
    ensure_data()
