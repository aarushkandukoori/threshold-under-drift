"""Download the UCI Gas Sensor Array Drift dataset (about 10 MB) into data/gas/."""
import os
import urllib.request
import zipfile

URL = "https://archive.ics.uci.edu/static/public/224/gas+sensor+array+drift+dataset.zip"
HERE = os.path.join(os.path.dirname(__file__), "..", "data")

if __name__ == "__main__":
    os.makedirs(HERE, exist_ok=True)
    zpath = os.path.join(HERE, "gas.zip")
    if not os.path.exists(os.path.join(HERE, "gas", "Dataset", "batch10.dat")):
        if not os.path.exists(zpath):
            print("downloading", URL)
            urllib.request.urlretrieve(URL, zpath)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(os.path.join(HERE, "gas"))
    print("gas data ready")
