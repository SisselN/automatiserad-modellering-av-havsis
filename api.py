# Kopplar upp mot YR:s API.

import requests
import os
import database_utils
from datetime import datetime
from email.utils import parsedate_to_datetime


# Identifierar mig
headers = {
    "User-Agent": "isutbredning/1.0 sisselnevestveit@hotmail.com"
}

# Anger vilken geografisk del jag vill ha karta över.
AREA = "norwegian_arctic_areas"
URL = f"https://api.met.no/weatherapi/icemap/1.0/?area={AREA}"

SAVE_DIR = "iskartor"

os.makedirs(SAVE_DIR, exist_ok=True)

# Hämtar datan.

def get_next_index_from_dir():
    """
    Hämtar index för namngivning.
    Returnerar nästa index som en int.
    """
    files = [f for f in os.listdir(SAVE_DIR) if f.endswith(".jpg")]
    return len(files) + 1

def get_map():
    """
    Hämtar den senaste kartan för valt område.
    Retrurnerar hur många sekunder vi ska vänta till nästa request.
    """
    r = requests.get(URL, headers=headers)
    r.raise_for_status()

    # Metadata:
    # När bilden togs
    last_modified = parsedate_to_datetime(r.headers["last-modified"])
    # När nästa bild ska tas
    expires = parsedate_to_datetime(r.headers["expires"])
    date = parsedate_to_datetime(r.headers["date"])

    # Sparar bilden med namnet på området plus index.
    index = get_next_index_from_dir()
    filename = f"{AREA}_{index:05d}.jpg"
    filepath = os.path.join(SAVE_DIR, filename)

    with open(filepath, "wb") as f:
        f.write(r.content)

    print(f"[{datetime.now()}] Sparade: {filepath}")
    print(f"Senast ändrad: {last_modified}, Ny uppdatering väntas: {expires}")

    # Spara metadata i databasen.
    database_utils.save_metadata(AREA, last_modified, expires, filepath)

    # Räkna ut hur länge vi ska vänta
    sleep_seconds = (expires - date).total_seconds()
    return max(0, sleep_seconds)
