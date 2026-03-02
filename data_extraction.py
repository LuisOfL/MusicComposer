import requests
from bs4 import BeautifulSoup
from pathlib import Path
import time

BASE_URL = "https://bitmidi.com"
OUT = Path("bitmidi_midis")
OUT.mkdir(exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

def get_song_pages(page_url):
    r = requests.get(page_url, headers=HEADERS, timeout=10)
    soup = BeautifulSoup(r.text, "html.parser")

    links = set()
    for a in soup.select("a[href]"):
        href = a["href"]
        # Las páginas de canciones son tipo /nombre-de-la-cancion-mid
        if href.endswith("-mid") and not href.startswith("http"):
            links.add(BASE_URL + href)

    return links


def get_midi_from_song(song_url):
    r = requests.get(song_url, headers=HEADERS, timeout=10)
    soup = BeautifulSoup(r.text, "html.parser")

    for a in soup.select("a[href$='.mid']"):
        return BASE_URL + a["href"]

    return None


def download_midi(url):
    name = url.split("/")[-1]
    path = OUT / name

    if path.exists():
        return

    print(f"⬇️ {name}")
    r = requests.get(url, headers=HEADERS, timeout=15)
    with open(path, "wb") as f:
        f.write(r.content)


def main(pages=20):
    song_pages = set()

    for p in range(1, pages + 1):
        url = f"{BASE_URL}/?page={p}"
        print(f"➡️ Escaneando {url}")
        song_pages |= get_song_pages(url)
        time.sleep(1)

    print(f"\n🎵 Encontradas {len(song_pages)} páginas de canciones\n")

    for song in song_pages:
        midi = get_midi_from_song(song)
        if midi:
            download_midi(midi)
            time.sleep(0.5)

    print("\n✅ Terminado")


if __name__ == "__main__":
    main(pages=50)