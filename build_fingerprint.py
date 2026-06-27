"""
build_fingerprint.py
=====================
Builds a "fingerprint" of a song and saves it as a CSV file.

For a given song (e.g. "Magical Mystery Tune.mp3") this script:
  1. Computes its spectrogram (STFT magnitude, in dB).
  2. Finds the "constellation" of strongest local-maxima peaks in the
     spectrogram (the (time, frequency) points that best characterise
     the song -- this is exactly the sparse picture Q3A talks about).
  3. Pairs nearby peaks together into hashes of the form
         hash = (f1, f2, dt)
     where f1 is the anchor peak's frequency, f2 is a nearby peak's
     frequency, and dt is the time gap between them. Each hash is
     stored together with the anchor time t1.
  4. Saves all (hash, t1) rows to a CSV named after the song
     (".mp3" stripped) inside a folder you hardcode below.

It also produces and saves three plots for every song:
  - the raw spectrogram
  - the spectrogram with the constellation of peaks overlaid
  - the constellation ALONE (peaks only, no spectrogram colours
    underneath -- just the dots, on a plain background)

Run it once per song (or loop over a folder of songs -- see bottom).
"""

import os
import gc
import csv
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter, generate_binary_structure, iterate_structure

# ----------------------------------------------------------------------
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>> HARDCODE ME <<<<<<<<<<<<<<<<<<<<<<<<<<<<
# ----------------------------------------------------------------------
DATABASE_FOLDER = "./csvs"   # where fingerprint CSVs are saved
PLOTS_FOLDER = "./plots"               # where the 3 plots per song are saved
# ----------------------------------------------------------------------

# ---- STFT / spectrogram parameters --------------------------------------
N_FFT = 4096          # window size for the FFT (frequency resolution)
HOP_LENGTH = 512       # hop between successive windows (time resolution)

# ---- Constellation (peak-picking) parameters -----------------------------
PEAK_NEIGHBORHOOD_SIZE = 30   # how large a local neighbourhood a point must
                              # dominate to be called a "peak"
MIN_AMPLITUDE_DB = -40        # ignore peaks quieter than this (mostly silence/noise)

# ---- Hashing (fingerprinting) parameters ---------------------------------
FAN_OUT = 8           # how many neighbouring peaks each anchor pairs with
MIN_TIME_DELTA = 0    # don't pair points closer than this many frames in time
MAX_TIME_DELTA = 200  # ...or further apart than this (keeps hashes "local")


def compute_spectrogram(y, sr):
    """Return spectrogram in dB, plus the frequency/time axes."""
    S = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    times = librosa.frames_to_time(np.arange(S_db.shape[1]), sr=sr, hop_length=HOP_LENGTH)
    return S_db, freqs, times


def get_constellation(S_db):
    """
    Find local-maxima peaks in the spectrogram.
    Returns a list of (freq_bin_index, time_frame_index) tuples,
    sorted by time.
    """
    struct = generate_binary_structure(2, 1)
    neighborhood = iterate_structure(struct, PEAK_NEIGHBORHOOD_SIZE)

    local_max = maximum_filter(S_db, footprint=neighborhood) == S_db
    above_threshold = S_db > MIN_AMPLITUDE_DB
    detected_peaks = local_max & above_threshold

    freq_idx, time_idx = np.where(detected_peaks)
    peaks = sorted(zip(time_idx, freq_idx), key=lambda p: p[0])
    return peaks  # list of (time_idx, freq_idx)


def generate_hashes(peaks):
    """
    Pair each anchor peak with up to FAN_OUT nearby peaks that come
    later in time, and turn each pair into a hash:
        hash_str = "f1|f2|dt"
    Returns a list of (hash_str, t1) rows ready to write to CSV.
    t1 is the anchor's time index (frame), which lets us later figure
    out the offset between a matched song and a query clip.
    """
    rows = []
    n = len(peaks)
    for i in range(n):
        t1, f1 = peaks[i]
        for j in range(1, FAN_OUT + 1):
            if i + j < n:
                t2, f2 = peaks[i + j]
                dt = t2 - t1
                if MIN_TIME_DELTA <= dt <= MAX_TIME_DELTA:
                    hash_str = f"{f1}|{f2}|{dt}"
                    rows.append((hash_str, t1))
    return rows


def plot_spectrogram(S_db, sr, save_path):
    plt.figure(figsize=(10, 5))
    librosa.display.specshow(S_db, sr=sr, hop_length=HOP_LENGTH,
                              x_axis="time", y_axis="hz", cmap="magma")
    plt.colorbar(format="%+2.0f dB")
    plt.title("Spectrogram")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_spectrogram_with_constellation(S_db, sr, peaks, save_path):
    plt.figure(figsize=(12, 6))
    librosa.display.specshow(S_db, sr=sr, hop_length=HOP_LENGTH,
                              x_axis="time", y_axis="hz", cmap="magma")
    plt.colorbar(format="%+2.0f dB")
    times = librosa.frames_to_time([p[0] for p in peaks], sr=sr, hop_length=HOP_LENGTH)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)[[p[1] for p in peaks]]
    plt.scatter(times, freqs, color="cyan", s=4, marker="o")
    plt.title("Spectrogram with Constellation of Peaks")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_constellation_only(S_db, sr, peaks, save_path):
    """Same peaks, but with the spectrogram colours stripped away --
    just the dots on a plain background, like a star chart."""
    times = librosa.frames_to_time([p[0] for p in peaks], sr=sr, hop_length=HOP_LENGTH)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)[[p[1] for p in peaks]]

    plt.figure(figsize=(12, 6))
    plt.scatter(times, freqs, color="cyan", s=4, marker="o")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title("Constellation Map (peaks only)")
    plt.xlim(0, S_db.shape[1] * HOP_LENGTH / sr)
    plt.ylim(0, sr / 2)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def fingerprint_song(song_path):
    """Full pipeline for one song: spectrogram -> constellation -> hashes -> CSV + plots."""
    os.makedirs(DATABASE_FOLDER, exist_ok=True)
    os.makedirs(PLOTS_FOLDER, exist_ok=True)

    song_name = os.path.splitext(os.path.basename(song_path))[0]  # strips .mp3
    print(f"Fingerprinting: {song_name}")

    y, sr = librosa.load(song_path, sr=None, mono=True)
    S_db, freqs, times = compute_spectrogram(y, sr)
    peaks = get_constellation(S_db)
    rows = generate_hashes(peaks)

    # --- save fingerprint CSV ---
    csv_path = os.path.join(DATABASE_FOLDER, f"{song_name}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["hash", "time"])
        writer.writerows(rows)
    print(f"  -> {len(rows)} hashes saved to {csv_path}")
    
    # --- save the 3 plots ---
    plot_spectrogram(S_db, sr, os.path.join(PLOTS_FOLDER, f"{song_name}_spectrogram.png"))
    plot_spectrogram_with_constellation(S_db, sr, peaks,
                                         os.path.join(PLOTS_FOLDER, f"{song_name}_constellation_overlay.png"))
    plot_constellation_only(S_db, sr, peaks,
                             os.path.join(PLOTS_FOLDER, f"{song_name}_constellation_only.png"))
    print(f"  -> plots saved to {PLOTS_FOLDER}")

    plt.close('all')  # Force Matplotlib to wipe all figures from memory
    del y, S_db, freqs, times, peaks, rows  # Delete the heavy variables
    gc.collect()  # Force the garbage collector to free up the RAM



if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Option A: fingerprint a single song
    # ------------------------------------------------------------------
    # SONG_PATH = "/path/to/your/song.mp3"   # <-- hardcode the song you want to process
    # fingerprint_song(SONG_PATH)

    # ------------------------------------------------------------------
    # Option B: fingerprint every .mp3 in a folder (uncomment to use)
    # ------------------------------------------------------------------
    SONGS_FOLDER = "/home/saksham/Documents/EE200 project/Q3/song"
    for fname in os.listdir(SONGS_FOLDER):
        if fname.lower().endswith(".mp3"):
            fingerprint_song(os.path.join(SONGS_FOLDER, fname))