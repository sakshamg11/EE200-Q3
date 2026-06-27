"""
match_clip.py
=============
Takes a short query clip, fingerprints it the same way build_fingerprint.py
fingerprints full songs, and matches it against every song's CSV in your
hardcoded database folder using the classic offset-histogram voting scheme:

  For every hash that the clip shares with a candidate song, compute
      offset = song_time - clip_time
  A TRUE match produces a sharp spike at one particular offset (all the
  shared hashes "agree" on how the clip is shifted in time relative to
  the song). A WRONG song just produces scattered, near-uniform offsets
  with no dominant spike.

The song whose histogram has the tallest spike (most votes in a single
offset bin) is reported as the match.

Outputs:
  - prints the matched song's name
  - plots the clip's spectrogram
  - plots the clip's spectrogram with its constellation of peaks
  - plots the offset histogram for the best-matching song (this is the
    evidence that decided the match)
"""

import os
import csv
import glob
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from collections import Counter

# Re-use the exact same spectrogram / constellation / hashing logic as the
# database builder, so query hashes are directly comparable to song hashes.
from build_fingerprint import (
    compute_spectrogram,
    get_constellation,
    generate_hashes,
    N_FFT,
    HOP_LENGTH,
)

# ----------------------------------------------------------------------
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>> HARDCODE ME <<<<<<<<<<<<<<<<<<<<<<<<<<<<
# ----------------------------------------------------------------------
DATABASE_FOLDER = "/home/saksham/Documents/EE200 project/Q3/csvs"  # folder of song fingerprint CSVs
TEMP_FOLDER = "/home/saksham/Documents/EE200 project/Q3/temp"                # where the clip's temp CSV is saved
# ----------------------------------------------------------------------


def fingerprint_clip(clip_path):
    """Compute spectrogram, constellation and hashes for a query clip.
    Also writes a temporary CSV of the clip's own hashes (handy for
    debugging / inspection), just like a mini version of a song's CSV."""
    os.makedirs(TEMP_FOLDER, exist_ok=True)

    clip_name = os.path.splitext(os.path.basename(clip_path))[0]
    y, sr = librosa.load(clip_path, sr=None, mono=True)
    S_db, freqs, times = compute_spectrogram(y, sr)
    peaks = get_constellation(S_db)
    rows = generate_hashes(peaks)

    temp_csv_path = os.path.join(TEMP_FOLDER, f"{clip_name}_temp.csv")
    with open(temp_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["hash", "time"])
        writer.writerows(rows)

    return S_db, sr, peaks, rows, temp_csv_path


def load_song_hashes(csv_path):
    """Load a song's fingerprint CSV into {hash: [t1, t1, ...]} for fast lookup
    (a hash can occur more than once in the same song)."""
    hash_to_times = {}
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for hash_str, t1 in reader:
            hash_to_times.setdefault(hash_str, []).append(int(t1))
    return hash_to_times


def match_against_song(clip_rows, song_hash_to_times):
    """
    Build the offset histogram between the clip's hashes and one song's
    hashes. Returns (offset_counter, best_offset, best_score).
    """
    offsets = []
    for hash_str, t_clip in clip_rows:
        if hash_str in song_hash_to_times:
            for t_song in song_hash_to_times[hash_str]:
                offsets.append(t_song - t_clip)

    if not offsets:
        return Counter(), None, 0

    offset_counts = Counter(offsets)
    best_offset, best_score = offset_counts.most_common(1)[0]
    return offset_counts, best_offset, best_score


def find_match(clip_rows, database_folder):
    """Try the clip against every song CSV in the database folder.
    Returns the winning song name, its score, and its offset histogram
    (for plotting), plus a dict of all songs' scores for reference."""
    best_song = None
    best_score = -1
    best_offset = None
    best_histogram = None
    all_scores = {}

    for csv_path in glob.glob(os.path.join(database_folder, "*.csv")):
        song_name = os.path.splitext(os.path.basename(csv_path))[0]
        song_hash_to_times = load_song_hashes(csv_path)
        offset_counts, offset, score = match_against_song(clip_rows, song_hash_to_times)
        all_scores[song_name] = score

        if score > best_score:
            best_score = score
            best_song = song_name
            best_offset = offset
            best_histogram = offset_counts

    return best_song, best_score, best_offset, best_histogram, all_scores


def plot_spectrogram(S_db, sr, title="Spectrogram of Query Clip"):
    plt.figure(figsize=(10, 5))
    librosa.display.specshow(S_db, sr=sr, hop_length=HOP_LENGTH,
                              x_axis="time", y_axis="hz", cmap="magma")
    plt.colorbar(format="%+2.0f dB")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_spectrogram_with_constellation(S_db, sr, peaks, title="Query Clip: Spectrogram + Constellation"):
    plt.figure(figsize=(10, 5))
    librosa.display.specshow(S_db, sr=sr, hop_length=HOP_LENGTH,
                              x_axis="time", y_axis="hz", cmap="magma")
    plt.colorbar(format="%+2.0f dB")
    times = librosa.frames_to_time([p[0] for p in peaks], sr=sr, hop_length=HOP_LENGTH)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)[[p[1] for p in peaks]]
    plt.scatter(times, freqs, color="cyan", s=8, marker="o")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_offset_histogram(offset_counter, song_name, best_offset):
    if not offset_counter:
        print("No matching hashes found against this song -- nothing to histogram.")
        return
    offsets = list(offset_counter.keys())
    counts = list(offset_counter.values())

    plt.figure(figsize=(10, 5))
    plt.bar(offsets, counts, width=1.0, color="steelblue")
    plt.axvline(best_offset, color="red", linestyle="--",
                label=f"Winning offset = {best_offset} (votes = {offset_counter[best_offset]})")
    plt.xlabel("Offset (song_time - clip_time, in STFT frames)")
    plt.ylabel("Number of matching hashes")
    plt.title(f"Offset Histogram vs. '{song_name}'")
    plt.legend()
    plt.tight_layout()
    plt.show()


def identify_clip(clip_path):
    print(f"Fingerprinting clip: {clip_path}")
    S_db, sr, peaks, clip_rows, temp_csv_path = fingerprint_clip(clip_path)
    print(f"  -> {len(clip_rows)} hashes (temp CSV: {temp_csv_path})")

    best_song, best_score, best_offset, best_histogram, all_scores = find_match(
        clip_rows, DATABASE_FOLDER
    )

    print("\nScores against every song in the database:")
    for song, score in sorted(all_scores.items(), key=lambda x: -x[1]):
        print(f"  {song}: {score} matching votes")

    if best_song is None or best_score == 0:
        print("\nNo match found.")
        return None

    print(f"\n>>> MATCHED SONG: {best_song}  (confidence votes = {best_score}) <<<")

    plot_spectrogram(S_db, sr)
    plot_spectrogram_with_constellation(S_db, sr, peaks)
    plot_offset_histogram(best_histogram, best_song, best_offset)

    return best_song


if __name__ == "__main__":
    CLIP_PATH = "/home/saksham/Documents/EE200 project/Q3/clip/1.mp3"  # <-- hardcode the clip you want to identify
    identify_clip(CLIP_PATH)