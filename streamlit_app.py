"""
=============================================================
Q3A & Q3B – Zapptain America / Signals to Softwares
Audio fingerprinting – works on Python 3.12 / 3.13 / 3.14
NO pydub, NO librosa, NO audioop, NO numba
Uses subprocess + ffmpeg to decode audio directly
=============================================================
"""

import os
import csv
import pickle
import io
import tempfile
import subprocess
import struct
import glob

import numpy as np
from scipy.ndimage import maximum_filter
from scipy.signal import spectrogram as scipy_spectrogram

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import streamlit as st

# ─────────────────────────────────────────────────────────
#  GLOBAL CONSTANTS
# ─────────────────────────────────────────────────────────

SAMPLE_RATE    = 22050
N_FFT          = 4096
HOP_LENGTH     = 512
N_PEAKS        = 10
FAN_VALUE      = 5
MIN_TIME_DELTA = 1
MAX_TIME_DELTA = 100
FREQ_RANGE     = 200
DB_FILE        = "fingerprint_db.pkl"
SONGS_FOLDER   = "songs"
THUMBS_FOLDER  = "database/thumbs"

# ─────────────────────────────────────────────────────────
#  AUDIO LOADING via ffmpeg subprocess
# ─────────────────────────────────────────────────────────

def _ffmpeg_to_pcm(input_path):
    cmd = [
        "ffmpeg", "-v", "quiet",
        "-i", input_path,
        "-f", "s16le",
        "-acodec", "pcm_s16le",
        "-ar", str(SAMPLE_RATE),
        "-ac", "1",
        "pipe:1"
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed: {result.stderr.decode('utf-8', errors='ignore')}"
        )
    return result.stdout


def load_audio(source):
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    if hasattr(source, "read"):
        source.seek(0)
        raw_data = source.read()
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp.write(raw_data)
            tmp_path = tmp.name
        try:
            pcm_bytes = _ffmpeg_to_pcm(tmp_path)
        finally:
            os.unlink(tmp_path)
    else:
        pcm_bytes = _ffmpeg_to_pcm(str(source))

    n_samples = len(pcm_bytes) // 2
    samples = struct.unpack(f"<{n_samples}h", pcm_bytes)
    y = np.array(samples, dtype=np.float32)
    y /= 32768.0
    return y, SAMPLE_RATE


# ─────────────────────────────────────────────────────────
#  STEP 1 – SPECTROGRAM
# ─────────────────────────────────────────────────────────

def compute_spectrogram(y, sr):
    freqs, times, Sxx = scipy_spectrogram(
        y, fs=sr, nperseg=N_FFT,
        noverlap=N_FFT - HOP_LENGTH, scaling="spectrum",
    )
    S_db = 10 * np.log10(Sxx + 1e-10)
    return S_db, freqs, times


def plot_spectrogram(S_db, freqs, times, title="Spectrogram"):
    fig, ax = plt.subplots(figsize=(10, 4))
    img = ax.pcolormesh(times, freqs, S_db, shading="auto", cmap="magma")
    fig.colorbar(img, ax=ax, label="Power (dB)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────
#  STEP 2 – CONSTELLATION
# ─────────────────────────────────────────────────────────

def extract_peaks(S_db, n_peaks=N_PEAKS):
    local_max = maximum_filter(S_db, size=20) == S_db
    strong    = S_db > S_db.max() - 60
    peak_mask = local_max & strong
    freq_indices, time_indices = np.where(peak_mask)
    peaks = []
    for t in np.unique(time_indices):
        mask_t    = time_indices == t
        f_at_t    = freq_indices[mask_t]
        strengths = S_db[f_at_t, t]
        top_idx   = np.argsort(strengths)[::-1][:n_peaks]
        for idx in top_idx:
            peaks.append((int(f_at_t[idx]), int(t)))
    return peaks


def plot_constellation(S_db, freqs, times, peaks, title="Constellation"):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.pcolormesh(times, freqs, S_db, shading="auto", cmap="magma", alpha=0.7)
    pt = [times[t] if t < len(times) else times[-1] for (f, t) in peaks]
    pf = [freqs[f] if f < len(freqs) else freqs[-1] for (f, t) in peaks]
    ax.scatter(pt, pf, s=15, c="cyan", marker="o", linewidths=0.5,
               label="Constellation peaks")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=7)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────
#  STEP 3 – FINGERPRINTING
# ─────────────────────────────────────────────────────────

def generate_hashes(peaks, fan_value=FAN_VALUE):
    hashes = []
    peaks_sorted = sorted(peaks, key=lambda x: x[1])
    for i, (f1, t1) in enumerate(peaks_sorted):
        count = 0
        for j in range(i + 1, len(peaks_sorted)):
            f2, t2  = peaks_sorted[j]
            delta_t = t2 - t1
            if delta_t < MIN_TIME_DELTA:
                continue
            if delta_t > MAX_TIME_DELTA:
                break
            if abs(f2 - f1) > FREQ_RANGE:
                continue
            h = int(f1) * 10**10 + int(f2) * 10**5 + int(delta_t)
            hashes.append((h, t1))
            count += 1
            if count >= fan_value:
                break
    return hashes


# ─────────────────────────────────────────────────────────
#  STEP 4 – DATABASE
# ─────────────────────────────────────────────────────────

def build_database(songs_folder=SONGS_FOLDER, db_file=DB_FILE):
    if os.path.exists(db_file):
        with open(db_file, "rb") as f:
            return pickle.load(f)

    db = {}
    os.makedirs(songs_folder, exist_ok=True)
    audio_files = [fn for fn in os.listdir(songs_folder)
                   if fn.lower().endswith((".mp3", ".wav", ".flac", ".ogg"))]

    if not audio_files:
        st.error(f"No audio files found in '{songs_folder}' folder.")
        return db

    progress = st.progress(0, text="Indexing songs…")
    for idx, filename in enumerate(audio_files):
        song_name = os.path.splitext(filename)[0]
        path      = os.path.join(songs_folder, filename)
        try:
            y, sr  = load_audio(path)
            S_db, freqs, times = compute_spectrogram(y, sr)
            peaks  = extract_peaks(S_db)
            hashes = generate_hashes(peaks)
            for (h, t) in hashes:
                db.setdefault(h, []).append((song_name, t))
        except Exception as e:
            st.warning(f"Skipped {filename}: {e}")
        progress.progress((idx + 1) / len(audio_files),
                          text=f"Indexed: {song_name}")

    with open(db_file, "wb") as f:
        pickle.dump(db, f)
    progress.empty()
    return db


# ─────────────────────────────────────────────────────────
#  STEP 5 – MATCHING
# ─────────────────────────────────────────────────────────

def match_query(query_y, query_sr, db):
    S_db, freqs, times = compute_spectrogram(query_y, query_sr)
    peaks  = extract_peaks(S_db)
    hashes = generate_hashes(peaks)

    offset_dict = {}
    for (h, qt) in hashes:
        if h not in db:
            continue
        for (song_name, db_t) in db[h]:
            offset = db_t - qt
            offset_dict.setdefault(song_name, {})
            offset_dict[song_name][offset] = \
                offset_dict[song_name].get(offset, 0) + 1

    if not offset_dict:
        return None, 0, {}, S_db, freqs, times, peaks

    best_song, best_count = None, 0
    for song, offsets in offset_dict.items():
        top = max(offsets, key=offsets.get)
        if offsets[top] > best_count:
            best_count = offsets[top]
            best_song  = song

    return best_song, best_count, offset_dict, S_db, freqs, times, peaks


def plot_offset_histogram(offset_dict, best_song):
    top_songs = sorted(offset_dict.items(),
                       key=lambda kv: max(kv[1].values()),
                       reverse=True)[:3]
    fig, axes = plt.subplots(1, len(top_songs), figsize=(12, 3))
    if len(top_songs) == 1:
        axes = [axes]
    for ax, (song, offsets) in zip(axes, top_songs):
        color = "crimson" if song == best_song else "steelblue"
        ax.bar(list(offsets.keys()), list(offsets.values()),
               width=1, color=color, alpha=0.8)
        ax.set_title(song[:25] + ("…" if len(song) > 25 else ""),
                     fontsize=8, color=color)
        ax.set_xlabel("Time offset (frames)")
        ax.set_ylabel("Hash count")
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────
#  STEP 6 – BATCH MODE
# ─────────────────────────────────────────────────────────

def run_batch_mode(query_files, db):
    rows = []
    for uploaded in query_files:
        y, sr = load_audio(uploaded.read())
        best_song, *_ = match_query(y, sr, db)
        rows.append({"filename": uploaded.name,
                     "prediction": best_song if best_song else "unknown"})
    return rows


def write_results_csv(rows, path="results.csv"):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "prediction"])
        writer.writeheader()
        writer.writerows(rows)
    return path


# ─────────────────────────────────────────────────────────
#  THUMBS HELPERS
# ─────────────────────────────────────────────────────────

def get_thumb_songs(thumbs_folder=THUMBS_FOLDER):
    """
    Return sorted list of unique song names that have a cover thumbnail.
    Cover thumbnails are named exactly '<song_name>.png' (no suffix like
    _spectrogram, _constellation_only, _constellation_overlay).
    """
    if not os.path.isdir(thumbs_folder):
        return []
    suffixes = (
        "_spectrogram.png",
        "_constellation_only.png",
        "_constellation_overlay.png",
    )
    songs = []
    for p in glob.glob(os.path.join(thumbs_folder, "*.png")):
        name = os.path.basename(p)
        if not any(name.endswith(s) for s in suffixes):
            songs.append(os.path.splitext(name)[0])
    return sorted(songs)


def thumb_path(song_name, kind="cover", thumbs_folder=THUMBS_FOLDER):
    """
    Build the path for a thumbnail image of a given kind.
    kind: 'cover' | 'spectrogram' | 'constellation_only' | 'constellation_overlay'
    """
    mapping = {
        "cover":                 f"{song_name}.png",
        "spectrogram":           f"{song_name}_spectrogram.png",
        "constellation_only":    f"{song_name}_constellation_only.png",
        "constellation_overlay": f"{song_name}_constellation_overlay.png",
    }
    return os.path.join(thumbs_folder, mapping[kind])


def show_song_thumbnails(song_name, thumbs_folder=THUMBS_FOLDER):
    """
    Render the three analysis thumbnails (spectrogram, overlay, peaks-only)
    for a matched song side-by-side. Falls back gracefully if any are missing.
    """
    kinds = [
        ("spectrogram",           "📊 Spectrogram"),
        ("constellation_overlay", "🔵 Constellation Overlay"),
        ("constellation_only",    "✦ Constellation Peaks Only"),
    ]
    cols = st.columns(len(kinds))
    for col, (kind, label) in zip(cols, kinds):
        path = thumb_path(song_name, kind, thumbs_folder)
        if os.path.exists(path):
            img = Image.open(path)
            col.image(img, caption=label, use_container_width=True)
        else:
            col.info(f"_{label} not available_")


# ─────────────────────────────────────────────────────────
#  HOME PAGE – THUMBNAIL GALLERY
# ─────────────────────────────────────────────────────────

def show_home_gallery(thumbs_folder=THUMBS_FOLDER, cols_per_row=4):
    """
    Display a responsive grid of cover thumbnails for every song in the
    thumbs folder. Each card shows the cover image with the song name.
    """
    songs = get_thumb_songs(thumbs_folder)

    if not songs:
        st.info(
            f"No thumbnails found in `{thumbs_folder}/`. "
            "The gallery will populate once thumbnails are generated."
        )
        return

    st.markdown(f"**{len(songs)} songs indexed** — browse the library below ↓")
    st.markdown("---")

    for row_start in range(0, len(songs), cols_per_row):
        row_songs = songs[row_start : row_start + cols_per_row]
        cols = st.columns(cols_per_row)
        for col, song in zip(cols, row_songs):
            cover = thumb_path(song, "cover", thumbs_folder)
            if os.path.exists(cover):
                col.image(Image.open(cover), use_container_width=True)
            else:
                col.markdown("🎵")
            display_name = song if len(song) <= 22 else song[:20] + "…"
            col.caption(f"**{display_name}**")


# ─────────────────────────────────────────────────────────
#  STREAMLIT UI
# ─────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Zapptain America – EE200",
        page_icon="⚡",
        layout="wide",
    )

    st.title("⚡ Zapptain America")
    st.markdown(
        "Shazam-style audio fingerprinting · "
        "spectrogram → constellation → (f₁, f₂, Δt) hashes → offset-histogram matching"
    )

    # ── Sidebar ──────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Settings")
        songs_path  = st.text_input("Songs folder path",  value=SONGS_FOLDER)
        thumbs_path = st.text_input("Thumbs folder path", value=THUMBS_FOLDER)
        if st.button("🔄 Re-index songs"):
            if os.path.exists(DB_FILE):
                os.remove(DB_FILE)
                st.success("Database cleared. Will re-index now.")
        st.markdown("---")
        st.caption(f"N_FFT={N_FFT} | Hop={HOP_LENGTH} | SR={SAMPLE_RATE}")

    # ── Load / build database ────────────────────────────
    db_exists  = os.path.exists(DB_FILE)
    dir_exists = os.path.isdir(songs_path)

    if not db_exists and not dir_exists:
        st.error(
            f"Folder `{songs_path}` not found and no pre-built database exists. "
            f"Add your .mp3 files to `{songs_path}/`."
        )
        st.stop()

    if not dir_exists:
        os.makedirs(songs_path, exist_ok=True)

    with st.spinner("Loading fingerprint database…"):
        db = build_database(songs_folder=songs_path, db_file=DB_FILE)

    st.sidebar.success(f"DB loaded – {len(db):,} unique hashes.")

    # ── Tabs ─────────────────────────────────────────────
    tab_home, tab_single, tab_batch = st.tabs(
        ["🏠 Library", "🎤 Single Clip", "📂 Batch Mode"]
    )

    # ════════════════════════════════════════════════════
    #  TAB 0 – HOME / LIBRARY GALLERY
    # ════════════════════════════════════════════════════
    with tab_home:
        st.subheader("🎵 Song Library")
        show_home_gallery(thumbs_folder=thumbs_path)

    # ════════════════════════════════════════════════════
    #  TAB 1 – SINGLE CLIP
    # ════════════════════════════════════════════════════
    with tab_single:
        st.subheader("Upload a Query Clip")
        uploaded = st.file_uploader(
            "Upload a short .mp3 or .wav clip",
            type=["mp3", "wav", "flac", "ogg"],
            key="single",
        )

        if uploaded:
            query_bytes = uploaded.read()
            st.audio(query_bytes)

            st.markdown("### 1 · Spectrogram of Query Clip")
            st.caption(
                "Bright = loud. Each column is one DFT window. "
                "A single DFT of the whole song loses all timing info."
            )
            y, sr = load_audio(query_bytes)
            S_db, freqs, times = compute_spectrogram(y, sr)
            fig = plot_spectrogram(S_db, freqs, times,
                                   title=f"Spectrogram – {uploaded.name}")
            st.pyplot(fig)
            plt.close(fig)

            st.markdown("### 2 · Constellation of Query Clip")
            st.caption("Only loud local maxima survive – sparse and noise-robust.")
            peaks = extract_peaks(S_db)
            fig = plot_constellation(S_db, freqs, times, peaks,
                                     title="Constellation (cyan dots)")
            st.pyplot(fig)
            plt.close(fig)

            st.markdown("### 3 · Match Result")
            with st.spinner("Matching against database…"):
                best, count, offsets, *_ = match_query(y, sr, db)

            if best:
                st.success(f"**Match found:** `{best}`  ({count} aligned hashes)")

                # ── Full pre-generated graphs for matched song ──
                st.markdown(f"### 4 · Full-Song Graphs for  _{best}_")
                st.caption(
                    "Pre-generated from the `thumbs/` folder: "
                    "spectrogram, constellation overlay, and peaks-only views "
                    "of the entire matched track."
                )
                show_song_thumbnails(best, thumbs_folder=thumbs_path)

            else:
                st.warning("No match found.")

            if offsets:
                st.markdown("### 5 · Offset Histogram")
                st.caption(
                    "True match → one very tall bar at the alignment offset. "
                    "Wrong songs → scattered low bars."
                )
                fig = plot_offset_histogram(offsets, best)
                st.pyplot(fig)
                plt.close(fig)

    # ════════════════════════════════════════════════════
    #  TAB 2 – BATCH MODE
    # ════════════════════════════════════════════════════
    with tab_batch:
        st.subheader("Batch Identification")
        batch_files = st.file_uploader(
            "Upload multiple clips",
            type=["mp3", "wav", "flac", "ogg"],
            accept_multiple_files=True,
            key="batch",
        )

        if batch_files and st.button("▶ Run Batch"):
            with st.spinner("Processing…"):
                rows = run_batch_mode(batch_files, db)

            st.dataframe(rows, use_container_width=True)

            # Show thumbnails for every unique matched song
            matched_songs = list({r["prediction"] for r in rows
                                   if r["prediction"] != "unknown"})
            if matched_songs:
                st.markdown("### Matched Song Graphs")
                for song in sorted(matched_songs):
                    st.markdown(f"#### {song}")
                    show_song_thumbnails(song, thumbs_folder=thumbs_path)
                    st.markdown("---")

            csv_path = write_results_csv(rows)
            with open(csv_path, "rb") as f:
                st.download_button(
                    "⬇ Download results.csv",
                    data=f.read(),
                    file_name="results.csv",
                    mime="text/csv",
                )


if __name__ == "__main__":
    main()
