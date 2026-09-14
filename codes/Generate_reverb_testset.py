"""
generate_reverb_testset.py
"""
import os
import random
import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve

SR = 16000


MODE = "ears"   

EARS_CLEAN_TEST_DIR = "/home/bubai-maji/speech_enhancement/data/mixtures/musan/test/clean"
EARS_NOISE_DIR = "/home/bubai-maji/speech_enhancement/data/noise/musan"   # your existing noise source
SNR_LEVELS = [2.5, 7.5, 12.5, 17.5]   # matches your existing test-set SNR protocol

OUTPUT_ROOT_EARS = "/home/bubai-maji/speech_enhancement/data/mixtures_reverb/ears_musan/test"



RIR_ROOT = "/home/bubai-maji/speech_enhancement/data/RIRS_NOISES/real_rirs_isotropic_noises"

random.seed(42)
np.random.seed(42)


def collect_rirs(root):
    rirs = []
    for r, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(".wav"):
                rirs.append(os.path.join(r, f))
    return rirs


def collect_files(root, ext=".wav"):
    out = []
    for r, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(ext):
                out.append(os.path.join(r, f))
    return out


def load_audio(path, sr=SR):
    x, file_sr = sf.read(path)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if file_sr != sr:
        import librosa
        x = librosa.resample(x, orig_sr=file_sr, target_sr=sr)
    return x.astype(np.float32)


def load_and_trim_rir(rir_path):
    """EARS paper's exact fix: trim the RIR to start at its peak-amplitude
    index, avoiding a systematic delay between reverberant signal and its
    clean reference."""
    rir, _ = sf.read(rir_path)
    if rir.ndim > 1:
        rir = rir.mean(axis=1)
    rir = rir.astype(np.float32)
    peak_idx = np.argmax(np.abs(rir))
    rir = rir[peak_idx:]
    rir = rir / (np.max(np.abs(rir)) + 1e-8)
    return rir


def apply_rir(signal, rir, target_len):
    wet = fftconvolve(signal, rir)[:target_len]
    if len(wet) < target_len:
        wet = np.pad(wet, (0, target_len - len(wet)))
    max_amp = np.max(np.abs(wet))
    if max_amp > 0.99:
        wet = wet / max_amp * 0.99
    return wet


def rms(x):
    return np.sqrt(np.mean(x ** 2) + 1e-8)


def mix_at_snr(clean, noise, snr_db):
    clean_rms = rms(clean)
    noise_rms = rms(noise)
    noise = noise * (clean_rms / (noise_rms * 10 ** (snr_db / 20)))
    mixture = clean + noise
    max_amp = np.max(np.abs(mixture))
    if max_amp > 0.99:
        mixture = mixture / max_amp * 0.99
    return mixture


# =====================================================
# Mode 1: EARS -- proper methodology (reverberate clean speech, then mix)
# =====================================================

def run_ears(rir_files):
    out_mix = os.path.join(OUTPUT_ROOT_EARS, "mix")
    out_clean = os.path.join(OUTPUT_ROOT_EARS, "clean")
    os.makedirs(out_mix, exist_ok=True)
    os.makedirs(out_clean, exist_ok=True)

    clean_files = sorted(os.listdir(EARS_CLEAN_TEST_DIR))
    noise_files = collect_files(EARS_NOISE_DIR)

    print(f"EARS: {len(clean_files)} clean test utterances, {len(noise_files)} noise files")

    for i, fname in enumerate(clean_files):
        clean = load_audio(os.path.join(EARS_CLEAN_TEST_DIR, fname))

        rir = load_and_trim_rir(random.choice(rir_files))
        reverb_clean = apply_rir(clean, rir, target_len=len(clean))

        noise_path = random.choice(noise_files)
        noise = load_audio(noise_path)
        if len(noise) < len(reverb_clean):
            noise = np.tile(noise, int(np.ceil(len(reverb_clean) / len(noise))))
        noise = noise[:len(reverb_clean)]

    
        snr = random.choice(SNR_LEVELS)
        dry_clean_rms = rms(clean)
        noise_rms = rms(noise)
        noise_scaled = noise * (dry_clean_rms / (noise_rms * 10 ** (snr / 20)))

        reverb_mix = reverb_clean + noise_scaled
        max_amp = np.max(np.abs(reverb_mix))
        if max_amp > 0.99:
            reverb_mix = reverb_mix / max_amp * 0.99

        sf.write(os.path.join(out_clean, fname), reverb_clean, SR)
        sf.write(os.path.join(out_mix, fname), reverb_mix, SR)

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(clean_files)} done")

    print(f"EARS reverb test set saved -> {OUTPUT_ROOT_EARS}")


# =====================================================
# Mode 2: VoiceBank-DEMAND 
# =====================================================

def run_voicebank_demand(rir_files):
    out_mix = os.path.join(OUTPUT_ROOT_VB, "mix")
    out_clean = os.path.join(OUTPUT_ROOT_VB, "clean")
    os.makedirs(out_mix, exist_ok=True)
    os.makedirs(out_clean, exist_ok=True)

    filenames = sorted(os.listdir(VB_MIX_DIR))
    print(f"VoiceBank-DEMAND: {len(filenames)} utterances "
          f"(NOTE: using simplified same-RIR-for-mix approach -- "
          f"pre-mixed dataset, no separate clean/noise available)")

    for i, fname in enumerate(filenames):
        mix_path = os.path.join(VB_MIX_DIR, fname)
        clean_path = os.path.join(VB_CLEAN_DIR, fname)
        if not os.path.exists(clean_path):
            continue

        mix = load_audio(mix_path)
        clean = load_audio(clean_path)

        rir = load_and_trim_rir(random.choice(rir_files))

        reverb_mix = apply_rir(mix, rir, target_len=len(mix))
        reverb_clean = apply_rir(clean, rir, target_len=len(clean))

        sf.write(os.path.join(out_mix, fname), reverb_mix, SR)
        sf.write(os.path.join(out_clean, fname), reverb_clean, SR)

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(filenames)} done")

    print(f"VoiceBank-DEMAND reverb test set saved -> {OUTPUT_ROOT_VB}")


def main():
    rir_files = collect_rirs(RIR_ROOT)
    print(f"Found {len(rir_files)} RIR files in {RIR_ROOT}")
    if len(rir_files) == 0:
        raise RuntimeError(f"No RIR .wav files found under {RIR_ROOT}")

    if MODE == "ears":
        run_ears(rir_files)
    elif MODE == "voicebank_demand":
        run_voicebank_demand(rir_files)
    else:
        raise ValueError(f"Unknown MODE: {MODE}")


if __name__ == "__main__":
    main()
