import os
import sys
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from pesq import pesq
from pystoi import stoi
from speechmos import dnsmos

# =====================================================
# PATHS -- each model lives in its own folder with model.py + checkpoint/
# =====================================================

sys.path.append("/home/bubai-maji/speech_enhancement/script/conv")
sys.path.append("/home/bubai-maji/speech_enhancement/script/dccr")
sys.path.append("/home/bubai-maji/speech_enhancement/script/mspnt")
sys.path.append("/home/bubai-maji/speech_enhancement/script/demus")
sys.path.append("/home/bubai-maji/speech_enhancement/script/propose")
sys.path.append("/home/bubai-maji/speech_enhancement/script/others")   # for data_loader.py

from conv.data_loader import SpeechEnhDataset

from conv.convsnet import ConvTasNet
from dccr.dccrn import DCCRN
from demus.demucs import Demucs
from mspnt.mpsenet import MPSENet
#from propose.ngdm_xlstm1 import NGDM_SE
#from propose.Ngdm_se_improved import NGDM_SE
from others.xlstm_senet import XLSTM_SENet
#from propose.Ngdm_se_mpsene import NGDM_SE
from propose.ngdm_xlstm import NGDM_SE
#from propose_ablation2.ngdm_xlstm import NGDM_SE
##################################
# CONFIG
##################################

DEVICE = "cuda:5" if torch.cuda.is_available() else "cpu"
SR = 16000

# One entry per test condition: (name, root_folder, output_csv)
TEST_SETS = [
    ("in_domain_EARS_MUSAN",
    "/home/bubai-maji/speech_enhancement/data/mixtures/musan/test",
     "results_in_domain_ears_musan_final.csv"),

    #("unseen_noise_EARS_DEMAND",
    # "/home/bubai-maji/speech_enhancement/data/mixtures/demand/test",
    # "results_unseen_noise_ears_demand.csv"),

    ("unseen_corpus_VoiceBank_DEMAND",
     "/home/bubai-maji/speech_enhancement/data/voicebank_demand/test",
     "results_voicebank_demand_final.csv"),

    ("out_of_domain_DNS_Challenge_2020",
     "/home/bubai-maji/speech_enhancement/data/dns_challenge_2020/test",
     "results_dns_challenge_2020_final.csv"),

    # add the reverberant test set here once built, e.g.:
    #("reverberant_EARS",
      #"/home/bubai-maji/speech_enhancement/data/mixtures_reverb/test",
      #"results_reverberant_final.csv"),
]

# Each model has its own checkpoint location -- store the FULL path directly
# (not a shared dir + filename), since they're not all in the same folder.
MODELS = {
    #"ConvTasNet": (ConvTasNet, "/home/bubai-maji/speech_enhancement/script/conv/checkpoint/best_ConvTasNet.pt"),
    #"DCCRN":      (DCCRN,      "/home/bubai-maji/speech_enhancement/script/dccr/checkpoint/best_DCCRN.pt"),
    #"MPSENet":    (MPSENet,    "/home/bubai-maji/speech_enhancement/script/mspnt/checkpoint/best_MPSENet.pt"),
    #"Demucs":     (Demucs,     "/home/bubai-maji/speech_enhancement/script/demus/checkpoint/best_Demucs.pt"),

    # not yet trained -- uncomment as each finishes:
    #"XLSTM_SENet": (XLSTM_SENet, "/home/bubai-maji/speech_enhancement/checkpoint_pm/best_XLSTM_SENet.pt"),
    # "NGDM_LSTM":    (NGDM_LSTM,    "/path/to/best_NGDM_LSTM.pt"),
    "NGDM_SE":      (NGDM_SE,     "/home/bubai-maji/speech_enhancement/script/propose/checkpoint/best_NGDM_SE.pt"),
}


##################################
# SI-SDR
##################################

def si_sdr(est, ref, eps=1e-8):
    ref = ref - ref.mean()
    est = est - est.mean()
    alpha = np.dot(est, ref) / (np.dot(ref, ref) + eps)
    proj = alpha * ref
    noise = est - proj
    return 10 * np.log10(
        (np.sum(proj**2) + eps) /
        (np.sum(noise**2) + eps)
    )


##################################
# PARAM COUNT
##################################

def count_params(model):
    return round(
        sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6, 3
    )


##################################
# LOAD MODEL
##################################

def load_model(cls, ckpt):
    model = cls().to(DEVICE)
    state = torch.load(ckpt, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


##################################
# METRICS
##################################

def compute_metrics(ref, est):
    L = min(len(ref), len(est))
    ref = ref[:L]
    est = est[:L]

    if L < SR:
        return None

    if np.sqrt(np.mean(est**2)) < 1e-5:
        return None

    scores = {}
    scores["SI-SDR"] = si_sdr(est, ref)

    try:
        scores["PESQ"] = pesq(SR, ref, est, "wb")
    except Exception:
        scores["PESQ"] = np.nan

    scores["STOI"] = stoi(ref, est, SR, extended=False)
    scores["ESTOI"] = stoi(ref, est, SR, extended=True)

    est_clipped = np.clip(est, -1, 1).astype(np.float32)

    try:
        mos = dnsmos.run(est_clipped, SR)
        scores["DNSMOS_SIG"] = mos["sig_mos"]
        scores["DNSMOS_BAK"] = mos["bak_mos"]
        scores["DNSMOS_OVRL"] = mos["ovrl_mos"]
    except Exception:
        scores["DNSMOS_SIG"] = np.nan
        scores["DNSMOS_BAK"] = np.nan
        scores["DNSMOS_OVRL"] = np.nan

    return scores


##################################
# EVALUATE ONE MODEL ON ONE TEST SET
##################################

def evaluate_model_on_testset(name, cls, ckpt_path, loader):

    if not os.path.exists(ckpt_path):
        print(f"  [skip] Missing checkpoint: {ckpt_path}")
        return None

    model = load_model(cls, ckpt_path)
    params = count_params(model)

    metrics_store = {
        "SI-SDR": [], "PESQ": [], "STOI": [], "ESTOI": [],
        "DNSMOS_SIG": [], "DNSMOS_BAK": [], "DNSMOS_OVRL": []
    }

    for noisy, clean in tqdm(loader, desc=f"  {name}", leave=False):
        noisy = noisy.to(DEVICE)

        with torch.no_grad():
            enhanced = model(noisy).squeeze().cpu().numpy()

        clean = clean.squeeze().numpy()

        scores = compute_metrics(clean, enhanced)
        if scores is None:
            continue

        for k in metrics_store:
            metrics_store[k].append(scores[k])

    final_scores = {"Model": name, "Params (M)": params}
    for k, v in metrics_store.items():
        arr = np.array(v)
        arr = arr[~np.isnan(arr)]
        final_scores[f"{k}_mean"] = np.mean(arr) if len(arr) else np.nan
        final_scores[f"{k}_std"] = np.std(arr) if len(arr) else np.nan

    del model
    torch.cuda.empty_cache()

    return final_scores


##################################
# MAIN
##################################

def main():

    for test_name, test_root, out_csv in TEST_SETS:

        print(f"\n{'='*70}")
        print(f"TEST SET: {test_name}")
        print(f"Root: {test_root}")
        print(f"{'='*70}")

        if not os.path.isdir(test_root):
            print(f"  [skip entire test set] Folder not found: {test_root}")
            continue

        dataset = SpeechEnhDataset(test_root)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

        results = []
        for name, (cls, ckpt_path) in MODELS.items():
            print(f"\nEvaluating {name} on {test_name}...")
            scores = evaluate_model_on_testset(name, cls, ckpt_path, loader)
            if scores is not None:
                results.append(scores)

        if not results:
            print(f"  No models evaluated for {test_name} -- skipping CSV.")
            continue

        df = pd.DataFrame(results)
        df = df.sort_values(by="PESQ_mean", ascending=False)

        print(f"\n---------- RESULTS: {test_name} ----------\n")
        print(df.to_string(index=False))

        df.to_csv(out_csv, index=False)
        print(f"\nSaved -> {out_csv}")

    print("\nAll test sets complete.")


if __name__ == "__main__":
    main()