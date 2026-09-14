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

sys.path.append("/home/bubai-maji/speech_enhancement/script/propose")
sys.path.append("/home/bubai-maji/speech_enhancement/script/others")   # for data_loader.py

from data_loader import SpeechEnhDataset
from ngdm_xlstm import NGDM_SE


##################################
# CONFIG
##################################

DEVICE = "cuda:6" if torch.cuda.is_available() else "cpu"
SR = 16000

TEST_SETS = [
    ("in_domain_EARS_MUSAN",
     "/home/bubai-maji/speech_enhancement/data/mixtures/musan/test",
     "results_ablations_in_domain.csv"),

   # ("unseen_noise_EARS_DEMAND",
     #"/home/bubai-maji/speech_enhancement/data/mixtures/demand/test",
    # "results_ablations_unseen_noise.csv"),

    #("unseen_corpus_VoiceBank_DEMAND",
    # "/home/bubai-maji/speech_enhancement/data/voicebank_demand/test",
     #"results_ablations_unseen_corpus.csv"),
]

# CRITICAL: each entry's constructor lambda MUST match the exact flags that
# specific checkpoint was TRAINED with, or you get the "Missing key(s)"
# error again. Verify each checkpoint path against your actual training
# runs before running this.
MODELS = {
    "NGDM_SE_ablation2_no_dual_memory": (
        lambda: NGDM_SE(use_dual_memory=False),
        "/home/bubai-maji/speech_enhancement/script/propose_ablation2/checkpoint_propose_ab2/best_NGDM_SE.pt",
    ),
    "NGDM_SE_ablation3_no_attention_fusion": (
        lambda: NGDM_SE(use_attention_fusion=False),
        "/home/bubai-maji/speech_enhancement/script/propose_ablation3/checkpoint_propose_ab3/best_NGDM_SE.pt",
    ),
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
    return 10 * np.log10((np.sum(proj**2)+eps) / (np.sum(noise**2)+eps))


def count_params(model):
    return round(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6, 3)


def load_model(factory, ckpt):
    model = factory().to(DEVICE)   # factory() applies the correct ablation flags
    state = torch.load(ckpt, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


def compute_metrics(ref, est):
    L = min(len(ref), len(est))
    ref, est = ref[:L], est[:L]

    if L < SR:
        return None
    if np.sqrt(np.mean(est**2)) < 1e-5:
        return None

    scores = {"SI-SDR": si_sdr(est, ref)}

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


def evaluate(name, factory, ckpt_path, loader):
    if not os.path.exists(ckpt_path):
        print(f"  [skip] Missing checkpoint: {ckpt_path}")
        return None

    model = load_model(factory, ckpt_path)
    params = count_params(model)

    metrics_store = {"SI-SDR": [], "PESQ": [], "STOI": [], "ESTOI": [],
                      "DNSMOS_SIG": [], "DNSMOS_BAK": [], "DNSMOS_OVRL": []}

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

    final = {"Model": name, "Params (M)": params}
    for k, v in metrics_store.items():
        arr = np.array(v)
        arr = arr[~np.isnan(arr)]
        final[f"{k}_mean"] = np.mean(arr) if len(arr) else np.nan
        final[f"{k}_std"] = np.std(arr) if len(arr) else np.nan

    del model
    torch.cuda.empty_cache()
    return final


def main():
    for test_name, test_root, out_csv in TEST_SETS:
        print(f"\n{'='*70}\nTEST SET: {test_name}\nRoot: {test_root}\n{'='*70}")

        if not os.path.isdir(test_root):
            print(f"  [skip entire test set] Folder not found: {test_root}")
            continue

        dataset = SpeechEnhDataset(test_root)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

        results = []
        for name, (factory, ckpt_path) in MODELS.items():
            print(f"\nEvaluating {name} on {test_name}...")
            scores = evaluate(name, factory, ckpt_path, loader)
            if scores is not None:
                results.append(scores)

        if not results:
            print(f"  No models evaluated for {test_name} -- skipping CSV.")
            continue

        df = pd.DataFrame(results)
        print(f"\n---------- RESULTS: {test_name} ----------\n")
        print(df.to_string(index=False))
        df.to_csv(out_csv, index=False)
        print(f"\nSaved -> {out_csv}")

    print("\nAll ablation test sets complete.")


if __name__ == "__main__":
    main()