"""
profile_efficiency.py

Standalone efficiency profiler -- addresses the meta-reviewer's request to
"add MAC/FLOPs/RTF figures". This is intentionally decoupled from
train.py: it does not need a trained checkpoint (FLOPs/params depend only
on architecture + input shape, not weight values), so you can run this in
parallel with training, today, for every model in the study.

Produces one row per model with:
    - Params (M)
    - MACs / FLOPs (G) for a fixed reference input length
    - RTF on GPU  (inference_time / audio_duration; lower is better; <1 = faster than real time)
    - RTF on CPU
    - Peak GPU memory during inference (MB)

Usage:
    pip install thop --break-system-packages   # if not already installed
    python profile_efficiency.py

Add/remove models in the MODELS_TO_PROFILE list at the bottom.
"""

import time
import statistics
import sys

import torch
import torch.nn as nn

# =====================================================
# PATHS -- match test_all_models.py's layout
# =====================================================

sys.path.append("/home/bubai-maji/speech_enhancement/script/conv")
sys.path.append("/home/bubai-maji/speech_enhancement/script/dccr")
sys.path.append("/home/bubai-maji/speech_enhancement/script/mspnt")
sys.path.append("/home/bubai-maji/speech_enhancement/script/demus")
sys.path.append("/home/bubai-maji/speech_enhancement/script/propose")
sys.path.append("/home/bubai-maji/speech_enhancement/script/others")

from conv.convsnet import ConvTasNet
from dccr.dccrn import DCCRN
from demus.demucs import Demucs
from mspnt.mpsenet import MPSENet
from others.xlstm_senet import XLSTM_SENet
from propose.ngdm_xlstm import NGDM_SE


try:
    from thop import profile as thop_profile
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False
    print("[warn] thop not installed -- MACs/FLOPs will be skipped. "
          "Install with: pip install thop --break-system-packages")


SR = 16000
REFERENCE_SECONDS = 1.0                     # profile on 1s of audio -> easy to read/scale
REFERENCE_LEN = int(SR * REFERENCE_SECONDS)
WARMUP_ITERS = 10
TIMED_ITERS = 50


def count_params(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6  # in millions


def count_macs_flops(model: nn.Module, input_len: int):
    """
    Returns (MACs_G, FLOPs_G) or (None, None) if profiling failed
    (e.g. thop not installed, or the model uses ops thop can't trace
    through such as torch.stft with return_complex=True).
    """
    if not THOP_AVAILABLE:
        return None, None

    model_cpu = model.to("cpu").eval()
    dummy_input = torch.randn(1, input_len)

    try:
        macs, params = thop_profile(model_cpu, inputs=(dummy_input,), verbose=False)
        macs_g = macs / 1e9
        flops_g = macs_g * 2  # standard convention: 1 MAC ~= 2 FLOPs
        return macs_g, flops_g
    except Exception as e:
        print(f"    [warn] thop failed for {model.__class__.__name__}: {e}")
        print("    (Common cause: complex-valued STFT ops aren't traceable by thop. "
              "See note in script docstring for a workaround.)")
        return None, None


@torch.no_grad()
def measure_rtf(model: nn.Module, input_len: int, device: str):
    """
    Real-Time Factor = inference_wall_clock_time / audio_duration_seconds.
    RTF < 1 means the model runs faster than real time (good for streaming/
    practical deployment claims). Includes warmup iterations (excluded from
    timing) to avoid first-call CUDA kernel compilation / cache effects.
    """
    model = model.to(device).eval()
    x = torch.randn(1, input_len, device=device)
    audio_duration_sec = input_len / SR

    # warmup
    for _ in range(WARMUP_ITERS):
        _ = model(x)
    if device == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(TIMED_ITERS):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    mean_time = statistics.mean(times)
    std_time = statistics.stdev(times) if len(times) > 1 else 0.0
    rtf = mean_time / audio_duration_sec

    peak_mem_mb = None
    if device == "cuda":
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    return rtf, mean_time, std_time, peak_mem_mb


def profile_model(model_cls, name: str, input_len: int = REFERENCE_LEN):
    print(f"\n=== {name} ===")
    row = {"model": name}

    # --- params ---
    model = model_cls()
    row["params_M"] = count_params(model)
    print(f"  Params        : {row['params_M']:.3f} M")

    # --- MACs / FLOPs (fresh instance -- thop moves to CPU internally) ---
    model_for_flops = model_cls()
    macs_g, flops_g = count_macs_flops(model_for_flops, input_len)
    row["MACs_G"] = macs_g
    row["FLOPs_G"] = flops_g
    if macs_g is not None:
        print(f"  MACs  (1s in) : {macs_g:.3f} G")
        print(f"  FLOPs (1s in) : {flops_g:.3f} G")
    else:
        print("  MACs/FLOPs    : skipped (see warning above)")

    # --- RTF on GPU ---
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        model_gpu = model_cls()
        rtf_gpu, mean_t, std_t, peak_mem = measure_rtf(model_gpu, input_len, "cuda")
        row["RTF_GPU"] = rtf_gpu
        row["GPU_time_ms"] = mean_t * 1000
        row["GPU_peak_mem_MB"] = peak_mem
        print(f"  RTF (GPU)     : {rtf_gpu:.5f}  ({mean_t*1000:.2f} ± {std_t*1000:.2f} ms per {REFERENCE_SECONDS:.1f}s input)")
        print(f"  Peak GPU mem  : {peak_mem:.1f} MB")
        del model_gpu
        torch.cuda.empty_cache()
    else:
        row["RTF_GPU"] = None
        print("  RTF (GPU)     : skipped (no CUDA device visible)")

    # --- RTF on CPU ---
    model_cpu = model_cls()
    rtf_cpu, mean_t_cpu, std_t_cpu, _ = measure_rtf(model_cpu, input_len, "cpu")
    row["RTF_CPU"] = rtf_cpu
    row["CPU_time_ms"] = mean_t_cpu * 1000
    print(f"  RTF (CPU)     : {rtf_cpu:.5f}  ({mean_t_cpu*1000:.2f} ± {std_t_cpu*1000:.2f} ms per {REFERENCE_SECONDS:.1f}s input)")

    return row


def print_summary_table(rows):
    print("\n" + "=" * 100)
    print("SUMMARY  (reference input = {:.1f}s of 16kHz audio, batch=1)".format(REFERENCE_SECONDS))
    print("=" * 100)
    header = f"{'Model':<18}{'Params(M)':>11}{'MACs(G)':>10}{'FLOPs(G)':>10}{'RTF_GPU':>10}{'RTF_CPU':>10}{'GPU_mem(MB)':>13}"
    print(header)
    print("-" * len(header))
    for r in rows:
        macs = f"{r['MACs_G']:.2f}" if r.get("MACs_G") is not None else "n/a"
        flops = f"{r['FLOPs_G']:.2f}" if r.get("FLOPs_G") is not None else "n/a"
        rtf_gpu = f"{r['RTF_GPU']:.4f}" if r.get("RTF_GPU") is not None else "n/a"
        rtf_cpu = f"{r['RTF_CPU']:.4f}" if r.get("RTF_CPU") is not None else "n/a"
        gpu_mem = f"{r['GPU_peak_mem_MB']:.1f}" if r.get("GPU_peak_mem_MB") is not None else "n/a"
        print(f"{r['model']:<18}{r['params_M']:>11.3f}{macs:>10}{flops:>10}{rtf_gpu:>10}{rtf_cpu:>10}{gpu_mem:>13}")
    print("=" * 100)
    print("RTF < 1.0 means faster than real time (e.g. RTF=0.05 processes 1s of audio in 50ms).")


def save_csv(rows, path="results_efficiency.csv"):
    import csv
    fieldnames = ["model", "params_M", "MACs_G", "FLOPs_G", "RTF_GPU", "GPU_time_ms",
                  "GPU_peak_mem_MB", "RTF_CPU", "CPU_time_ms"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"\nSaved -> {path}")


if __name__ == "__main__":

    # ---- register every model you want in the paper's table here ----
    MODELS_TO_PROFILE = [
        (ConvTasNet,    "ConvTasNet"),
        (DCCRN,         "DCCRN"),
        (MPSENet,       "MPSENet"),
        (Demucs,        "Demucs"),
        (XLSTM_SENet, "XLSTM_SENet"),
        # (NGDM_LSTM,   "NGDM_LSTM"),
        (NGDM_SE,     "NGDM_SE"),
    ]

    all_rows = []
    for model_cls, name in MODELS_TO_PROFILE:
        try:
            row = profile_model(model_cls, name)
            all_rows.append(row)
        except Exception as e:
            print(f"[error] failed to profile {name}: {e}")

    print_summary_table(all_rows)
    save_csv(all_rows, "results_efficiency.csv")