import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from chunk_dataset import RandomChunkSpeechDataset
from data_loader import SpeechEnhDataset
from mrstft_loss import MultiResolutionSTFTLoss

from ngdm_xlstm import NGDM_SE

# =====================================================
# CUDA
# =====================================================

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

os.environ["CUDA_VISIBLE_DEVICES"] = "5"   # <-- CONFIRM free: nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv

DEVICE = torch.device("cuda:0")
print("GPU:", torch.cuda.get_device_name(0))


# =====================================================
# CONFIG
# =====================================================

SR = 16000
BATCH_SIZE = 8       # fast backbone (time-only mLSTM + cheap standard
                      # attention for freq-context) -- no gradient
                      # accumulation needed, same as the original 17.27dB run
EPOCHS = 60
LR = 2e-4
LOG_EVERY = 50

SANITY_CHECK_AFTER_EPOCH = 2
SANITY_CHECK_MIN_SISDR_DB = -5.0

DATA_ROOT = "/home/bubai-maji/speech_enhancement/data/mixtures/musan"
SAVE_DIR = "/home/bubai-maji/speech_enhancement/script/propose/checkpoint"

os.makedirs(SAVE_DIR, exist_ok=True)


# =====================================================
# SI-SDR
# =====================================================

def si_sdr_loss(est, ref, eps=1e-8):
    ref = ref - ref.mean(dim=-1, keepdim=True)
    est = est - est.mean(dim=-1, keepdim=True)
    alpha = torch.sum(est * ref, dim=-1, keepdim=True) / (torch.sum(ref**2, dim=-1, keepdim=True) + eps)
    proj = alpha * ref
    noise = est - proj
    ratio = (proj.pow(2).sum(dim=-1) + eps) / (noise.pow(2).sum(dim=-1) + eps)
    return -10 * torch.log10(ratio).mean()


def si_sdr_metric(est, ref, eps=1e-8):
    with torch.no_grad():
        return -si_sdr_loss(est, ref).item()


# =====================================================
# TRAIN  (sync-free loss tracking + progress logging)
# =====================================================

def train_epoch(model, loader, optimizer, scaler, mrstft, log_every=LOG_EVERY):

    model.train()
    total_loss = torch.zeros((), device=DEVICE)
    n_batches = len(loader)

    epoch_start = time.time()
    last_log_time = epoch_start

    for i, (noisy, clean) in enumerate(loader):

        noisy = noisy.to(DEVICE, non_blocking=True)
        clean = clean.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type="cuda"):
            enhanced = model(noisy)

            l1 = F.l1_loss(enhanced, clean)
            sisdr = si_sdr_loss(enhanced, clean)
            mr = mrstft(enhanced, clean)

            loss = 0.4*sisdr + 0.3*mr + 0.3*l1

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.detach()

        if (i + 1) % log_every == 0:
            now = time.time()
            steps_done = i + 1
            elapsed = now - epoch_start
            sec_per_step = (now - last_log_time) / log_every
            steps_per_sec = 1.0 / sec_per_step if sec_per_step > 0 else float("inf")
            remaining_steps = n_batches - steps_done
            eta_sec = remaining_steps * sec_per_step
            print(
                f"  step {steps_done}/{n_batches}"
                f" | loss {loss.item():.3f}"
                f" | {sec_per_step*1000:.0f} ms/step ({steps_per_sec:.2f} it/s)"
                f" | elapsed {elapsed/60:.1f} min"
                f" | ETA this epoch {eta_sec/60:.1f} min",
                flush=True,
            )
            last_log_time = now

    return (total_loss / n_batches).item()


# =====================================================
# VALIDATION -- real SI-SDR, not just L1
# =====================================================

@torch.no_grad()
def validate(model, loader):
    model.eval()

    total_l1 = torch.zeros((), device=DEVICE)
    total_sisdr_db = 0.0
    n_batches = 0

    for noisy, clean in loader:
        noisy = noisy.to(DEVICE)
        clean = clean.to(DEVICE)

        enhanced = model(noisy)

        total_l1 += F.l1_loss(enhanced, clean)
        total_sisdr_db += si_sdr_metric(enhanced, clean)
        n_batches += 1

    return {
        "l1": (total_l1 / n_batches).item(),
        "sisdr_db": total_sisdr_db / n_batches,
    }


# =====================================================
# MAIN
# =====================================================

class EarlyStopping:
    def __init__(self, patience=7, min_delta=1e-2):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("-inf")
        self.counter = 0
        self.stop = False

    def step(self, val_sisdr_db):
        if val_sisdr_db > self.best + self.min_delta:
            self.best = val_sisdr_db
            self.counter = 0
            return True
        else:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.stop = True
            return False


def main():

    print("\nLoading datasets...")

    train_ds = RandomChunkSpeechDataset(
        os.path.join(DATA_ROOT, "train"),
        chunk_sec=2.5
    )

    val_ds = SpeechEnhDataset(
        os.path.join(DATA_ROOT, "val")
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )

    val_dl = DataLoader(val_ds, batch_size=4)

    print("\nInitializing model...")

    # ---- SELECT VARIANT HERE ----
    model = NGDM_SE().to(DEVICE)   # full proposed model (default: all flags True)
    # model = NGDM_SE(use_noise_routing=False).to(DEVICE)      # ablation: w/o noise routing
    # model = NGDM_SE(use_dual_memory=False).to(DEVICE)        # ablation: w/o dual memory
    # model = NGDM_SE(use_attention_fusion=False).to(DEVICE)   # ablation: w/o attention fusion
    # model = NGDM_SE(use_freq_context=False).to(DEVICE)       # ablation: w/o frequency context

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {model.__class__.__name__} | Params: {n_params/1e6:.3f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)
    scaler = GradScaler("cuda")
    mrstft = MultiResolutionSTFTLoss().to(DEVICE)

    early_stopper = EarlyStopping(patience=7, min_delta=1e-2)

    print("\n========== START TRAINING ==========\n")

    for epoch in range(EPOCHS):

        epoch_t0 = time.time()

        train_loss = train_epoch(model, train_dl, optimizer, scaler, mrstft)
        val = validate(model, val_dl)
        scheduler.step(val["sisdr_db"])

        epoch_time_min = (time.time() - epoch_t0) / 60

        print(
            f"Epoch {epoch+1}/{EPOCHS}"
            f" | Train {train_loss:.3f}"
            f" | Val L1 {val['l1']:.4f}"
            f" | Val SI-SDR {val['sisdr_db']:.2f} dB"
            f" | Time {epoch_time_min:.1f} min",
            flush=True,
        )

        if epoch + 1 == SANITY_CHECK_AFTER_EPOCH:
            if val["sisdr_db"] < SANITY_CHECK_MIN_SISDR_DB:
                print(
                    f"\n*** WARNING: val SI-SDR ({val['sisdr_db']:.2f} dB) is still "
                    f"very low after epoch {SANITY_CHECK_AFTER_EPOCH}. Investigate before "
                    f"continuing to spend GPU time. ***\n",
                    flush=True,
                )
            else:
                print(f"  [sanity check OK: val SI-SDR {val['sisdr_db']:.2f} dB]\n", flush=True)

        improved = early_stopper.step(val["sisdr_db"])

        if improved:
            save_path = os.path.join(SAVE_DIR, f"best_{model.__class__.__name__}.pt")
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model (SI-SDR {val['sisdr_db']:.2f} dB) -> {save_path}")

        if early_stopper.stop:
            print("\nEarly stopping triggered.")
            break

    print("\n========== TRAINING COMPLETE ==========\n")
    print(f"Best validation SI-SDR: {early_stopper.best:.2f} dB")


if __name__ == "__main__":
    main()