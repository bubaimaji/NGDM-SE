import torch
import torch.nn as nn


# --------------------------------------------
# Single STFT Loss
# --------------------------------------------
class STFTLoss(nn.Module):

    def __init__(self, fft_size, hop, win_length):
        super().__init__()

        window = torch.hann_window(win_length)
        self.register_buffer("window", window)

        self.fft_size = fft_size
        self.hop = hop
        self.win_length = win_length

    def forward(self, pred, target):

        pred_spec = torch.stft(
            pred,
            n_fft=self.fft_size,
            hop_length=self.hop,
            win_length=self.win_length,
            window=self.window,
            return_complex=True
        )

        target_spec = torch.stft(
            target,
            n_fft=self.fft_size,
            hop_length=self.hop,
            win_length=self.win_length,
            window=self.window,
            return_complex=True
        )

        pred_mag = pred_spec.abs()
        target_mag = target_spec.abs()

        # Spectral convergence
        sc = torch.norm(target_mag - pred_mag, p="fro") / (
            torch.norm(target_mag, p="fro") + 1e-8
        )

        # Log magnitude
        log_mag = torch.mean(
            torch.abs(
                torch.log(target_mag + 1e-7) -
                torch.log(pred_mag + 1e-7)
            )
        )

        return sc + log_mag


# --------------------------------------------
# Multi Resolution STFT
# --------------------------------------------
class MultiResolutionSTFTLoss(nn.Module):

    def __init__(self):
        super().__init__()

        self.losses = nn.ModuleList([

            STFTLoss(1024, 120, 600),
            STFTLoss(512, 50, 240),
            STFTLoss(256, 25, 120),

        ])

    def forward(self, pred, target):

        total = 0.0

        for loss in self.losses:
            total += loss(pred, target)

        return total / len(self.losses)
