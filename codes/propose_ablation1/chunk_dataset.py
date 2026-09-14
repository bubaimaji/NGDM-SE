import os
import random
import soundfile as sf
import torch
from torch.utils.data import Dataset

SR = 16000


class RandomChunkSpeechDataset(Dataset):
    """
    Research-grade random chunk dataset.

    ✔ aligned mix/clean cropping
    ✔ high GPU throughput
    ✔ avoids numpy->tensor overhead
    ✔ stable for SI-SDR training
    ✔ supports future dynamic chunking
    """

    def __init__(
        self,
        root_dir,
        chunk_sec=2,
        sample_rate=16000
    ):

        self.mix_dir = os.path.join(root_dir, "mix")
        self.clean_dir = os.path.join(root_dir, "clean")

        self.files = sorted(os.listdir(self.mix_dir))

        self.sr = sample_rate
        self.chunk_len = int(chunk_sec * sample_rate)

        print(f"\nLoaded {len(self.files)} files from {root_dir}")
        print(f"Chunk length: {chunk_sec}s ({self.chunk_len} samples)")

    # --------------------------------------------------
    def __len__(self):
        return len(self.files)

    # --------------------------------------------------
    def load_audio(self, path):
        audio, sr = sf.read(path)

        if sr != self.sr:
            raise RuntimeError(
                f"Sample rate mismatch in {path}"
            )

        # convert once → float32 tensor
        return torch.from_numpy(audio).float()

    # --------------------------------------------------
    def random_crop(self, mix, clean):
        """
        Always crop BOTH signals at the same location.
        """

        length = mix.shape[0]

        # ---- PAD if shorter than chunk ----
        if length < self.chunk_len:

            pad_size = self.chunk_len - length

            mix = torch.nn.functional.pad(
                mix, (0, pad_size)
            )

            clean = torch.nn.functional.pad(
                clean, (0, pad_size)
            )

            return mix, clean

        # ---- TRUE random crop ----
        start = random.randint(0, length - self.chunk_len)
        end = start + self.chunk_len

        return mix[start:end], clean[start:end]

    # --------------------------------------------------
    def __getitem__(self, idx):

        file = self.files[idx]

        mix_path = os.path.join(self.mix_dir, file)
        clean_path = os.path.join(self.clean_dir, file)

        mix = self.load_audio(mix_path)
        clean = self.load_audio(clean_path)

        mix, clean = self.random_crop(mix, clean)

        return mix, clean
