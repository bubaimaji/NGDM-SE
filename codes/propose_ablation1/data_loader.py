# dataloader.py
import os
import torch
import librosa
from torch.utils.data import Dataset

SR = 16000

class SpeechEnhDataset(Dataset):
    def __init__(self, root):
        self.mix = sorted(os.listdir(os.path.join(root,"mix")))
        self.root = root

    def __len__(self):
        return len(self.mix)

    def __getitem__(self, idx):
        name = self.mix[idx]
        noisy,_ = librosa.load(os.path.join(self.root,"mix",name), sr=SR)
        clean,_ = librosa.load(os.path.join(self.root,"clean",name), sr=SR)
        #return torch.tensor(noisy), torch.tensor(clean)
        return torch.from_numpy(noisy).float(), torch.from_numpy(clean).float()


        #return torch.tensor(noisy), torch.tensor(clean), name
    
    