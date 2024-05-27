import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class SimulationDataset(Dataset):
    def __init__(self, obs, intervention, regimens):
        self.obs = obs
        self.intervention = intervention
        self.regimens = regimens
        self.tensors = (obs, intervention, regimens)

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return self.obs[idx], ~self.intervention[idx], self.regimens[idx]
