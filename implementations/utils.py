import pydash
import torch

import numpy as np

def to_torch_batch(batch, device, is_episodic):
    '''Mutate a batch (dict) to make its values from numpy into PyTorch tensor'''
    for key in batch:
        if is_episodic:  # for episodic format
            batch[key] = np.concatenate(batch[key])
        elif pydash.is_list(batch[key]):
            batch[key] = np.array(batch[key])
        batch[key] = torch.from_numpy(batch[key].astype(np.float32)).to(device)
    return batch
