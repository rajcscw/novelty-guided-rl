import numpy as np 
import torch
from itertools import product

def get_index_array(indices: torch.Tensor):
    ix_indices_1 = []
    ix_indices_2 = []
    for ix_1 in range(indices.shape[0]):
        for ix_2 in indices[ix_1]:
            ix_indices_1.append(ix_1)
            ix_indices_2.append(int(ix_2))
    return torch.tensor(ix_indices_1), torch.tensor(ix_indices_2)

def apply_sparsity(encoded, sparsity_level):
    # apply the sparseness
    sorted_indices = torch.argsort(encoded, descending=True, dim=1)
    k_sparse = int(encoded.shape[1] * sparsity_level)
    top_indices = sorted_indices[:,:k_sparse]
    top_indices = get_index_array(top_indices) 
    masks = torch.zeros_like(encoded)
    masks[top_indices] = 1.0
    encoded = encoded * masks
    return encoded

input = np.random.randint(1, 10, size=(5, 6))
print("The input ")
print(input)

encoded = apply_sparsity(torch.from_numpy(input), sparsity_level=0.5)
print("Sparsed ")
print(encoded)
