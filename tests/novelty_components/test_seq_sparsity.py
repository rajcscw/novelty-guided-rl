import numpy as np 
import torch

def get_index_array(indices: torch.Tensor):
    ix_indices_1 = []
    ix_indices_2 = []
    for ix_1 in range(indices.shape[0]):
        for ix_2 in indices[ix_1]:
            ix_indices_1.append(ix_1)
            ix_indices_2.append(int(ix_2))
    return torch.tensor(ix_indices_1), torch.tensor(ix_indices_2)

def _apply_sparsity(encoded: torch.Tensor, sparsity_level=0.7) -> torch.Tensor:
    """Applies sparsity to the inner layer of hidden representation
    
    Arguments:
        encoded {torch.Tensor} -- hidden vector of RNN with shape (n_layers x batch_size x n_hidden)

    Returns:
        torch.Tensor -- returns output and hidden representation of RNN
    """
    n_layers = encoded.shape[0]
    inner_most_layer = encoded[n_layers-1]
    sorted_indices = torch.argsort(inner_most_layer, descending=True)
    k_sparse = int(inner_most_layer.shape[1] * sparsity_level)
    non_top_indices = sorted_indices[:,k_sparse:]
    non_top_indices = get_index_array(non_top_indices)
    masks = torch.ones_like(encoded)
    masks[n_layers-1, non_top_indices[0], non_top_indices[1]] = 0.0  
    encoded = encoded * masks
    return encoded


input = np.random.randint(1, 10, size=(2, 5, 4))
print("The input")
print(input)

print("Sparsed ones")
print(_apply_sparsity(torch.from_numpy(input), sparsity_level=0.5))