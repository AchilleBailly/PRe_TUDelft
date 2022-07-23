import numpy as np
import torch

if __name__ == "__main__":
    x = torch.tensor([[1, 2, 3], [1, 2, 3]])
    y = x.repeat(1, 4, 1)
    print("test", y)
