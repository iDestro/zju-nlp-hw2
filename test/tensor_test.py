import torch

x = torch.Tensor([1, 2, 3])
y = torch.Tensor([1, 2, 3])
z = torch.Tensor([1, 2, 3])

m = [x.numpy() for i in [x, y, z]]

tensor = torch.tensor(m)
print(tensor)