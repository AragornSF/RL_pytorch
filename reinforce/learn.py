import torch

x = torch.Tensor([[2, 3, 4],
                 [4, 5, 6]])

print(x)
y = torch.Tensor([[2, 3, 4],
                 [4, 5, 6]])

print(x.mul(y))
print(x.shape)

print(x.mul(y).cuda())
