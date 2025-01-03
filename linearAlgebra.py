import torch
x = torch.rand(5, 3)
print(x)

a = torch.tensor([[1.0],[2.0],[4.0],[8.0]])
b = torch.tensor([[1.0],[0.5],[0.25],[0.125]])
print(a-b)
torch.sigmoid(b)

c = torch.tensor([4, -4, 0, 2])
torch.relu(c)

a = torch.rand((3,4,2))
print(a)