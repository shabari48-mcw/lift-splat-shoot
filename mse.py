import torch
import torch.nn.functional as F


t1=torch.load("orgfinal.pt",weights_only=True)
t2=torch.load("modfinal.pt",weights_only=True)

mse = F.mse_loss(t1, t2)
print(mse.item())