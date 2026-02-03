import torch
from src.models.simple_cnn import SimpleCNN

model = SimpleCNN()
dummy_input = torch.randn(1, 3, 256, 256)

output = model(dummy_input)
print("Output shape:", output.shape)
