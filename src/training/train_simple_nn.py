import torch
import torch.nn as nn
from src.models.simple_nn import SimpleNN

# Dummy dataset
X = torch.randn(100, 10)
y = torch.randint(0, 2, (100,))

model = SimpleNN(input_dim=10, hidden_dim=32, output_dim=2)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(20):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
