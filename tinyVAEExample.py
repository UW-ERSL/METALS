import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

# ===== YOUR EXACT CONFIG (3 samples, 3 attrs, latent=2) =====
DATASET_SIZE = 3
INPUT_SIZE = 3
HIDDEN_SIZE = 16     # Reduced to simplify, test fit
EPOCHS = 500         # Reduced epochs with restart
LR = 5e-5           # Slightly higher LR for faster convergence
BATCH_SIZE = 3       # Whole dataset per batch
LATENT_DIM = 2       # YOU REQUESTED!

print(f"🚀 YOUR AUTOENCODER: 3 samples × 3 attrs → latent=2")
print(f"   Hidden={HIDDEN_SIZE}, Epochs={EPOCHS}, LR={LR}")

# ===== STEP 1: YOUR 3x3 DATASET (NORMALIZED) =====
your_data = torch.tensor([
    [0.40, 0.20, 0.5],  # Material A
    [0.70, 0.60, 0.8],  # Material B
    [1.00, 1.00, 1.0]   # Material C
], dtype=torch.float32)

# Normalize to [0,1] based on min/max
data_min = torch.tensor([0.2, 0.2, 0.5])  # Min values per attribute
data_max = torch.tensor([1.0, 1.0, 1.0])  # Max values per attribute
your_data_normalized = (your_data - data_min) / (data_max - data_min)

dataset = TensorDataset(your_data_normalized)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ===== STEP 2: YOUR SIMPLE AUTOENCODER MODEL =====
class SimpleAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_dim):
        super().__init__()
        # Encoder
        self.enc1 = nn.Linear(input_size, hidden_size)
        self.enc2 = nn.Linear(hidden_size, latent_dim)
        # Decoder
        self.dec1 = nn.Linear(latent_dim, hidden_size)
        self.dec2 = nn.Linear(hidden_size, input_size)
    
    def encode(self, x):
        h1 = F.relu(self.enc1(x))
        return self.enc2(h1)  # No logvar, just latent
    
    def decode(self, z):
        h1 = F.relu(self.dec1(z))
        return self.dec2(h1)  # Linear output
    
    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z  # Return reconstruction and latent

# ===== STEP 3: LOSS (PURE RECONSTRUCTION) =====
def autoencoder_loss(recon_x, x):
    return F.mse_loss(recon_x, x, reduction='sum')

# ===== STEP 4: TRAINING (OPTIMIZED) =====
def train_autoencoder(model, dataloader, epochs, lr, data_min, data_max):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr/10)
    
    recon_losses = []
    
    print("\n🚀 Training your 3-sample Autoencoder...\n")
    print("Epoch | Loss | Recon | LR")
    print("-" * 25)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        total_recon = 0
        
        for batch in dataloader:
            data = batch[0] if isinstance(batch, list) else batch
            optimizer.zero_grad()
            recon, _ = model(data)  # Unpack only recon and latent
            loss = autoencoder_loss(recon, data)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            
            train_loss += loss.item()
            with torch.no_grad():
                r_loss = F.mse_loss(recon, data, reduction='sum').item()
                total_recon += r_loss
        
        scheduler.step()
        avg_loss = train_loss / DATASET_SIZE
        avg_recon = total_recon / DATASET_SIZE
        
        recon_losses.append(avg_recon)
        
        if epoch % 50 == 0 or epoch < 10:
            print(f"{epoch:5d} | {avg_loss:5.3f} | {avg_recon:5.3f} | {scheduler.get_last_lr()[0]:.1e}")
    
    # Check reconstructions (denormalize for comparison)
    with torch.no_grad():
        model.eval()
        recon_all, _ = model(your_data_normalized)
        recon_denorm = (recon_all * (data_max - data_min)) + data_min
        print("\nOriginal vs Reconstructed (Denormalized):")
        for i, (orig, recon) in enumerate(zip(your_data, recon_denorm)):
            error = torch.abs(orig - recon).mean().item() * 100
            attr_errors = torch.abs(orig - recon) * 100
            print(f"Sample {i+1}: Orig={orig.numpy()}, Recon={recon.numpy()}, Mean Error={error:.1f}%, "
                  f"Attr Errors={attr_errors.numpy()}%")
    
    return recon_losses

# ===== STEP 5: RUN YOUR AUTOENCODER! =====
model = SimpleAutoencoder(INPUT_SIZE, HIDDEN_SIZE, LATENT_DIM)
recon_losses = train_autoencoder(model, dataloader, EPOCHS, LR, data_min, data_max)

print("\n✅ YOUR AUTOENCODER IS TRAINED!")
print(f"Final Loss: {recon_losses[-1]:.3f}")

# ===== STEP 6: PLOT YOUR 2D LATENT SPACE! =====
with torch.no_grad():
    model.eval()
    z_samples = []
    for batch in dataloader:
        data = batch[0] if isinstance(batch, list) else batch
        z = model.encode(data)
        z_samples.extend(z.numpy())
    z_samples = np.array(z_samples)

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(recon_losses, 'r-', label='Reconstruction Loss')
plt.title('Training Loss')
plt.legend()
plt.yscale('log')

plt.subplot(1, 3, 2)
plt.scatter(z_samples[:, 0], z_samples[:, 1], c='red', s=200)
for i, txt in enumerate(['Material A', 'Material B', 'Material C']):
    plt.annotate(txt, (z_samples[i, 0], z_samples[i, 1]))
plt.title('YOUR 2D Latent Space')
plt.xlabel('Latent Dim 1')
plt.ylabel('Latent Dim 2')
plt.grid(True)

plt.subplot(1, 3, 3)
orig = your_data.numpy()
recon_all, _ = model(your_data_normalized)
recon_all_denorm = (recon_all * (data_max - data_min)) + data_min
recon_all = recon_all_denorm.detach().numpy()
for i in range(3):
    plt.plot([0, 1], [orig[i], recon_all[i]], 'o-', label=f'Attr {i+1}' if i == 0 else "_", linewidth=2)
plt.plot([0, 1], [0, 0], 'k--')  # Zero line for reference
plt.title('Original vs Reconstructed')
plt.legend(['Attr 1', 'Attr 2', 'Attr 3'])
plt.grid(True)

plt.tight_layout()
plt.show()

# ===== STEP 7: SAVE YOUR MODEL =====
torch.save(model.state_dict(), 'your_material_autoencoder_optimized.pth')
print("💾 Model saved as 'your_material_autoencoder_optimized.pth'")