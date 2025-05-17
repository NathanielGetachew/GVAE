import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from vae_model import VAE

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained VAE model
def load_trained_vae(model_path, input_dim=28*28, hidden_dim=360, latent_dim=20):
    model = VAE(input_dim, hidden_dim, latent_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    return model

# Load dataset
def load_data(batch_size):
    testx = np.load("data/testX.npy").astype(np.float32)
    test_loader = DataLoader(TensorDataset(torch.tensor(testx)), batch_size=batch_size, shuffle=False)
    return test_loader]

def mask_input(data):
    batch_size, num_pixels = data.shape  # (batch, 28*28)
    img_size = int(torch.sqrt(torch.tensor(num_pixels)))  # Ensure it's 28

    # Reshape into (batch_size, 28, 28)
    data = data.view(batch_size, img_size, img_size)

    # Create a copy of the data before masking (to keep original intact)
    masked_data = data.clone()

    # Mask the left half (first 14 columns)
    masked_data[:, :, : img_size // 2] = 0

    return masked_data.view(batch_size, -1)  # Reshape back to (batch, 28*28)


# Compute Masked Mean Squared Error (MMSE)
def compute_mmse(model, data_loader):
    total_mmse = 0
    num_samples = 0

    with torch.no_grad():
        for batch in data_loader:
            batch = batch[0].view(batch[0].size(0), -1).to(device)
            masked_batch = mask_input(batch)

            # Reconstruct using VAE
            reconstructed, _, _ = model(masked_batch)

            # Compute MSE only for masked pixels (left half)
            mse_loss = torch.nn.functional.mse_loss(
                reconstructed[:, : 14 * 28], batch[:, : 14 * 28], reduction="sum"
            )

            total_mmse += mse_loss.item()
            num_samples += batch.size(0)

    return total_mmse / num_samples

# Visualize original vs masked images
def visualize_masking(original, masked, num_samples=5):
    original = original.view(-1, 28, 28).cpu().numpy()
    masked = masked.view(-1, 28, 28).cpu().numpy()

    fig, axes = plt.subplots(2, num_samples, figsize=(10, 4))
    for i in range(num_samples):
        axes[0, i].imshow(original[i], cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(masked[i], cmap="gray")
        axes[1, i].axis("off")

    axes[0, 0].set_title("Original")
    axes[1, 0].set_title("Masked (Left Half)")

    plt.savefig("masked_visual_fixed1.png")
    print("Saved masked visualization as masked_visual_fixed.png")

if __name__ == "__main__":
    vae = load_trained_vae("models/vae.pth")
    test_loader = load_data(batch_size=10)

    for batch in test_loader:
        batch = batch[0].view(batch[0].size(0), -1).to(device)
        masked_batch = mask_input(batch)

        # Visualize the updated masking
        visualize_masking(batch, masked_batch)
        break

    mmse_value = compute_mmse(vae, test_loader)
    print(f"Masked Mean Squared Error (MMSE): {mmse_value:.6f}")
