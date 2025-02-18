import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.mixture import GaussianMixture
import numpy as np
from tqdm import tqdm
import logging
from vae_model import VAE

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define device globally
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
def load_data(batch_size):
    trainx = np.load("data/trainX.npy").astype(np.float32)  # Features
    trainy = np.load("data/trainY.npy")  # Labels for training
    testx = np.load("data/testX.npy").astype(np.float32)  # Features
    testy = np.load("data/testY.npy")  # Labels for testing

    train_loader = DataLoader(TensorDataset(torch.tensor(trainx), torch.tensor(trainy)), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.tensor(testx), torch.tensor(testy)), batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

# Rescale gradients
def rescale_gradients(model, max_norm=5.0):
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)

# Define VAE Loss
def vae_loss(reconstructed, original, mean, log_var):
    # Binary Cross Entropy Loss (BCE)
    bce_loss = nn.functional.binary_cross_entropy(reconstructed, original, reduction='sum')
    
    # KL Divergence
    kl_div = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp()) / original.size(0)  # Normalize by batch size
    
    # Combine BCE and KL divergence for VAE loss
    total_loss = bce_loss + kl_div
    return total_loss

# Fit GMM after training
def fit_gmm(features, n_components=10):
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
    gmm.fit(features)
    return gmm

# Monte Carlo estimation of log-likelihood
def monte_carlo_log_likelihood(gmm, vae, data_loader, n_samples=5000):
    gmm_samples, _ = gmm.sample(n_samples)
    z_samples = torch.tensor(gmm_samples, dtype=torch.float32).to(device)
    log_p_z = gmm.score_samples(gmm_samples)  # Log probability under GMM
    log_p_x_given_z = []

    # Decode z_samples to get p(x|z)
    with torch.no_grad():
        for i in range(0, n_samples, data_loader.batch_size):
            batch_z = z_samples[i:i + data_loader.batch_size]
            recon_x = vae.decode(batch_z)  # ✅ FIX: Call decode() explicitly
            log_p_x_given_z.extend(-torch.nn.functional.binary_cross_entropy(
                recon_x, recon_x, reduction='none'
            ).sum(dim=1).cpu().numpy())  # Compute reconstruction likelihood

    log_p_x_given_z = np.array(log_p_x_given_z)
    log_likelihood = np.mean(log_p_z + log_p_x_given_z)
    return log_likelihood


# Train VAE function
def train_vae():
    input_dim = 28 * 28  # MNIST
    hidden_dim = 360
    latent_dim = 20
    batch_size = 200
    num_epochs = 50
    learning_rate = 0.04
    lambda_l2 = 1e-4

    model = VAE(input_dim, hidden_dim, latent_dim).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    train_loader, test_loader = load_data(batch_size)
    
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch, _ in train_loader:
            batch = batch.view(batch.size(0), -1).to(device)
            
            optimizer.zero_grad()
            recon_x, mean, log_var = model(batch)
            loss = vae_loss(recon_x, batch, mean, log_var)
            
            # L2 Regularization
            l2_reg = sum(torch.sum(param**2) for param in model.parameters() if param.requires_grad)
            loss += lambda_l2 * l2_reg
            
            loss.backward()
            rescale_gradients(model)
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"Epoch [{epoch + 1}/{num_epochs}] | VAE Loss: {epoch_loss / len(train_loader.dataset):.4f}")
    
    # Switch to evaluation mode and extract latent features
    model.eval()
    all_latent_features = []
    logging.info("Extracting latent features...")
    
    with torch.no_grad():
        for batch, _ in tqdm(train_loader, desc="Processing Latent Vectors"):
            batch = batch.view(batch.size(0), -1).to(device)
            _, mean, _ = model(batch)
            all_latent_features.append(mean.cpu().numpy())
    
    all_latent_features = np.concatenate(all_latent_features, axis=0)
    gmm = fit_gmm(all_latent_features)
    
    # Calculate final log-likelihood using Monte Carlo method
    log_likelihood = monte_carlo_log_likelihood(gmm, model, train_loader)
    logging.info(f"Monte Carlo Log Likelihood: {log_likelihood:.4f}")
    print(f"Monte Carlo Log Likelihood: {log_likelihood:.4f}")
    
    # Save trained model
    torch.save(model.state_dict(), "models/vae.pth")
    return gmm, model

if __name__ == "__main__":
    gmm, model = train_vae()
