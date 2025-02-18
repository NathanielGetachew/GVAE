import torch
from torch import nn

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        
        # Encoder with 4 layers (manually defined)
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # Input to hidden layer 1
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # Hidden layer 1 to hidden layer 2
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)  # Output mean
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)  # Output log variance
        
        # Decoder
        self.fc4 = nn.Linear(latent_dim, hidden_dim)  # Latent space to hidden layer 1
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)  # Hidden layer 1 to hidden layer 2
        self.fc6 = nn.Linear(hidden_dim, input_dim)  # Hidden layer 2 to output
        
        self.relu = nn.ReLU()  # ReLU activation
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation for output normalization

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)  # Standard deviation (exp(log_var / 2))
        eps = torch.randn_like(std)  # Random noise with the same shape as std
        return mean + eps * std  # Reparameterization trick

    def forward(self, x):
        # Encode the input
        x = self.relu(self.fc1(x))  # Apply ReLU after first layer
        x = self.relu(self.fc2(x))  # Apply ReLU after second layer
        mean = self.fc_mean(x)  # Compute mean
        log_var = self.fc_logvar(x)  # Compute log variance
        
        # Reparameterize to get latent variable z
        z = self.reparameterize(mean, log_var)

        # Decode the latent variable
        x_reconstructed = self.relu(self.fc4(z))  # Apply ReLU after first decoder layer
        x_reconstructed = self.relu(self.fc5(x_reconstructed))  # Apply ReLU after second decoder layer
        x_reconstructed = self.sigmoid(self.fc6(x_reconstructed))  # Normalize output to [0, 1]

        return x_reconstructed, mean, log_var
