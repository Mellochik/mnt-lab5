import torch.nn as nn


class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=2, stride=2),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class SequenceAutoencoder(nn.Module):
    def __init__(self, input_size=64, hidden_size=128, latent_size=32, num_layers=1):
        super(SequenceAutoencoder, self).__init__()
        
        self.encoder = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.enc_fc = nn.Linear(hidden_size, latent_size)
        self.decoder = nn.GRU(latent_size, hidden_size, num_layers, batch_first=True)
        self.dec_fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        _, h = self.encoder(x)
        z = self.enc_fc(h[-1])
        z = z.unsqueeze(1).repeat(1, x.size(1), 1)
        out, _ = self.decoder(z)
        out = self.dec_fc(out)
        return out

    def encode(self, x):
        _, h = self.encoder(x)
        z = self.enc_fc(h[-1])
        return z


class MLPAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MLPAutoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, input_size),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encode(self, x):
        return self.encoder(x)
