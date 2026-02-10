from model.DiT.dit import DiffusionTransformer
from model.autoencoder.autoencoder import DeepCompressionAutoEncoder
import torch
import torch.nn.functional as F
import torch.util.dataloder as Dataloader

class Pipeline:
    def __init__(self, autoencoder, dit, optimizer, max_timestamps=1000, *args, **kwds):
        # Initialize your pipeline components here

        self.autoencoder = autoencoder
        self.dit = dit
        self.optimizer = optimizer
        self.max_timesteps = max_timestamps

    def __call__(self, input):
        batch_size = latent.shape[0]
        latent = self.encode(input)

        noise = torch.randn_like(latent)

        t = torch.randint(0, self.max_timesteps, (batch_size,), device=self.device).long()

        noisy_latent = self.add_noise(latent, t, noise, self.max_timesteps)

        noise_pred = self.diffusion_process(noisy_latent, t)

        loss = F.mse_loss(noise_pred, noise)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def encode(self, input):
        return self.autoencoder.encode(input)

    def diffusion_process(self, input, t):
        return self.dit(input, t)

    def add_noise(self, input, t):
        return self.dit.add_noise(input, t)


def main():
    dataloader = Dataloader()
    autoencoder = DeepCompressionAutoEncoder()
    dit = DiffusionTransformer(in_channels=32 * 80, out_channels=32 * 80)
    optimizer = torch.optim.Adam(list(autoencoder.parameters()) + list(dit.parameters()), lr=1e-4)
    MAX_TIMESTAMPS = 100
    num_epochs = 20
    # Trai the model here
    pipeline = Pipeline(autoencoder, dit, optimizer, max_timestamps=MAX_TIMESTAMPS)

    for epoch in range(num_epochs):
        for batch in dataloader:
            input = batch['mel_spectrogram']
            loss = pipeline(input)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
    
    

if __name__ == "__main__":
    main()