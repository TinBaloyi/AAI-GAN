import torch
import numpy as np
from tqdm import tqdm
from utils.loss_calc import plot_gan_losses


def train_gan(generator, discriminator, dataloader, epochs, device, opt_g, opt_d, criterion):
    generator.train()
    discriminator.train()


    g_losses, d_losses = [], []


    for epoch in range(epochs):
        g_loss_epoch, d_loss_epoch = 0.0, 0.0


    for real_samples in tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}', leave=False):
        real_samples = real_samples[0].to(device)
        batch_size = real_samples.size(0)


        # Train Discriminator
        opt_d.zero_grad()
        z = torch.randn(batch_size, generator.model[0].in_features, device=device)
        fake_samples = generator(z)


        real_labels = torch.ones((batch_size, 1), device=device)
        fake_labels = torch.zeros((batch_size, 1), device=device)


        real_loss = criterion(discriminator(real_samples), real_labels)
        fake_loss = criterion(discriminator(fake_samples.detach()), fake_labels)
        d_loss = (real_loss + fake_loss) / 2


        d_loss.backward()
        opt_d.step()


        # Train Generator
        opt_g.zero_grad()
        gen_labels = torch.ones((batch_size, 1), device=device)
        g_loss = criterion(discriminator(fake_samples), gen_labels)


        g_loss.backward()
        opt_g.step()


        g_loss_epoch += g_loss.item() * batch_size
        d_loss_epoch += d_loss.item() * batch_size


    g_losses.append(g_loss_epoch / len(dataloader.dataset))
    d_losses.append(d_loss_epoch / len(dataloader.dataset))


    print(f"Epoch [{epoch+1}/{epochs}] D_loss: {d_losses[-1]:.4f} G_loss: {g_losses[-1]:.4f}")


    plot_gan_losses(g_losses, d_losses)
    return generator, discriminator, g_losses, d_losses




def generate_synthetic_data(generator, num_samples, latent_dim, device):
    generator.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim, device=device)
        synthetic_data = generator(z).cpu().numpy()
    return synthetic_data