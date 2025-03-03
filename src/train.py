import torch
import torch.nn as nn
import torch.optim as optim
from dataset_loader import get_dataloader
from model import Generator, Discriminator
from utils import save_model
import os

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

latent_dim = 50
img_channels = 3
epochs = 5000
batch_size = 4
device = "cpu"

dataset_path = "datasets/my_dataset/"
assert os.path.exists(dataset_path), "Dataset folder not found!"

dataloader = get_dataloader(dataset_path, batch_size=batch_size)

generator = Generator(latent_dim, img_channels).to(device)
discriminator = Discriminator(img_channels).to(device)

criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

for epoch in range(epochs):
    for real_imgs in dataloader:
        real_imgs = real_imgs.to(device)
        batch_size = real_imgs.size(0)

        z = torch.randn(batch_size, latent_dim).to(device)
        fake_imgs = generator(z)

        optimizer_D.zero_grad()
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)
        real_loss = criterion(discriminator(real_imgs), real_labels)
        fake_loss = criterion(discriminator(fake_imgs.detach()), fake_labels)
        d_loss = real_loss + fake_loss
        d_loss.backward()
        optimizer_D.step()

        optimizer_G.zero_grad()
        g_loss = criterion(discriminator(fake_imgs), real_labels)
        g_loss.backward()
        optimizer_G.step()

    if epoch % 100 == 0:
        save_model(generator, "./models/stylegan2_generator.pth")
        save_model(discriminator, "./models/stylegan2_discriminator.pth")
        print(f"Epoch {epoch+1}/{epochs} | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

