import torch
import matplotlib.pyplot as plt
from model import Generator

latent_dim = 50
img_channels = 3
device = "cpu"

generator = Generator(latent_dim, img_channels).to(device)
generator.load_state_dict(torch.load("./models/stylegan2_generator.pth", map_location=device))
generator.eval()

def generate_image():
    z = torch.randn(1, latent_dim).to(device)
    fake_img = generator(z).detach().cpu().squeeze(0).permute(1, 2, 0) * 0.5 + 0.5  # Rescale to [0,1]
    plt.imshow(fake_img)
    plt.axis("off")
    plt.show()

generate_image()
