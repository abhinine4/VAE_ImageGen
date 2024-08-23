import torch
import torch.nn as nn
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import os

def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


def generate_images(model, num_images, latent_dim, device, output_dir):
    model.eval()
    with torch.no_grad():
        latent_vectors = torch.randn(num_images, latent_dim).to(device)
        generated_images = model.decode(latent_vectors)
        generated_images = (generated_images + 1) / 2
        generated_images = generated_images.cpu()
        
        for i, img in enumerate(generated_images):
            save_image(img, f"{output_dir}/generated_image_{i+1}.png")
        
        grid_size = min(int(num_images**0.5), 8)
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(20, 20))
        for i, ax in enumerate(axes.flatten()):
            if i < num_images:
                ax.imshow(generated_images[i].permute(1, 2, 0))
                ax.axis('off')
            else:
                ax.axis('off')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/generated_grid.png", dpi=300)
        plt.close()