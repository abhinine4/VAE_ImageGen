import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import *
from cnn_vae.model import CVAE
from cnn_vae.dataset import RanjeetFaceDataset
from cnn_vae.train import train
from cnn_vae.utils import generate_images, visualize_latent_space

def main():
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = RanjeetFaceDataset(data_dir=data_dir, transform=train_transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    model = CVAE(latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    train(model, dataloader, optimizer, scheduler, device, num_epochs, checkpoint_dir)
        
    os.makedirs(output_dir, exist_ok=True)
    generate_images(model, num_images=64, latent_dim=latent_dim, device=device, output_dir=output_dir)

    os.makedirs(latent_dir, exist_ok=True)
    visualize_latent_space(model, dataloader, device, latent_dim, latent_dir)
    
    print(f"Generated images saved to {output_dir}")

if __name__ == "__main__":
    main()