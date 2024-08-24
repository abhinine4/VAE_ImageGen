import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
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
    
    val_size = int(len(dataset) * validation_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    model = CVAE(latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    train(model, train_loader, val_loader, optimizer, scheduler, device, num_epochs, checkpoint_dir, loss_dir)
        
    os.makedirs(output_dir, exist_ok=True)
    generate_images(model, num_images=64, latent_dim=latent_dim, device=device, output_dir=output_dir)

    os.makedirs(latent_dir, exist_ok=True)
    visualize_latent_space(model, train_loader, device, latent_dim, latent_dir)
    
    print(f"Generated images saved to {output_dir}")

if __name__ == "__main__":
    main()