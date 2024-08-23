import torch
import os
from cnn_vae.model import CVAE
from cnn_vae.utils import generate_images
from config import inference_dir, latent_dim, checkpoint_dir, num_images

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CVAE(latent_dim).to(device)
    
    checkpoint_path = checkpoint_dir+"/best_model.pth" 
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    
    os.makedirs(inference_dir, exist_ok=True)
    generate_images(model, num_images=num_images, latent_dim=latent_dim, device=device, output_dir=inference_dir)
    
    print(f"Images saved at {inference_dir}")

if __name__ == "__main__":
    main()