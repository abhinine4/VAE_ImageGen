import torch
import math
import os
from .utils import loss_function

def train(model, dataloader, optimizer, scheduler, device, num_epochs, checkpoint_dir):
    model.train()
    for epoch in range(num_epochs):
        train_loss = 0
        best_loss = math.inf
        best_model_state = None

        for batch_idx, data in enumerate(dataloader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        
        avg_loss = train_loss / len(dataloader.dataset)
        scheduler.step(avg_loss)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}')

        if (epoch + 1) % 50 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"cnn_vae_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_state = model.state_dict()
        
    best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
    torch.save(best_model_state, best_model_path)
    print(f"Best model saved with loss {best_loss:.4f}")