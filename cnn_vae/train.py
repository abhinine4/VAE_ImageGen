import torch
import math
import os
from .utils import loss_function, plot_loss_curve

def evaluate(model, data_loader, device):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            val_loss += loss.item()
    
    avg_val_loss = val_loss / len(data_loader.dataset)
    return avg_val_loss

def train(model, train_loader, val_loader, optimizer, scheduler, device, num_epochs, checkpoint_dir, loss_dir):
    best_val_loss = math.inf
    best_model_state = None
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        train_loss = 0
        model.train()
        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        
        avg_train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)
        val_loss = evaluate(model, val_loader, device)
        val_losses.append(val_loss)

        scheduler.step(val_loss)
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {val_loss:.4f}')

        if (epoch + 1) % 50 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"cnn_vae_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
        
    best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
    torch.save(best_model_state, best_model_path)
    print(f"Best model saved with validation loss {best_val_loss:.4f}")
    plot_loss_curve(train_losses, val_losses, save_dir=loss_dir, filename='loss_curve.png')
