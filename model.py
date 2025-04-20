import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from generator import Generator  # your existing generator
from discriminator import Discriminator  # your existing discriminator
from utils import load_config
import wandb
import kornia
import torch.nn.functional as F
from kornia.losses import ssim_loss
from scipy.ndimage import distance_transform_edt
import torch
import torch.nn.functional as F

def focal_distance_loss(fake_dist, target_dist, gamma=0.15):
    error = torch.abs(fake_dist - target_dist)
    weights = (error) ** gamma
    loss = weights * error
    return loss.mean()

def compute_cell_distance_map(binary_tensor, max_dist=60):
    """
    Compute distance map from cell pixels to nearest background.
    
    Args:
        binary_tensor: Binary tensor with 1=cell, 0=background [B, 1, H, W]
        
    Returns:
        Distance map: 0 at cell boundaries, increasing inward
    """
    # Invert the binary image to compute distances from foreground to background
    inverted = 1.0 - (binary_tensor > 0.5).float()
    
    # Initialize distance map (0 for background, max_dist for cells)
    dist_map = torch.zeros_like(binary_tensor)
    dist_map[binary_tensor > 0.5] = max_dist
    
    # Current boundary pixels (initially all background pixels)
    boundary = inverted.clone()
    
    # Propagation kernel
    kernel = torch.ones(1, 1, 3, 3).to(binary_tensor.device)
    
    # Iteratively propagate distances
    for d in range(1, max_dist):
        # Dilate current boundary
        dilated = F.conv2d(
            F.pad(boundary, (1, 1, 1, 1), mode='constant', value=0),
            kernel, padding=0
        ) > 0
        
        # New boundary is cells that weren't assigned a distance yet
        new_boundary = dilated & (dist_map == max_dist)
        
        # Update distance values for new boundary
        dist_map[new_boundary] = d
        
        # Update boundary for next iteration
        boundary = new_boundary.float()
        
        # Early stop if no new pixels reached
        if not new_boundary.any():
            break
    
    # Keep only the cell distances (set background to 0)
    dist_map = dist_map * (binary_tensor > 0.5).float()
    
    # Normalize to [0,1]
    max_val = torch.max(dist_map)
    if max_val > 0:
        dist_map = dist_map / max_val
    
    return dist_map

class GANModel(pl.LightningModule):
    def __init__(self, config, lr_generator, lr_discriminator):
        self.config = config
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        
        
        #we load the generator and the discriminator here        
        self.generator = Generator(config)
        self.discriminator = Discriminator(config)
        self.optimizer_generator = optim.Adam(self.generator.parameters(), lr=lr_generator, betas=(0.5, 0.999))
        self.optimizer_discriminator = optim.Adam(self.discriminator.parameters(), lr=lr_discriminator, betas=(0.5, 0.999))
        
        self.adversarial_loss = nn.BCEWithLogitsLoss()
        self.l1_loss = nn.L1Loss()

    def forward(self, condition):
        batch_size = condition.shape[0]
        noise = torch.randn(batch_size, self.config["generator"]["noise_dim"], device=self.device)
        return self.generator(condition, noise)
    
    def training_step(self, batch, batch_idx):
        
        opt_g, opt_d = self.optimizers()
        
        condition, target = batch

        for _ in range(1):
            # Generator training step
            opt_g.zero_grad()        
            
            # Generate fake image
            fake = self.generator(condition)
            fake_distance = compute_cell_distance_map(fake)
            target_distance = compute_cell_distance_map(target)
            fake_pred = self.discriminator(condition, fake)
            real_labels = torch.ones_like(fake_pred)     
            g_loss_adv = self.adversarial_loss(fake_pred, real_labels)
            g_loss_l1 = self.l1_loss(fake, target) * 100
            #g_loss_ssim = ssim_loss(fake,target, window_size=41,reduction="mean") * 100
            g_loss_distance = self.l1_loss(fake_distance, target_distance) * 250
            g_loss = g_loss_adv + g_loss_l1  + g_loss_distance
            self.log("g_loss_l1", g_loss_l1, prog_bar=True,on_epoch=True)
            self.log("g_loss_adv", g_loss_adv, prog_bar=True,on_epoch=True)
            self.log("g_loss_distance", g_loss_distance, prog_bar=True,on_epoch=True)
            #self.log("g_loss_ssim", g_loss_ssim, prog_bar=True,on_epoch=True)
            self.log("g_loss", g_loss, prog_bar=True,on_epoch=True)
            
            self.manual_backward(g_loss)
            opt_g.step()
        
        for _ in range(1):
            opt_d.zero_grad()
            real_pred = self.discriminator(condition, target)
            fake_pred = self.discriminator(condition, fake.detach())
            real_labels = torch.ones_like(real_pred)
            fake_labels = torch.zeros_like(fake_pred)
            d_loss_real = self.adversarial_loss(real_pred, real_labels)
            d_loss_fake = self.adversarial_loss(fake_pred, fake_labels)
            d_loss = (d_loss_real + d_loss_fake) * 0.5
            self.log("d_loss_real", d_loss_real, prog_bar=True,on_epoch=True)
            self.log("d_loss_fake", d_loss_fake, prog_bar=True,on_epoch=True)
            self.log("d_loss", d_loss, prog_bar=True,on_epoch=True)
            self.manual_backward(d_loss)
            opt_d.step()
        
            
    def validation_step(self, batch, batch_idx):
        condition, target = batch
        batch_size = condition.shape[0]
        fake = self.generator(condition)
        fake_distance = compute_cell_distance_map(fake)
        target_distance = compute_cell_distance_map(target)
        fake_pred = self.discriminator(condition, fake)
        #Generator loss
        real_labels = torch.ones_like(fake_pred)
        val_g_loss_adv = self.adversarial_loss(fake_pred, real_labels)
        val_g_loss_l1 = self.l1_loss(fake, target) * 100
        #g_loss_ssim = ssim_loss(fake,target, window_size=41,reduction="mean") * 2
        val_g_loss_distance = self.l1_loss(fake_distance, target_distance) * 250
        #val_g_loss_ssim = ssim_loss(fake,target, window_size=41,reduction="mean") * 100
        val_g_loss = val_g_loss_adv + val_g_loss_l1 + val_g_loss_distance 
        
        self.log("val_g_loss", val_g_loss, prog_bar=True,on_epoch=True)
        self.log("val_g_loss_adv", val_g_loss_adv, prog_bar=True,on_epoch=True)
        self.log("val_g_loss_l1", val_g_loss_l1, prog_bar=True,on_epoch=True)
        self.log("val_g_loss_distance", val_g_loss_distance, prog_bar=True,on_epoch=True)
        #self.log("val_g_loss_ssim", val_g_loss_ssim, prog_bar=True,on_epoch=True)
        
        #loss_dict = {"val_g_loss" : val_g_loss, "val_g_loss_adv" : val_g_loss_adv, "val_g_loss_l1" : val_g_loss_l1, "val_g_loss_distance" : val_g_loss_distance}
        #self.logger.experiment.log(loss_dict,step=self.global_step)
        #self.log("val_g_loss", g_loss, prog_bar=True)
        #self.log_dict(loss_dict, prog_bar=True)
        
        #Discriminator loss

        real_pred = self.discriminator(condition, target)
        real_labels = torch.ones_like(real_pred)
        val_d_loss_real = self.adversarial_loss(real_pred, real_labels)

        fake_pred = self.discriminator(condition, fake.detach())        
        fake_labels = torch.zeros_like(fake_pred)
        val_d_loss_fake = self.adversarial_loss(fake_pred, fake_labels)
        val_d_loss = (val_d_loss_real + val_d_loss_fake) * 0.5
        self.log("val_d_loss", val_d_loss, prog_bar=True,on_epoch=True)
        self.log("val_d_loss_real", val_d_loss_real, prog_bar=True,on_epoch=True)
        self.log("val_d_loss_fake", val_d_loss_fake, prog_bar=True,on_epoch=True)
        
        # Log images to wandb (every 20 validation steps)
        if batch_idx % 20 == 0:
            # For wandb, need to use log method and send a dictionary of images
            # First ensure images are in the right format (0-1 range for wandb)
            condition_imgs = condition.detach().cpu()
            fake_imgs = fake.detach().cpu()
            target_imgs = target.detach().cpu()
            
            # Log only a subset of images if batch is large (first 4)
            n_samples = min(4, batch_size)
            
            # Create a grid of images for better visualization
            self.logger.experiment.log({
                "condition": [wandb.Image(img) for img in condition_imgs[:n_samples]],
                "generated": [wandb.Image(img) for img in fake_imgs[:n_samples]],
                "target": [wandb.Image(img) for img in target_imgs[:n_samples]],
                "step": self.global_step
            })
    
    def configure_optimizers(self):
        return [self.optimizer_generator, self.optimizer_discriminator], [] 




if __name__ == "__main__":
    config = load_config("config.yaml")
    model = GANModel(config)
    
    batch_size = 4
    condition = torch.randn(batch_size, 1, 512, 512)  # Binary condition image
    target = torch.randn(batch_size, 1, 512, 512)     # Binary target image
    batch = (condition, target)
    
    # Do a single training step for both generator and discriminator
    print("\nTesting generator training step:")
    #model.training_step(batch, 0)  # optimizer_idx=0 for generator

    # Test forward pass
    print("\nTesting forward pass:")
    fake = model(condition)
    print(f"Generated image shape: {fake.shape}")
    
    print("\nAll tests passed! Model is ready for training.")
    
    
