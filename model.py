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


class GANModel(pl.LightningModule):
    def __init__(self, config, lr_generator, lr_discriminator):
        self.config = config
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        
        
        #we load the generator and the discriminator here        
        self.generator = Generator(config)
        self.discriminator = Discriminator(config)
        self.optimizer_generator = optim.Adam(self.generator.parameters(), lr=lr_generator)
        self.optimizer_discriminator = optim.Adam(self.discriminator.parameters(), lr=lr_discriminator)
        
        self.adversarial_loss = nn.BCEWithLogitsLoss()
        self.l1_loss = nn.L1Loss()

    def forward(self, condition):
        batch_size = condition.shape[0]
        noise = torch.randn(batch_size, self.config["generator"]["noise_dim"], device=self.device)
        return self.generator(condition, noise)
    
    def training_step(self, batch, batch_idx):
        
        opt_g, opt_d = self.optimizers()
        
        condition, target = batch

        for _ in range(3):
            # Generator training step
            opt_g.zero_grad()        
            
            # Generate fake image
            fake = self.generator(condition)
            fake_pred = self.discriminator(condition, fake)
            real_labels = torch.ones_like(fake_pred).fill_(0.98)            
            g_loss_adv = self.adversarial_loss(fake_pred, real_labels)
            g_loss_l1 = self.l1_loss(fake, target) * 70
            g_loss_ssim = ssim_loss(fake,target, window_size=41,reduction="mean") * 2
            g_loss = g_loss_adv + g_loss_l1 + g_loss_ssim
            
            self.log("g_loss_l1", g_loss_l1, prog_bar=True)
            self.log("g_loss_adv", g_loss_adv, prog_bar=True)
            self.log("g_loss_ssim", g_loss_ssim, prog_bar=True)
            self.log("g_loss", g_loss, prog_bar=True)
            
            self.manual_backward(g_loss)
            opt_g.step()
        
        for _ in range(1):
            opt_d.zero_grad()
            real_pred = self.discriminator(condition, target)
            fake_pred = self.discriminator(condition, fake.detach())
            real_labels = torch.ones_like(real_pred).fill_(0.98)
            fake_labels = torch.zeros_like(fake_pred).fill_(0.02)
            d_loss_real = self.adversarial_loss(real_pred, real_labels)
            d_loss_fake = self.adversarial_loss(fake_pred, fake_labels)
            d_loss = (d_loss_real + d_loss_fake) * 0.5
            self.log("d_loss_real", d_loss_real, prog_bar=True)
            self.log("d_loss_fake", d_loss_fake, prog_bar=True)
            self.log("d_loss", d_loss, prog_bar=True)
            self.manual_backward(d_loss)
            opt_d.step()
        
            
    def validation_step(self, batch, batch_idx):
        condition, target = batch
        batch_size = condition.shape[0]
        
        fake = self.generator(condition)
        fake_pred = self.discriminator(condition, fake)
        #Generator loss
        real_labels = torch.ones_like(fake_pred).fill_(0.98)
        g_loss_adv = self.adversarial_loss(fake_pred, real_labels)
        g_loss_l1 = self.l1_loss(fake, target) * 70
        g_loss_ssim = ssim_loss(fake,target, window_size=41,reduction="mean") * 2
        g_loss = g_loss_adv + g_loss_ssim + g_loss_l1
        self.log("g_val_loss", g_loss, prog_bar=True)
        
        #Discriminator loss

        real_pred = self.discriminator(condition, target)
        real_labels = torch.ones_like(real_pred).fill_(0.98)
        d_loss_real = self.adversarial_loss(real_pred, real_labels)

        fake_pred = self.discriminator(condition, fake.detach())        
        fake_labels = torch.zeros_like(fake_pred).fill_(0.02)
        d_loss_fake = self.adversarial_loss(fake_pred, fake_labels)
        d_loss = (d_loss_real + d_loss_fake) * 0.5
        self.log("d_val_loss", d_loss, prog_bar=True)
        
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
    
    
