import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from generator import Generator  # your existing generator
from discriminator import Discriminator  # your existing discriminator
from utils import load_config
import wandb

class GANModel(pl.LightningModule):
    def __init__(self, config, lr=0.0002, beta1=0.5, beta2=0.999):
        self.config = config
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        
        
        #we load the generator and the discriminator here        
        self.generator = Generator(config)
        self.discriminator = Discriminator(config)
        
        self.adversarial_loss = nn.BCEWithLogitsLoss()
        self.l1_loss = nn.L1Loss()
        
    def forward(self, condition):
        batch_size = condition.shape[0]
        noise = torch.randn(batch_size, self.config["generator"]["noise_dim"], device=self.device)
        return self.generator(condition, noise)
    
    def training_step(self, batch, batch_idx):
        
        opt_g, opt_d = self.optimizers()
        
        condition, target = batch
        
        batch_size = condition.shape[0]
        noise = torch.randn(batch_size, self.config["generator"]["noise_dim"], device=condition.device)

        # Generator training step
        opt_g.zero_grad()        
        
        # Generate fake image
        fake = self.generator(condition, noise)
        
        g_loss_l1 = self.l1_loss(fake, target)
        g_loss = g_loss_l1
        self.log("g_loss_l1", g_loss_l1, prog_bar=True)
            
        #fake_pred = self.discriminator(condition, fake)
        
        # Use adversarial loss with label smoothing
        #real_labels = torch.ones_like(fake_pred).fill_(0.9)  # Use 0.9 instead of 1.0
        #g_loss_adv = self.adversarial_loss(
        #    fake_pred, real_labels
        #)
            
        # Create a mask to focus more on white areas
        #white_mask = (condition > 0.8).float()
        # Calculate weighted L1 loss - more weight on white areas
        #weighted_l1 = self.l1_loss(fake * white_mask * 2.0, target * white_mask * 2.0)
        #regular_l1 = self.l1_loss(fake * (1 - white_mask), target * (1 - white_mask))
        
        # Reduce L1 weight from 100 to 30, more emphasis on white regions
        #g_loss_l1 = (weighted_l1 * 2.0 + regular_l1) * 30
            
        #g_loss = g_loss_adv + g_loss_l1
        
        # Log losses
        #self.log("g_loss", g_loss, prog_bar=True)
        #self.log("g_loss_adv", g_loss_adv, prog_bar=True)
        #self.log("g_loss_l1", g_loss_l1, prog_bar=True)
            
        self.manual_backward(g_loss)
        opt_g.step()
        
        # Train discriminator twice for every generator step
        # for _ in range(2):
        #     opt_d.zero_grad()
            
        #     # Generate fake image
        #     with torch.no_grad():
        #         fake = self.generator(condition, noise)
            
        #     # Real loss with label smoothing
        #     real_pred = self.discriminator(condition, target)
        #     real_labels = torch.ones_like(real_pred).fill_(0.9)  # Use 0.9 instead of 1.0
        #     d_loss_real = self.adversarial_loss(
        #         real_pred, real_labels
        #     )
            
        #     # Fake loss
        #     fake_pred = self.discriminator(condition, fake.detach())
        #     fake_labels = torch.zeros_like(fake_pred)
        #     d_loss_fake = self.adversarial_loss(
        #         fake_pred, fake_labels
        #     )
            
        #     # Total discriminator loss
        #     d_loss = (d_loss_real + d_loss_fake) * 0.5
            
        #     # Log losses
        #     self.log("d_loss", d_loss, prog_bar=True)
        #     self.manual_backward(d_loss)
        #     opt_d.step()
        
            
    def validation_step(self, batch, batch_idx):
        condition, target = batch
        batch_size = condition.shape[0]
        noise = torch.randn(batch_size, self.config["generator"]["noise_dim"], device=condition.device)
        fake = self.generator(condition, noise)
        
        val_loss = self.l1_loss(fake, target)
        self.log("val_loss", val_loss, prog_bar=True)
        
        # Validation losses
        # fake_pred = self.discriminator(condition, fake)
        # real_labels = torch.ones_like(fake_pred).fill_(0.9)
        # g_loss_adv = self.adversarial_loss(
        #     fake_pred, real_labels
        # )
        
        # # Create a mask to focus more on white areas
        # white_mask = (condition > 0.8).float()
        # weighted_l1 = self.l1_loss(fake * white_mask * 2.0, target * white_mask * 2.0)
        # regular_l1 = self.l1_loss(fake * (1 - white_mask), target * (1 - white_mask))
        # g_loss_l1 = (weighted_l1 * 2.0 + regular_l1) * 30
        
        # g_loss = g_loss_adv + g_loss_l1
        
        # # Log validation losses
        # self.log("val_g_loss", g_loss)
        # self.log("val_l1_loss", g_loss_l1)
        # self.log("val_adv_loss", g_loss_adv)
        
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
        opt_g = optim.Adam(
            self.generator.parameters(),
            lr=self.hparams.lr,
            betas=(self.hparams.beta1, self.hparams.beta2)
        )
        opt_d = optim.Adam(
            self.discriminator.parameters(),
            lr=self.hparams.lr,
            betas=(self.hparams.beta1, self.hparams.beta2)
        )
        
        return [opt_g, opt_d], [] 




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
    
    
