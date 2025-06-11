import os
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import pytorch_ssim

os.makedirs("checkpoints", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

ORIGINAL_FOLDER = "preprocessed/original"
DAMAGED_FOLDER = "preprocessed/damaged"
STRUCTURE_FOLDER = "preprocessed/structure_maps"

IMAGE_SIZE = 256
BATCH_SIZE = 8
EPOCHS = 100
LEARNING_RATE = 0.0002

# Data transformation
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Custom Dataset Loader
class TempleDataset(Dataset):
    def __init__(self, original_folder, structure_folder, transform=None):
        self.original_folder = original_folder
        self.structure_folder = structure_folder
        self.file_list = os.listdir(original_folder)
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        original_path = os.path.join(self.original_folder, file_name)
        structure_path = os.path.join(self.structure_folder, os.path.splitext(file_name)[0] + ".npy")

        original_img = cv2.imread(original_path)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        original_img = self.transform(original_img)

        input_tensor = np.load(structure_path)
        input_tensor = torch.tensor(input_tensor, dtype=torch.float32)

        return input_tensor, original_img

# Load dataset
dataset = TempleDataset(ORIGINAL_FOLDER, STRUCTURE_FOLDER, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

#Self-Attention 
class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.key = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.value = nn.Conv2d(in_dim, in_dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # batch size, channels, height, and width 
        B, C, H, W = x.size()
        #Reshape
        proj_query = self.query(x).view(B, -1, H * W).permute(0, 2, 1)
        proj_key = self.key(x).view(B, -1, H * W)
        attention = torch.bmm(proj_query, proj_key)
        attention = torch.softmax(attention, dim=-1)
        proj_value = self.value(x).view(B, -1, H * W)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1)).view(B, C, H, W)
        return self.gamma * out + x

class AdaIN(nn.Module):
    def __init__(self, epsilon=1e-5):
        super(AdaIN, self).__init__()
        self.epsilon = epsilon

    def forward(self, content, style):
        c_mean, c_std = content.mean([2, 3], keepdim=True), content.std([2, 3], keepdim=True)
        s_mean, s_std = style.mean([2, 3], keepdim=True), style.std([2, 3], keepdim=True)
        return s_std * (content - c_mean) / (c_std + self.epsilon) + s_mean


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def down_block(in_channels, out_channels, apply_batchnorm=True):
            layers = [nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False)]
            if apply_batchnorm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2))
            return nn.Sequential(*layers)

        def up_block(in_channels, out_channels, apply_dropout=False):
            layers = [
                nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ]
            if apply_dropout:
                layers.append(nn.Dropout(0.5))
            return nn.Sequential(*layers)

        self.attn = SelfAttention(512)
        self.adain = AdaIN()

        self.down1 = down_block(6, 64, apply_batchnorm=False)
        self.down2 = down_block(64, 128)
        self.down3 = down_block(128, 256)
        self.down4 = down_block(256, 512)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, 4, stride=2, padding=1, bias=False),
            nn.ReLU()
        )

        self.up1 = up_block(512, 512, apply_dropout=True)
        self.up2 = up_block(1024, 256)
        self.up3 = up_block(512, 128)
        self.up4 = up_block(256, 64)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, 3, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)

        b = self.bottleneck(d4)
        b = self.attn(b)
        b = self.adain(b, d4)

        u1 = self.up1(b)
        u1 = self.adain(u1, d4)
        u1 = torch.cat([u1, d4], dim=1)

        u2 = self.up2(u1)
        u2 = torch.cat([u2, d3], dim=1)
        u3 = self.up3(u2)
        u3 = torch.cat([u3, d2], dim=1)
        u4 = self.up4(u3)
        u4 = torch.cat([u4, d1], dim=1)

        return self.final(u4)


class GlobalDiscriminator(nn.Module):
    def __init__(self):
        super(GlobalDiscriminator, self).__init__()
        self.block1 = nn.Sequential(nn.Conv2d(9, 64, 4, stride=2, padding=1), nn.LeakyReLU(0.2))
        self.block2 = nn.Sequential(nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2))
        self.block3 = nn.Sequential(nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2))
        self.block4 = nn.Sequential(nn.Conv2d(256, 512, 4, stride=2, padding=1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2))
        self.final = nn.Sequential(nn.Conv2d(512, 1, 3, stride=1, padding=1), nn.Sigmoid())

    def forward(self, x, y, return_features=False):
        x = torch.cat([x, y], dim=1)
        f1 = self.block1(x)
        f2 = self.block2(f1)
        f3 = self.block3(f2)
        f4 = self.block4(f3)
        out = self.final(f4)
        return [f1, f2, f3, f4] if return_features else out


class PatchDiscriminator(GlobalDiscriminator):
    def forward(self, x, y):
        x_patch = x[:, :, 64:192, 64:192]
        y_patch = y[:, :, 64:192, 64:192]
        return super().forward(x_patch, y_patch, return_features=False)


# Initialize models
generator = Generator().to(device)
global_discriminator = GlobalDiscriminator().to(device)
patch_discriminator = PatchDiscriminator().to(device)

# Losses
criterion = nn.BCELoss()
l1_loss = nn.L1Loss()

optimizer_G = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
optimizer_D_global = optim.Adam(global_discriminator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
optimizer_D_patch = optim.Adam(patch_discriminator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))


if __name__ == "__main__":
# Training Loop
    best_loss = float("inf")
    for epoch in range(EPOCHS):
        print(f"\nStarting Epoch {epoch+1}/{EPOCHS}")

        start_time = time.time()
        for damaged_img, original_img in dataloader:
            damaged_img, original_img = damaged_img.to(device), original_img.to(device)
            fake_images = generator(damaged_img)

            # === Discriminators ===
            optimizer_D_global.zero_grad()
            real_out_g = global_discriminator(damaged_img, original_img)
            fake_out_g = global_discriminator(damaged_img, fake_images.detach())
            loss_D_global = 0.5 * (criterion(real_out_g, torch.ones_like(real_out_g)) +
                                criterion(fake_out_g, torch.zeros_like(fake_out_g)))
            loss_D_global.backward()
            optimizer_D_global.step()

            optimizer_D_patch.zero_grad()
            real_out_p = patch_discriminator(damaged_img, original_img)
            fake_out_p = patch_discriminator(damaged_img, fake_images.detach())
            loss_D_patch = 0.5 * (criterion(real_out_p, torch.ones_like(real_out_p)) +
                                criterion(fake_out_p, torch.zeros_like(fake_out_p)))
            loss_D_patch.backward()
            optimizer_D_patch.step()

            # === Generator ===
            optimizer_G.zero_grad()
            adv_loss = criterion(global_discriminator(damaged_img, fake_images), torch.ones_like(real_out_g)) + \
                    criterion(patch_discriminator(damaged_img, fake_images), torch.ones_like(real_out_p))

            l1 = l1_loss(fake_images, original_img) * 100

            feats_real = global_discriminator(damaged_img, original_img, return_features=True)
            feats_fake = global_discriminator(damaged_img, fake_images, return_features=True)
            perceptual = sum(l1_loss(f, r) for f, r in zip(feats_fake, feats_real)) * 10

            ssim_val = pytorch_ssim.ssim(fake_images, original_img.to(fake_images.device))
            ssim_loss = (1 - ssim_val) * 100

            loss_G = adv_loss + l1 + perceptual + ssim_loss
            loss_G.backward()
            optimizer_G.step()

        elapsed = time.time() - start_time
        print(f"Epoch [{epoch+1}/{EPOCHS}] | G: {loss_G.item():.4f} | Dg: {loss_D_global.item():.4f} | Dp: {loss_D_patch.item():.4f} | Time: {elapsed:.2f}s")

        if loss_G.item() < best_loss:
            best_loss = loss_G.item()
            torch.save(generator.state_dict(), "checkpoints/best_generator.pth")
            torch.save(global_discriminator.state_dict(), "checkpoints/best_global_discriminator.pth")
            torch.save(patch_discriminator.state_dict(), "checkpoints/best_patch_discriminator.pth")
            print(f"Best model saved at epoch {epoch+1} with Generator Loss: {best_loss:.4f}")

    print("Training Complete!")
