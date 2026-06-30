"""
scripts/train_unet.py

Membuat Custom Dataset PyTorch dengan teknik patch-based (256x256),
membangun arsitektur model U-Net sederhana, dan menyiapkan fungsi training loop.
"""

import os
import sys
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import rasterio
from pathlib import Path

# Setup direktori data
PROCESSED_SENTINEL_DIR = os.path.join("data", "processed", "sentinel2")
LABEL_DIR = os.path.join("data", "processed", "labels")

# --- 1. CUSTOM DATASET PYTORCH (PATCH-BASED) ---
class PeatFireDataset(Dataset):
    def __init__(self, nbr_files, patch_size=256):
        self.patch_size = patch_size
        self.samples = []
        
        # Iterasi setiap citra besar untuk mendaftarkan koordinat ubin (patches)
        for nbr_path in nbr_files:
            scene_name = os.path.basename(nbr_path)
            mask_path = os.path.join(LABEL_DIR, scene_name.replace("_NBR.tif", "_MASK.tif"))
            
            if not os.path.exists(mask_path):
                continue
                
            with rasterio.open(nbr_path) as src:
                height, width = src.shape
                
            # Bagi koordinat gambar besar menjadi grid berukuran patch_size
            for y in range(0, height - patch_size + 1, patch_size):
                for x in range(0, width - patch_size + 1, patch_size):
                    self.samples.append({
                        "nbr_path": nbr_path,
                        "mask_path": mask_path,
                        "x": x,
                        "y": y
                    })
                    
        print(f"Dataset berhasil diinisialisasi dengan total {len(self.samples)} ubin berukuran {patch_size}x{patch_size}.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        x, y = sample["x"], sample["y"]
        p_size = self.patch_size
        
        # Gunakan jendela baca (Window) rasterio agar tidak meload seluruh gambar ke RAM
        window = rasterio.windows.Window(x, y, p_size, p_size)
        
        with rasterio.open(sample["nbr_path"]) as src_nbr:
            nbr_patch = src_nbr.read(1, window=window).astype(np.float32)
            # Normalisasi nilai NBR dari rentang [-1, 1] menjadi [0, 1] agar gradien training stabil
            nbr_patch = (nbr_patch + 1.0) / 2.0 
            
        with rasterio.open(sample["mask_path"]) as src_mask:
            mask_patch = src_mask.read(1, window=window).astype(np.float32)
            
        # Format tensor PyTorch: [Channel, Height, Width]
        x_tensor = torch.from_numpy(nbr_patch).unsqueeze(0) # Shape: [1, 256, 256]
        y_tensor = torch.from_numpy(mask_patch).unsqueeze(0) # Shape: [1, 256, 256]
        
        return x_tensor, y_tensor


# --- 2. ARSITEKTUR REKAYASA MODEL U-NET SEDERHANA ---
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class SimpleUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        # Encoder (Downsampling)
        self.inc = DoubleConv(in_channels, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        
        # Decoder (Upsampling)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(128, 64) # 64 (dari up) + 64 (dari skip connection encoder)
        
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(64, 32) # 32 + 32
        
        self.outc = nn.Conv2d(32, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Jalur Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        
        # Jalur Decoder dengan Skip Connections
        u1 = self.up1(x3)
        u1 = torch.cat([u1, x2], dim=1)
        u1 = self.conv_up1(u1)
        
        u2 = self.up2(u1)
        u2 = torch.cat([u2, x1], dim=1)
        u2 = self.conv_up2(u2)
        
        logits = self.outc(u2)
        return self.sigmoid(logits)


# --- 3. PIPELINE UTAMA TRAINING ---
# --- 3. PIPELINE UTAMA TRAINING ---
def main():
    # Cari semua citra NBR hasil preprocess
    nbr_files = glob.glob(os.path.join(PROCESSED_SENTINEL_DIR, "*_NBR.tif"))
    if not nbr_files:
        print("Tidak ada data NBR untuk training. Jalankan skrip preprocess lebih dulu.")
        return
        
    # Inisialisasi Dataset dan Loader
    dataset = PeatFireDataset(nbr_files, patch_size=256)
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)
    
    # Inisialisasi Model, Loss Function, dan Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleUNet(in_channels=1, out_channels=1).to(device)
    
    # BCELoss cocok untuk segmentasi biner (Kebakaran vs Aman)
    criterion = nn.BCELoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    print(f"Memulai uji coba training loop menggunakan perangkat: {device}\n")
    
    # ==================== BAGIAN YANG DIUBAH ====================
    # Jalankan Full Training (Contoh: 5 Epoch)
    NUM_EPOCHS = 5
    best_loss = float('inf')
    os.makedirs("models", exist_ok=True)
    
    print(f"Memulai FULL TRAINING selama {NUM_EPOCHS} Epoch...\n")
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Backward pass & Optimasi
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Cetak perkembangan setiap 100 batch agar log terminal rapi
            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | Batch [{batch_idx}/{len(train_loader)}] | Loss: {loss.item():.4f}")
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"==> Epoch {epoch+1} Selesai. Rata-rata Loss: {avg_epoch_loss:.4f}")
        
        # Simpan checkpoint model terbaik jika loss-nya lebih kecil dari epoch sebelumnya
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            checkpoint_path = os.path.join("models", "best_unet_peatfire.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"[*] Model terbaik disimpan dengan loss: {best_loss:.4f} -> {checkpoint_path}")
            
    print("\n[SUKSES] Proses pelatihan model selesai seluruhnya!")
    # ============================================================

if __name__ == "__main__":
    main()