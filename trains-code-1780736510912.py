import os
import glob
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

# Modular imports from your src directory
from src.data.dataset import HuBMAPDataset, get_transforms, collate_fn, load_annotations
from src.models.architecture import VasculatureDualHeadModel
from src.utils.losses import compute_multi_task_loss

DATA_DIR = "/kaggle/input/competitions/hubmap-hacking-the-human-vasculature"

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, total_box, total_mask = 0, 0, 0
    
    for imgs, bboxes, labels, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        bboxes = [b.to(device) for b in bboxes]
        
        optimizer.zero_grad()
        box_preds, mask_preds = model(imgs)
        
        loss, b_loss, m_loss = compute_multi_task_loss(box_preds, mask_preds, bboxes, masks, device)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_box += b_loss
        total_mask += m_loss
        
    return total_loss / len(loader), total_box / len(loader), total_mask / len(loader)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    df = pd.read_csv(f"{DATA_DIR}/tile_meta.csv")
    image_paths = glob.glob(f"{DATA_DIR}/**/*.tif", recursive=True)
    path_dict = {os.path.splitext(os.path.basename(p))[0]: p for p in image_paths}
    df['image_path'] = df['id'].map(path_dict)
    df = df.dropna(subset=['image_path']).reset_index(drop=True)
    
    annotations_dict = load_annotations(f"{DATA_DIR}/polygons.jsonl")
    full_dataset = HuBMAPDataset(df, annotations_dict, transforms=get_transforms())
    
    train_size = int(0.85 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_set, _ = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    
    train_loader = DataLoader(train_set, batch_size=4, shuffle=True, collate_fn=collate_fn, num_workers=2, pin_memory=True, drop_last=True)
    
    model = VasculatureDualHeadModel().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-2)
    
    NUM_EPOCHS = 15
    best_loss = float('inf')

    print(f"🚀 Commencing full training on {device} for {NUM_EPOCHS} epochs...")

    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        loss, b_loss, m_loss = train_one_epoch(model, train_loader, optimizer, device)
        print(f"[TRAIN] Total Loss: {loss:.4f} | Box: {b_loss:.4f} | Mask: {m_loss:.4f}")
        
        if loss < best_loss:
            best_loss = loss
            save_path = "weights/convnext_hubmap_best_weights.pth"
            os.makedirs("weights", exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, save_path)
            print(f"💾 Loss improved! New best model saved to {save_path}")