import torch
import numpy as np
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.data import random_split

# Modular imports
from src.models.architecture import VasculatureDualHeadModel
from src.data.dataset import HuBMAPDataset, get_transforms, load_annotations

DATA_DIR = "/kaggle/input/competitions/hubmap-hacking-the-human-vasculature"

def visualize_model_predictions(model, dataset, device, num_samples=3):
    """Samples images, runs inference, and plots predicted masks and bounding boxes."""
    model.eval()
    
    # Standard ImageNet normalization stats
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    for idx in indices:
        img_tensor, true_bboxes, _, true_mask = dataset[idx]
        img_input = img_tensor.unsqueeze(0).to(device)
        
        with torch.no_grad():
            box_preds, mask_preds = model(img_input)
            
        pred_mask = torch.sigmoid(mask_preds[0, 0]).cpu().numpy()
        pred_mask_binary = pred_mask > 0.5
        
        img_h, img_w = pred_mask.shape
        cx = torch.sigmoid(box_preds[0, 0]).item()
        cy = torch.sigmoid(box_preds[0, 1]).item()
        w = torch.exp(box_preds[0, 2].clamp(-5, 5)).item()
        h = torch.exp(box_preds[0, 3].clamp(-5, 5)).item()
        
        x_min = (cx - w / 2.0) * img_w
        y_min = (cy - h / 2.0) * img_h
        box_width = w * img_w
        box_height = h * img_h
        
        img_display = img_tensor.cpu().numpy()
        img_display = (img_display * std) + mean
        img_display = np.clip(img_display, 0, 1).transpose(1, 2, 0)
        true_mask_display = true_mask[0].cpu().numpy()
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        axes[0].imshow(img_display)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        axes[1].imshow(img_display)
        axes[1].imshow(true_mask_display, cmap='jet', alpha=0.4)
        for true_box in true_bboxes:
            tb_x_min, tb_y_min, tb_x_max, tb_y_max = true_box
            rect = patches.Rectangle((tb_x_min, tb_y_min), tb_x_max - tb_x_min, tb_y_max - tb_y_min, 
                                     linewidth=2, edgecolor='green', facecolor='none')
            axes[1].add_patch(rect)
        axes[1].set_title("Ground Truth (Green Box)")
        axes[1].axis('off')
        
        axes[2].imshow(img_display)
        axes[2].imshow(pred_mask_binary, cmap='jet', alpha=0.4)
        pred_rect = patches.Rectangle((x_min, y_min), box_width, box_height, 
                                      linewidth=2, edgecolor='red', facecolor='none')
        axes[2].add_patch(pred_rect)
        axes[2].set_title("Model Prediction (Red Box)")
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Setup Data for testing
    df = pd.read_csv(f"{DATA_DIR}/tile_meta.csv")
    image_paths = glob.glob(f"{DATA_DIR}/**/*.tif", recursive=True)
    path_dict = {os.path.splitext(os.path.basename(p))[0]: p for p in image_paths}
    df['image_path'] = df['id'].map(path_dict)
    df = df.dropna(subset=['image_path']).reset_index(drop=True)
    
    annotations_dict = load_annotations(f"{DATA_DIR}/polygons.jsonl")
    full_dataset = HuBMAPDataset(df, annotations_dict, transforms=get_transforms())
    
    # 2. Load Model & Weights
    inference_model = VasculatureDualHeadModel().to(device)
    weights_path = "weights/convnext_hubmap_best_weights.pth"
    
    print(f"Loading weights from {weights_path}...")
    checkpoint = torch.load(weights_path, map_location=device)
    inference_model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Weights loaded successfully from Epoch {checkpoint['epoch']}!")
    
    # 3. Visualize
    visualize_model_predictions(inference_model, full_dataset, device, num_samples=3)
