import torch
import torch.nn.functional as F
import torchvision.ops as ops

def compute_multi_task_loss(box_preds, mask_preds, target_bboxes, target_masks, device):
    alpha, gamma = 0.25, 2.0
    bce_loss = F.binary_cross_entropy_with_logits(mask_preds, target_masks, reduction='none')
    mask_probs = torch.sigmoid(mask_preds)
    p_t = mask_probs * target_masks + (1 - mask_probs) * (1 - target_masks)
    focal_loss = (alpha * target_masks + (1 - alpha) * (1 - target_masks)) * bce_loss * ((1 - p_t) ** gamma)
    focal_loss = focal_loss.mean()
    
    smooth = 1e-6
    tp = (mask_probs * target_masks).sum(dim=(2, 3))
    fp = (mask_probs * (1 - target_masks)).sum(dim=(2, 3))
    fn = ((1 - mask_probs) * target_masks).sum(dim=(2, 3))
    tversky_loss = 1.0 - ((tp + smooth) / (tp + 0.7 * fn + 0.3 * fp + smooth)).mean()
    
    mask_loss = (0.5 * focal_loss) + (0.5 * tversky_loss)
    
    box_loss = torch.tensor(0.0, device=device)
    batch_size = box_preds.shape[0]
    valid_box_count = 0
    
    cx, cy = torch.sigmoid(box_preds[:, 0]), torch.sigmoid(box_preds[:, 1])
    w, h = torch.exp(box_preds[:, 2].clamp(-5, 5)), torch.exp(box_preds[:, 3].clamp(-5, 5))
    img_h, img_w = mask_preds.shape[-2:]
    
    pred_coords = torch.stack([
        (cx - w / 2.0) * img_w, (cy - h / 2.0) * img_h,
        (cx + w / 2.0) * img_w, (cy + h / 2.0) * img_h
    ], dim=1)
    
    for i in range(batch_size):
        img_boxes = target_bboxes[i]
        if img_boxes.shape[0] > 0:
            areas = (img_boxes[:, 2] - img_boxes[:, 0]) * (img_boxes[:, 3] - img_boxes[:, 1])
            target_box = img_boxes[torch.argmax(areas)].unsqueeze(0).to(device)
            pred_box = pred_coords[i].unsqueeze(0)
            box_loss += ops.generalized_box_iou_loss(pred_box, target_box, reduction='sum')
            valid_box_count += 1
            
    if valid_box_count > 0:
        box_loss = box_loss / valid_box_count
        
    return box_loss + (2.0 * mask_loss), box_loss.item(), mask_loss.item()