"""
CLIP Appearance Extractor for SAM3 Tracker

Simple batch extraction: One image + N masks → N CLIP embeddings.
No caching or smoothing — handled externally by Track class.
"""

import cv2
import numpy as np
import torch
from typing import List, Tuple, Optional
from PIL import Image
import torchvision.transforms as transforms
from config_loader import cfg

# ============================================================================
# CLIP IMPORT (with fallback)
# ============================================================================

try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("⚠️  CLIP not installed. Install with: pip install git+https://github.com/openai/CLIP.git")


# ============================================================================
# CLIP APPEARANCE EXTRACTOR
# ============================================================================

class CLIPAppearanceExtractor:
    """
    Simple CLIP extractor: One image + N masks → N embeddings.
    
    No internal caching or smoothing — these are handled by the Track class.
    """
    
    def __init__(
        self,
        model_name: str = None,
        device: str = None,
        input_size: int = None,
        crop_margin: float = None,
        mask_dilation: int = None,
        background_blur: int = None
    ):
        """
        Initialize CLIP model.
        
        Args:
            model_name: CLIP model name (e.g., "ViT-B/32").
            device: Device to run model on ('cuda' or 'cpu').
            input_size: CLIP input resolution.
            crop_margin: Padding margin around object (fraction of bbox).
            mask_dilation: Dilation kernel size for mask expansion.
            background_blur: Gaussian blur kernel for background suppression.
        """
        if not CLIP_AVAILABLE:
            raise ImportError(
                "CLIP is not installed. Install with:\n"
                "pip install git+https://github.com/openai/CLIP.git"
            )

        model_name = cfg.clip.model_name if model_name is None else model_name
        if device is None:
            device = cfg.clip.device
            if device == "cuda" and not torch.cuda.is_available():
                device = "cpu"
         
        self.device = device
        self.input_size = cfg.clip.input_size if input_size is None else input_size
        self.crop_margin = cfg.clip.crop_margin if crop_margin is None else crop_margin
        self.mask_dilation = cfg.clip.mask_dilation_kernel if mask_dilation is None else mask_dilation
        self.background_blur = cfg.clip.background_blur if background_blur is None else background_blur
        
        # Load CLIP model
        self.model, _ = clip.load(model_name, device=device)
        self.model.eval()
        
        # CLIP preprocessing transform (matches training)
        self.clip_transform = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.CenterCrop(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])
    
    def _dilate_mask(self, mask: np.ndarray) -> np.ndarray:
        """Dilate mask to include object boundaries."""
        if self.mask_dilation <= 0:
            return mask
        
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.mask_dilation, self.mask_dilation)
        )
        dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
        return dilated.astype(bool)
    
    def _crop_object(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Crop object from image with margin.
        
        Returns:
            cropped_image: [H_crop, W_crop, 3]
            cropped_mask: [H_crop, W_crop]
        """
        H, W = image.shape[:2]
        
        # Find bounding box of mask
        coords = np.column_stack(np.where(mask > 0))
        
        if len(coords) == 0:
            # Fallback: center crop if no valid mask
            center_x, center_y = W // 2, H // 2
            size = min(H, W) // 2
            x1 = max(0, center_x - size)
            y1 = max(0, center_y - size)
            x2 = min(W, center_x + size)
            y2 = min(H, center_y + size)
        else:
            y_min, x_min = np.min(coords, axis=0)
            y_max, x_max = np.max(coords, axis=0)
            
            # Add margin as fraction of bbox size
            bbox_w = x_max - x_min
            bbox_h = y_max - y_min
            
            margin_x = int(bbox_w * self.crop_margin)
            margin_y = int(bbox_h * self.crop_margin)
            
            x1 = max(0, x_min - margin_x)
            y1 = max(0, y_min - margin_y)
            x2 = min(W, x_max + margin_x)
            y2 = min(H, y_max + margin_y)
        
        # Crop
        cropped_image = image[y1:y2, x1:x2].copy()
        cropped_mask = mask[y1:y2, x1:x2].copy()
        
        return cropped_image, cropped_mask
    
    def _blur_background(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> np.ndarray:
        """Blur background to focus CLIP on object."""
        if mask is None or np.sum(mask) == 0:
            return image
        
        # Gaussian blur
        blurred = cv2.GaussianBlur(
            image,
            (self.background_blur, self.background_blur),
            0
        )
        
        # Blend: sharp object + blurred background
        mask_3d = mask.astype(np.float32)[:, :, np.newaxis]
        result = (mask_3d * image + (1 - mask_3d) * blurred).astype(np.uint8)
        
        return result
    
    def _add_letterbox(
        self,
        image: np.ndarray,
        target_size: Optional[int] = None
    ) -> np.ndarray:
        """Add letterbox padding to make image square for CLIP."""
        if target_size is None:
            target_size = self.input_size
        
        H, W = image.shape[:2]
        
        # Scale to fit within target
        scale = min(target_size / H, target_size / W)
        new_H = int(H * scale)
        new_W = int(W * scale)
        
        # Resize
        resized = cv2.resize(
            image,
            (new_W, new_H),
            interpolation=cv2.INTER_LINEAR
        )
        
        # Create square canvas with gray padding
        letterboxed = np.ones(
            (target_size, target_size, 3),
            dtype=np.uint8
        ) * 114
        
        # Center the resized image
        y_offset = (target_size - new_H) // 2
        x_offset = (target_size - new_W) // 2
        letterboxed[
            y_offset:y_offset + new_H,
            x_offset:x_offset + new_W
        ] = resized
        
        return letterboxed
    
    def extract_from_frame(
        self,
        image: np.ndarray,
        masks: List[np.ndarray],
        batch_size: int = None
    ) -> np.ndarray:
        """
        Extract CLIP embeddings for N objects from single frame.
        
        Args:
            image: Full frame RGB image [H, W, 3].
            masks: List of N binary masks [H, W] for each object.
            batch_size: Maximum batch size for CLIP processing.
        
        Returns:
            embeddings: Array [N, D] of CLIP embeddings (L2-normalized).
        """
        batch_size = cfg.clip.batch_size if batch_size is None else batch_size
        N = len(masks)
        
        if N == 0:
            return np.zeros((0, cfg.clip.embedding_dim))
        
        # === STEP 1: Prepare all crops ===
        pil_crops = []
        
        for i in range(N):
            mask = masks[i]
            
            # Dilate mask
            if mask is not None:
                mask = self._dilate_mask(mask)
            
            # Crop object
            cropped_image, cropped_mask = self._crop_object(image, mask)
            
            # Blur background
            if cropped_mask is not None:
                cropped_image = self._blur_background(
                    cropped_image,
                    cropped_mask
                )
            
            # Add letterbox
            letterboxed = self._add_letterbox(cropped_image)
            
            # Convert to PIL
            pil_crops.append(Image.fromarray(letterboxed))
        
        # === STEP 2: Process in batches ===
        all_embeddings = []
        
        for batch_start in range(0, N, batch_size):
            batch_end = min(batch_start + batch_size, N)
            batch_crops = pil_crops[batch_start:batch_end]
            
            # Apply CLIP transforms
            batch_tensors = [
                self.clip_transform(crop) for crop in batch_crops
            ]
            batch_tensor = torch.stack(batch_tensors).to(self.device)
            
            # Extract embeddings
            with torch.no_grad():
                embeddings = self.model.encode_image(batch_tensor)
                # L2 normalize for cosine similarity
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
                embeddings_np = embeddings.cpu().numpy()
            
            all_embeddings.append(embeddings_np)
        
        # Concatenate all batches
        return np.vstack(all_embeddings)  # [N, D]
    
    def extract_for_detections(
        self,
        image: np.ndarray,
        detections: List[dict]
    ) -> List[np.ndarray]:
        """
        Convenience method: extract embeddings for list of detections.
        
        Args:
            image: Full frame RGB image.
            detections: List of detection dicts with 'mask' key.
        
        Returns:
            List of CLIP embeddings [D] for each detection.
        """
        masks = [det.get("mask", None) for det in detections]
        embeddings = self.extract_from_frame(image, masks)
        return [embeddings[i] for i in range(len(detections))]


# ============================================================================
# GLOBAL EXTRACTOR (Simple Lazy Initialization)
# ============================================================================

_clip_extractor: Optional[CLIPAppearanceExtractor] = None


def get_clip_extractor(device: Optional[str] = None) -> CLIPAppearanceExtractor:
    """Get or create global CLIP extractor instance."""
    global _clip_extractor
    
    if _clip_extractor is None:
        target_device = device if device is not None else cfg.clip.device
        if target_device == "cuda" and not torch.cuda.is_available():
            target_device = "cpu"
        _clip_extractor = CLIPAppearanceExtractor(device=target_device)
    
    return _clip_extractor
