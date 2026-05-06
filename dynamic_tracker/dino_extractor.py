"""
DINOv2 Appearance Extractor for SAM3 Tracker

Optimized batch extraction: One image + N masks → N DINOv2 embeddings.
Better than CLIP for fine-grained visual similarity (color, texture, details).
"""

import cv2
import numpy as np
import torch
from typing import List, Tuple, Optional
from PIL import Image
import torchvision.transforms as transforms

# ============================================================================
# DINO IMPORT (with fallback)
# ============================================================================

try:
    import timm
    DINO_AVAILABLE = True
except ImportError:
    DINO_AVAILABLE = False
    print("⚠️  timm not installed. Install with: pip install timm")


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration parameters for DINOv2 appearance extraction."""
    
    # === DINO MODEL ===
    DINO_MODEL_NAME = "vit_small_patch14_dinov2"  # Options: vit_small, vit_base, vit_large
    DINO_INPUT_SIZE = 518 # Standard DINO input resolution
    DINO_EMBEDDING_DIM = 384  # Output dimension for vit_small (1024 for vit_base)
    
    # === CROP PREPROCESSING ===
    CROP_MARGIN = 0.2  # 20% padding around object bounding box
    MASK_DILATION_KERNEL = 5  # Pixels for mask dilation
    BACKGROUND_BLUR = 15  # Gaussian blur kernel size for background
    
    # === BATCH PROCESSING ===
    DINO_BATCH_SIZE = 32  # Max objects per DINO forward pass
    DINO_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    
    # === EMBEDDING SMOOTHING ===
    DINO_EMA_ALPHA = 0.15  # Slow update for appearance stability


# ============================================================================
# DINOV2 APPEARANCE EXTRACTOR
# ============================================================================

class DINOAppearanceExtractor:
    """
    DINOv2 extractor: One image + N masks → N embeddings.
    
    Better than CLIP for fine-grained visual similarity.
    No internal caching or smoothing — handled by Track class.
    """
    
    def __init__(
        self,
        model_name: str = Config.DINO_MODEL_NAME,
        device: str = Config.DINO_DEVICE,
        input_size: int = Config.DINO_INPUT_SIZE,
        crop_margin: float = Config.CROP_MARGIN,
        mask_dilation: int = Config.MASK_DILATION_KERNEL,
        background_blur: int = Config.BACKGROUND_BLUR
    ):
        """
        Initialize DINOv2 model.
        
        Args:
            model_name: DINO model name (timm).
            device: Device to run model on ('cuda' or 'cpu').
            input_size: DINO input resolution.
            crop_margin: Padding margin around object.
            mask_dilation: Dilation kernel size for mask.
            background_blur: Gaussian blur kernel for background.
        """
        if not DINO_AVAILABLE:
            raise ImportError(
                "timm is not installed. Install with:\n"
                "pip install timm"
            )
        
        self.device = device
        self.input_size = input_size
        self.crop_margin = crop_margin
        self.mask_dilation = mask_dilation
        self.background_blur = background_blur
        
        # Load DINOv2 model (no classification head)
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.model.eval().to(device)
        
        # DINO preprocessing transform (ImageNet normalization)
        self.dino_transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet mean
                std=[0.229, 0.224, 0.225]   # ImageNet std
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
        """Blur background to focus DINO on object."""
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
        """Add letterbox padding to make image square for DINO."""
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
        batch_size: int = Config.DINO_BATCH_SIZE
    ) -> np.ndarray:
        """
        Extract DINOv2 embeddings for N objects from single frame.
        
        Args:
            image: Full frame RGB image [H, W, 3].
            masks: List of N binary masks [H, W] for each object.
            batch_size: Maximum batch size for DINO processing.
        
        Returns:
            embeddings: Array [N, D] of DINOv2 embeddings (L2-normalized).
        """
        N = len(masks)
        
        if N == 0:
            return np.zeros((0, Config.DINO_EMBEDDING_DIM))
        
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
            
            # Apply DINO transforms
            batch_tensors = [
                self.dino_transform(crop) for crop in batch_crops
            ]
            batch_tensor = torch.stack(batch_tensors).to(self.device)
            
            # Extract embeddings (no gradient)
            with torch.no_grad():
                embeddings = self.model(batch_tensor)
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
            List of DINOv2 embeddings [D] for each detection.
        """
        masks = [det.get("mask", None) for det in detections]
        embeddings = self.extract_from_frame(image, masks)
        return [embeddings[i] for i in range(len(detections))]


# ============================================================================
# GLOBAL EXTRACTOR (Simple Lazy Initialization)
# ============================================================================

_dino_extractor: Optional[DINOAppearanceExtractor] = None


def get_dino_extractor(device: Optional[str] = None) -> DINOAppearanceExtractor:
    """Get or create global DINOv2 extractor instance."""
    global _dino_extractor
    
    if _dino_extractor is None:
        target_device = device if device is not None else Config.DINO_DEVICE
        _dino_extractor = DINOAppearanceExtractor(device=target_device)
    
    return _dino_extractor