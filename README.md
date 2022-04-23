# kaggle-happy-whale
Code and solution for the [Happywhale - Whale and Dolphin Identification](https://www.kaggle.com/competitions/happy-whale-and-dolphin) competition (81 / 1588 place).

## Dataset
Duplicates have been removed. For training we used cropped images generated using YOLOv5. We did crop by dorsal fin for species who have it and crop by full body for the rest species. The same rule was applied for the test dataset. 

## Validation
Stratified 5 folds split using `invidual_id` + `image_ratio (height VS width)`. Single inviduals are labeled as new individuals.

## Modeling
Our approach is based ArcFace-Head + Gem-Pooling architectures trained on ArcFace loss. We used EfficientNet family models as encoders (backbones) for our final ensemble. 

## Training set up
- Epochs: 25
- Optimizer: Adam
- LR: 0.001
- Scheduler: Cosine
- Mixed precision: ON
- Gradient accumulation: ON (2-8 based on GPU)
- Gradient clipping: OFF
- Batchnorms freeze: OFF

## Data pipeline
```
self.transforms = A.Compose([
    A.Resize(384, 512, p=1.0),

    A.OneOf([
        A.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.12, hue=0.1, p=0.75),
        A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=15, val_shift_limit=15, p=0.75),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.75),
    ], p=0.75),

    A.OneOf([
        A.Blur(blur_limit=2.5, p=0.75),
        A.MotionBlur(blur_limit=(3, 5), p=0.75),
    ], p=0.75),

    A.OneOf([
        A.Affine(shear=21, mode=cv2.BORDER_REPLICATE, p=0.75),
        A.Compose([
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05,
            rotate_limit=16, border_mode=cv2.BORDER_REPLICATE, p=0.75),
            A.GridDistortion(distort_limit=0.1, border_mode=cv2.BORDER_REPLICATE, p=0.75),
        ], p=0.75)
    ], p=0.75),

    A.OneOf([
        A.Cutout(num_holes=12, max_h_size=21, max_w_size=21, p=0.75),
        A.Cutout(num_holes=6, max_h_size=42, max_w_size=42, p=0.75),
    ], p=0.75),

    A.CLAHE(clip_limit=2.5, p=0.75),
    A.ImageCompression(quality_lower=90, quality_upper=100, p=0.75),
    A.Normalize(p=1.0),
])
```

## Inference and Post-processing
1. Switch model to headless mode.
2. Extract train embeddings.
3. Extract test embeddings.
4. Find the closest sample in train dataset for each sample in test dataset using KNN based on cosine similarity.
5. Label test samples with the highest similarity less than threshold (0.5-0.6 for different models) as a new individual.

## Didn't work
- Subcenter ArcFace head
- Segmentation mask as 4th channel or for background removal
- Smart Resize
- DOLG
- Grayscale 

**Thanks a lot to my team;)**
