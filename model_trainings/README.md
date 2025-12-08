# Model Training Documentation

This directory contains the YOLOv8 model training pipeline for the Vision360 autonomous vehicle project. The models are trained to detect traffic signs, traffic lights, and other road-related objects.

## 📁 Directory Structure

```
model_trainings/
├── train_yolov8n.ipynb          # Main training notebook
├── data_processing.ipynb         # Data preprocessing and augmentation
├── download_data.ipynb           # Dataset download scripts
├── testing.ipynb                 # Model testing and evaluation
├── yolov8n.pt                   # Base YOLOv8n pretrained model
├── traffic_lights_and_signs.pt  # Trained model for traffic detection
├── traffic_lights_and_signs_real_car.pt  # Real car optimized model
└── data/                        # Training datasets directory
```

## 🚀 Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Environment Setup

1. Create a `.env` file in the project root with your Weights & Biases API key:
```
WANDB_API_KEY=your_wandb_api_key_here
```

2. Ensure your dataset is properly structured in YOLO format with a `data.yaml` configuration file.

## 📓 Notebooks Overview

### 1. `train_yolov8n.ipynb` - Main Training Pipeline

This is the primary notebook for training YOLOv8 models with Weights & Biases integration.

**Key Features:**
- Automated experiment tracking with W&B
- Configurable training parameters
- Multiple training versions with different hyperparameters
- Support for model freezing and fine-tuning
- Automated model checkpointing

**Training Configuration Example:**
```python
config = {
    "model_arch": "yolov8n.pt",
    "epochs": 70,
    "batch": 64,
    "imgsz": 640,
    "patience": 10,
    "task": "detect"
}
```

**Common Hyperparameter Configurations:**
- **Version 1-3**: Batch 8-32, Image size 400-480
- **Version 4-5**: Batch 32-48, Image size 480
- **Version 6+**: Batch 64, Image size 640

### 2. Model Validation

The notebook includes cells for:
- Validating trained models on test sets
- Computing mAP (mean Average Precision) metrics
- Per-class performance analysis
- COCO subset validation for road-related classes

### 3. Video Inference

Test trained models on video files:
```python
model = YOLO("traffic_lights_and_signs.pt")
results = model.predict(
    source="video.mp4",
    conf=0.25,
    save=True,
    save_conf=True
)
```

## 🎯 Training Workflow

### Step 1: Prepare Your Dataset

Ensure your dataset follows YOLO format:
```
data/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── data.yaml
```

### Step 2: Configure Training Parameters

Modify the config dictionary in the training cell:
- `epochs`: Number of training epochs (default: 70)
- `batch`: Batch size (adjust based on GPU memory)
- `imgsz`: Input image size (320, 480, 640, etc.)
- `patience`: Early stopping patience
- `workers`: Number of data loading workers

### Step 3: Run Training

Execute the training cell. The script will:
1. Initialize W&B run with auto-generated name
2. Load YOLOv8 base model
3. Train on your dataset
4. Save checkpoints every N epochs
5. Log metrics to W&B
6. Save best model weights

### Step 4: Validate Model

Run validation cells to:
- Evaluate model on test set
- Check mAP scores
- Analyze per-class performance

### Step 5: Test on Videos

Use video inference cells to visualize model performance on real footage.

## 📊 Experiment Tracking

All training runs are automatically logged to Weights & Biases with:
- Run name: `yolov8n_v{version}_{timestamp}_{arch}_epochs{e}_batch{b}_imgsz{size}`
- Metrics: Loss, mAP, precision, recall
- Model architecture and hyperparameters
- Training/validation images

## 🔧 Advanced Features

### Model Freezing

To freeze backbone layers during training:
```python
freeze=10  # Freeze first 10 layers
```

### Custom Data Augmentation

Modify mosaic augmentation closing epoch:
```python
close_mosaic=20  # Stop mosaic augmentation after epoch 20
```

### Checkpoint Frequency

Control checkpoint saving:
```python
save_period=10  # Save checkpoint every 10 epochs
```

## 📈 Model Versions

### Available Trained Models

1. **traffic_lights_and_signs.pt**
   - General traffic detection model
   - Trained on synthetic + real data

2. **traffic_lights_and_signs_real_car.pt**
   - Optimized for real car deployment
   - Fine-tuned on actual vehicle footage

3. **Custom Training Runs**
   - Stored in `capstone/` directory
   - Format: `yolov8n_v{N}_*`

## 🎓 Tips & Best Practices

1. **Batch Size**: Start with smaller batch sizes (8-16) for initial experiments, increase for final training
2. **Image Size**: Balance between accuracy (larger) and speed (smaller)
3. **Early Stopping**: Use patience parameter to avoid overfitting
4. **Data Quality**: Ensure labels are accurate and consistent
5. **Validation**: Always validate on a held-out test set
6. **GPU Memory**: Monitor GPU usage and adjust batch size accordingly

## 📚 Additional Resources

- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- [Weights & Biases Documentation](https://docs.wandb.ai/)
- [YOLO Data Format Guide](https://docs.ultralytics.com/datasets/)

## 🔗 Related Files

- Main vision system: `traffic_sign_light_detector.py`
- Configuration: `../config/config.yaml`
- Deployment models: `../models/`

## 📝 Notes

- All training runs automatically sync to W&B project "capstone"
- Entity name: "iv-drip-counter"
- Models are saved with best weights based on mAP metric
- Training progress can be monitored in real-time via W&B dashboard
