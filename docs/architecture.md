# Architecture Guide

## Overview

The sage_learning codebase implements generative models for goal prediction in robotic manipulation. The architecture follows a modular design with pluggable components.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           TRAINING PIPELINE                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  .npz files ──► MaskDiffusionDataModule ──► GenerativeModule ──► .ckpt │
│                      (data loading)           (training)                │
│                                                                         │
│  GenerativeModule contains:                                             │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  network: DiT (Diffusion Transformer)                           │   │
│  │  path: FlowMatchingPath or DiffusionPath                        │   │
│  │  sampler: ODESampler or DDPMSampler                             │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                          INFERENCE PIPELINE                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  JSON scene ──► GoalInferenceModel ──► Goal proposals (x, y, θ)        │
│                                                                         │
│  GoalInferenceModel contains:                                           │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  model: GenerativeModule (loaded from checkpoint)               │   │
│  │  image_converter: MLImageConverterAdapter (JSON → images)       │   │
│  │  post-processing: mask → SE(2) pose extraction                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. GenerativeModule (`src/model/generative_module.py`)

The main PyTorch Lightning module that orchestrates training and inference.

**Key Methods:**
- `training_step()`: Samples from path, computes loss
- `sample()`: Generates samples using the configured sampler
- `sample_from_model()`: Legacy interface for inference

**Configuration:**
```python
GenerativeModule(
    network: nn.Module,          # DiT backbone
    path: BasePath,              # FlowMatchingPath or DiffusionPath
    sampler: BaseSampler,        # ODESampler or DDPMSampler
    optimizer: Any,              # AdamW partial
    aux_loss_weight: float = 0.0,
    context_channels: int = 5,
    target_channels: int = 1,
)
```

### 2. DiT Network (`src/model/dit/dit.py`)

Diffusion Transformer architecture with Adaptive Layer Normalization (AdaLN).

**Architecture:**
```
Input (B, 6, H, W)
    ↓
PatchEmbed (4×4 patches) → (B, num_patches, dim)
    ↓
+ Positional Embeddings
    ↓
8× TransformerBlockAdaLN (with time conditioning)
    ↓
LayerNorm
    ↓
Unpatchify (ConvTranspose2d)
    ↓
Output (B, 1, H, W)
```

**Time Conditioning:**
- Time `t ∈ [0, 1]` is embedded via sinusoidal encoding
- AdaLN modulates layer norm parameters based on time
- Each transformer block has its own time projection

### 3. Paths (`src/model/paths/`)

Define the interpolation between noise (x₀) and data (x₁) during training.

**FlowMatchingPath:**
```python
# Optimal Transport Conditional Flow Matching
x_t = (1 - t) * x_0 + t * x_1          # Linear interpolation
target = x_1 - x_0                      # Velocity (dx/dt)
prediction_type = 'velocity'
```

**DiffusionPath:**
```python
# DDPM Forward Process
α_t = cumprod(1 - β_t)
x_t = sqrt(α_t) * x_1 + sqrt(1 - α_t) * x_0    # Noisy interpolation
target = x_0                                     # Noise (ε)
prediction_type = 'noise'
```

### 4. Samplers (`src/model/samplers/`)

Define how to generate samples from noise during inference.

**ODESampler (for Flow Matching):**
```python
# Solve ODE: dx/dt = v(x, t)
for t in linspace(0, 1, num_steps):
    v = model(x_t, t)
    x_{t+dt} = x_t + v * dt  # Euler, Midpoint, or RK4
```

**DDPMSampler (for Diffusion):**
```python
# Reverse diffusion process
for t in range(T, 0, -1):
    ε_pred = model(x_t, t)
    x_{t-1} = (x_t - β_t * ε_pred / sqrt(1-α_t)) / sqrt(1-β_t) + σ_t * z
```

### 5. Data Module (`src/data/mask_diffusion_data.py`)

Loads .npz files containing scene images and target masks.

**Dataset Keys:**
```python
{
    'robot_image': (H, W),           # Robot position
    'goal_image': (H, W),            # Robot goal
    'movable_objects_image': (H, W), # All movable objects
    'static_objects_image': (H, W),  # Walls/obstacles
    'target_object': (H, W),         # Selected object
    'target_goal': (H, W),           # Ground truth goal
}
```

**Transforms:**
1. ToTensor
2. Resize to `image_size` (default 64)
3. Normalize to [-1, 1]

### 6. GoalInferenceModel (`ktamp_learning/goal_inference_model.py`)

High-level inference API that handles:
1. Loading model from checkpoint
2. Converting JSON scene to image tensors
3. Running inference
4. Converting output mask to SE(2) poses

## Data Flow

### Training
```
1. Load batch: {robot, goal, movable, static, target_object, target_goal}
2. Build context: concat([robot, goal, movable, static, target_object]) → (B, 5, H, W)
3. Sample noise: x_0 ~ N(0, 1) → (B, 1, H, W)
4. Sample time: t ~ U(0, 1) → (B,)
5. Interpolate: x_t = path.sample(x_0, x_1=target_goal, t)
6. Build input: concat([context, x_t]) → (B, 6, H, W)
7. Predict: v_pred = network(input, t) → (B, 1, H, W)
8. Loss: MSE(v_pred, target) + aux_loss_weight * dice_loss
```

### Inference
```
1. Build context from scene JSON → (1, 5, H, W)
2. Initialize: x_0 ~ N(0, 1) → (samples, 1, H, W)
3. For each step t ∈ [0, 1]:
   a. Build input: concat([context, x_t])
   b. Predict velocity: v = network(input, t)
   c. Update: x_t = x_t + v * dt
4. Threshold: mask = (x_T > 0.5)
5. Extract pose: find_rectangle_corners(mask) → (x, y, θ)
```

## Configuration System

Uses Hydra for configuration management. Configs are composed from:

```
config/
├── train_flow_matching.yaml     # Main training config
├── model/
│   ├── generative_flow_matching.yaml
│   └── diffusion_only_goal.yaml
├── data/
│   └── mask_diffusion_data.yaml
├── network/
│   └── dit.yaml
├── trainer/
│   └── default.yaml
└── callbacks/
    └── default.yaml
```

## Design Patterns

1. **Strategy Pattern**: Paths and Samplers are interchangeable strategies
2. **Dependency Injection**: Components are injected via Hydra instantiation
3. **Template Method**: GenerativeModule defines training loop, delegates to components
4. **Adapter Pattern**: GoalInferenceModel adapts model to high-level API
