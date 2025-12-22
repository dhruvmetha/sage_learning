# Cropped Diffusion Model Inference Pipeline

This document describes the complete inference flow for the cropped diffusion goal prediction model.

## Overview

The cropped diffusion model predicts object goal poses in SE(2) space (x, y, theta). It uses:
- **Local context**: 5m × 5m window centered on the object being pushed
- **Cropped output**: Model predicts a center-cropped region (e.g., 32×32) which is padded back to context size (64×64)

## Architecture Summary

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        INFERENCE PIPELINE                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1. INPUT PREPARATION                                                    │
│     JSON Message + XML → Local Masks (5 channels, 224×224)              │
│                                                                          │
│  2. PREPROCESSING                                                        │
│     224×224 → Resize to 64×64 → Normalize to [-1, 1]                    │
│                                                                          │
│  3. DIFFUSION SAMPLING                                                   │
│     Context (5, 64, 64) → DiT → N samples at (1, 32, 32)               │
│                                                                          │
│  4. OUTPUT PADDING                                                       │
│     32×32 → Pad with -1 → 64×64 (predictions in center)                │
│                                                                          │
│  5. GOAL EXTRACTION                                                      │
│     Threshold → Find rectangle → Extract (px, py, angle)               │
│                                                                          │
│  6. COORDINATE CONVERSION                                                │
│     Pixel (64×64) → World coordinates (x, y, theta)                    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Step-by-Step Flow

### Step 1: Model Loading (`GoalInferenceModel.__init__`)

```python
model = GoalInferenceModel(
    model_path="/path/to/cropped_diffusion_crossattn/2025-12-16/04-45-01",
    device="cuda",
    sampler_method="ddim",  # Override sampler
    num_steps=20            # Sampling steps
)
```

**What happens:**
1. Load config from `{model_path}/.hydra/config.yaml`
2. Instantiate `GenerativeModuleCropped` with `DiTCroppedCrossAttn` network
3. Load checkpoint weights
4. Detect settings:
   - `use_local = True` (default) → Use local object-centered masks
   - `context_size = 64` → Input/output resolution
   - `crop_size = 32` → Model's actual prediction size

**Console output:**
```
✅ Goal model loaded successfully: GenerativeModuleCropped
  Sampler: HFDiffusionSampler (method: ddim)
  Using local (object-centered) masks
  Cropped output model: context=64, crop=32 (padded to context)
```

### Step 2: Input Preparation (`_infer_local`)

```python
valid_goals = model.infer(
    json_message=scene_state,      # Current scene state
    xml_path="/path/to/env.xml",   # MuJoCo environment
    robot_goal=(5.0, 3.0),         # Robot's target position
    selected_object="box_1",       # Object to push
    samples=32                     # Number of goal samples
)
```

**2a. Create Local Masks**

Uses `MLImageConverterAdapter.create_local_masks()`:

```python
local_data = image_converter.create_local_masks(
    data_point=json_message,
    selected_object="box_1",
    robot_goal_pos=(5.0, 3.0),
    crop_size_meters=5.0,      # 5m × 5m window
    highres_size=1024,         # Render at high res first
    output_size=224            # Downsample to 224×224
)
```

**Local masks generated (all 224×224, centered on object):**

| Channel | Name | Content |
|---------|------|---------|
| 0 | `local_static` | Walls and static obstacles |
| 1 | `local_movable` | Other movable objects (not target) |
| 2 | `local_target_object` | The object being pushed |
| 3 | `local_robot_region` | Robot's reachable area |
| 4 | `local_goal_sample_region` | Goal region circles |

**Metadata returned:**
- `object_center`: (x, y) world position of selected object
- `object_theta`: Current rotation of object (radians)
- `crop_size_meters`: 5.0
- `resolution`: 5.0 / 224 ≈ 0.022 m/pixel

**2b. Stack and Transform**

```python
# Stack 5 channels
input_channels = [
    local_data['local_static'],
    local_data['local_movable'],
    local_data['local_target_object'],
    local_data['local_robot_region'],
    local_data['local_goal_sample_region'],
]
inp = np.concatenate(input_channels, axis=-1)  # (224, 224, 5)

# Transform: ToTensor → Resize → Normalize
transform = transforms.Compose([
    transforms.ToTensor(),                           # (5, 224, 224)
    transforms.Resize((64, 64)),                     # (5, 64, 64)
    transforms.Lambda(lambda x: x * 2 - 1),          # [-1, 1]
])
inp_tensor = transform(inp).unsqueeze(0).to(device)  # (1, 5, 64, 64)
```

### Step 3: Diffusion Sampling (`GenerativeModuleCropped.sample_from_model`)

```python
goal_samples = model.sample_from_model(
    context=inp_tensor,  # (1, 5, 64, 64)
    samples=32,          # Generate 32 samples
    num_steps=20         # DDIM steps
)
# Returns: (32, 1, 64, 64) - already padded!
```

**Inside `sample_from_model`:**

```python
def sample_from_model(self, context, samples=32, num_steps=20):
    # Repeat context for batch sampling
    context_repeated = context.repeat(samples, 1, 1, 1)  # (32, 5, 64, 64)

    # Initialize noise at CROP size (not context size!)
    x_init = torch.randn(samples, 1, self.crop_size, self.crop_size)  # (32, 1, 32, 32)

    # Define model function with context conditioning
    def model_fn(x, t):
        return self.network(x, t, context_repeated)  # DiTCroppedCrossAttn

    # Run DDIM sampling
    samples_out = self.sampler.sample(
        model=model_fn,
        x_init=x_init,
        num_steps=num_steps
    )  # (32, 1, 32, 32)

    # PAD to context size for inference compatibility
    if self.crop_size < self.context_size:
        pad = (self.context_size - self.crop_size) // 2  # (64-32)//2 = 16
        samples_out = F.pad(samples_out, (pad, pad, pad, pad), value=-1)
        # Now: (32, 1, 64, 64) with actual prediction in center 32×32

    return samples_out
```

**Key insight:** The model predicts a 32×32 region, but output is padded to 64×64 with `-1` values around the edges. The actual prediction is in the center.

### Step 4: Post-processing and Normalization

```python
# Convert from [-1, 1] to [0, 1] and move to numpy
goal_samples = (goal_samples.permute(0, 2, 3, 1).cpu().numpy() + 1) / 2
# Shape: (32, 64, 64, 1)
```

### Step 5: Goal Extraction (per sample)

For each of the 32 samples:

```python
for i, goal_sample in enumerate(goal_samples):
    # 5a. Threshold to binary mask
    goal_mask = (goal_sample[:, :, 0] > 0.5).astype(np.uint8)

    # 5b. Validate: skip if multiple disconnected regions
    num_labels, _, _, _ = cv2.connectedComponentsWithStats(goal_mask)
    if num_labels > 2:  # Background + 1 object expected
        continue

    # 5c. Find rectangle center and angle
    corners, rect, center, angle = find_rectangle_corners(goal_mask)
    # center: (px, py) in 64×64 pixel space
    # angle: rotation in degrees

    if center is None:
        continue
```

### Step 6: Coordinate Conversion

**6a. Pixel to World Position**

```python
world_x, world_y = image_converter.pixel_to_world_local(
    px=center[0],
    py=center[1],
    object_center=local_data['object_center'],
    crop_size_meters=5.0,
    output_size=64  # context_size, NOT crop_size!
)
```

**The conversion formula:**
```python
resolution = crop_size_meters / output_size  # 5.0 / 64 = 0.078 m/pixel
center_px = output_size / 2  # 32

world_x = object_center[0] + (px - center_px) * resolution
world_y = object_center[1] + (py - center_px) * resolution
```

**Example:**
- Object at world position (3.0, 4.0)
- Predicted center at pixel (40, 28)
- `world_x = 3.0 + (40 - 32) * 0.078 = 3.0 + 0.625 = 3.625`
- `world_y = 4.0 + (28 - 32) * 0.078 = 4.0 - 0.312 = 3.688`

**6b. Angle to Theta**

```python
# Get current object angle from input mask
obj_mask = cv2.resize(local_data['local_target_object'], (64, 64))
_, _, _, obj_angle = find_rectangle_corners(obj_mask)  # degrees

# Compute angle difference
angle_diff_deg = goal_angle - obj_angle

# Handle 180° ambiguity (rectangles look same rotated 180°)
if angle_diff_deg > 90:
    angle_diff_deg -= 180
elif angle_diff_deg < -90:
    angle_diff_deg += 180

# Convert to world theta
goal_theta = object_theta + np.radians(angle_diff_deg)

# Normalize to [-π, π]
while goal_theta > np.pi:
    goal_theta -= 2 * np.pi
while goal_theta < -np.pi:
    goal_theta += 2 * np.pi
```

### Step 7: Return Valid Goals

```python
valid_goals.append({
    'index': i,
    'x': world_x,
    'y': world_y,
    'theta': goal_theta,
    'goal_sample': goal_sample,      # Raw 64×64 output
    'input_channels': inp_for_goal   # For visualization
})

return valid_goals  # List of SE(2) goals in world coordinates
```

## Coordinate Spaces Summary

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         COORDINATE SPACES                                 │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  WORLD SPACE (meters)                                                     │
│  ├── Full environment bounds (e.g., 0-10m × 0-8m)                        │
│  └── Object positions in absolute coordinates                            │
│                                                                           │
│  LOCAL WORLD SPACE (meters)                                               │
│  ├── 5m × 5m window centered on object                                   │
│  └── Object at center (0, 0) relative to crop                           │
│                                                                           │
│  HIGH-RES PIXEL SPACE (1024×1024)                                        │
│  └── Intermediate rendering for quality                                  │
│                                                                           │
│  INPUT PIXEL SPACE (224×224)                                             │
│  ├── Local masks before model                                            │
│  └── resolution = 5.0m / 224px ≈ 0.022 m/px                             │
│                                                                           │
│  CONTEXT PIXEL SPACE (64×64)                                             │
│  ├── Model input after resize                                            │
│  ├── Output after padding                                                │
│  └── resolution = 5.0m / 64px ≈ 0.078 m/px                              │
│                                                                           │
│  CROP PIXEL SPACE (32×32)                                                │
│  ├── Model's actual prediction region                                    │
│  └── Center of context space                                             │
│                                                                           │
└──────────────────────────────────────────────────────────────────────────┘
```

## Key Design Decisions

### Why Local Masks?
- **Focused context**: 5m window captures relevant obstacles
- **Translation invariance**: Model learns relative positions
- **Consistent scale**: Same resolution regardless of environment size

### Why Cropped Output?
- **Computational efficiency**: Predict 32×32 instead of 64×64
- **Focused prediction**: Object goal is near the center
- **Padding for compatibility**: Inference code works with fixed 64×64 output

### Why Pad with -1?
- `-1` in normalized space corresponds to 0 after denormalization
- Threshold at 0.5 ignores padded region
- Only center 32×32 contains valid predictions

## Configuration Reference

From `region_opening_ml_collection.yaml`:

```yaml
# Model path
ml_goal_model: /path/to/cropped_diffusion_crossattn/...

# Inference settings
ml_device: cuda
ml_samples: 32              # Number of diffusion samples
ml_sampler_method: ddim     # DDIM for fast sampling
ml_num_steps: 20            # Sampling steps

# Primitive alignment (after inference)
ml_match_position_tolerance: 0.2  # 20cm
ml_match_angle_tolerance: 0.2     # ~11°
ml_k_nearest: 1                   # Top-1 voting
```

## Model Config (from training)

```yaml
context_size: 64            # Input resolution
crop_size: 32               # Output prediction size

model:
  _target_: src.model.generative_module_cropped.GenerativeModuleCropped
  network:
    _target_: src.model.dit.dit_cropped_crossattn.DiTCroppedCrossAttn
    context_channels: 5     # 5 input masks
    out_ch: 1               # 1 output channel (goal mask)
    dim: 256                # Transformer dimension
    depth: 8                # Transformer blocks
```

## Debugging Tips

### Check Model Output Shape
```python
print(f"Raw output shape: {goal_samples.shape}")  # Should be (N, 64, 64, 1)
print(f"Center value range: {goal_samples[:, 16:48, 16:48, :].min():.2f} to {goal_samples[:, 16:48, 16:48, :].max():.2f}")
print(f"Padding values: {goal_samples[:, 0, 0, 0]}")  # Should be close to 0
```

### Visualize Predictions
```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
axes[0].imshow(goal_samples[0, :, :, 0], cmap='gray')
axes[0].set_title('Sample 0')
axes[1].imshow(goal_samples[0, :, :, 0] > 0.5, cmap='gray')
axes[1].set_title('Thresholded')
# ... add more
plt.show()
```

### Check Coordinate Conversion
```python
# Pixel (32, 32) should map to object center
test_x, test_y = pixel_to_world_local(32, 32, object_center, 5.0, 64)
print(f"Center pixel maps to: ({test_x:.3f}, {test_y:.3f})")
print(f"Object center is: {object_center}")
# These should match!
```
