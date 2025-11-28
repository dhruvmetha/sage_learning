# Goal Inference Model Documentation

## Overview

This package provides a specialized inference model for robotic manipulation goal prediction:

- **`GoalInferenceModel`**: Generates goal poses in SE(2) space for a selected object

The model provides a clean, reusable interface for predicting where to push a known object.

## Architecture

```
json_message + xml_path + robot_goal + selected_object
                    ↓
          [GoalInferenceModel]
                    ↓
              goal_proposals (x, y, theta)
```

## Installation

### Prerequisites
- **Environment**: mjxrl conda environment
- **Models**: Trained generative models in outputs/ directory
- **Dependencies**: PyTorch, Hydra, OpenCV, NumPy

### Setup
```bash
# Activate environment
conda activate /common/users/dm1487/envs/mjxrl

# Install learning package
cd /path/to/sage_learning
pip install -e .
```

## API Reference

### GoalInferenceModel

#### Constructor
```python
GoalInferenceModel(model_path: str, device: str = "cuda")
```

**Parameters:**
- `model_path`: Path to trained goal model (e.g., `"outputs/rel_reach_coord_goal_dit/mse/2025-08-10_06-59-27"`)
- `device`: PyTorch device ("cuda" or "cpu")

#### infer() Method
```python
infer(json_message: dict, xml_path: str, robot_goal: list, selected_object: str, samples: int = 32) -> list
```

**Parameters:**
- `json_message`: Environment state data (see Input Format below)
- `xml_path`: Relative path to MuJoCo XML file
- `robot_goal`: Target position as `[x, y]` coordinates
- `selected_object`: Name of object to generate goals for
- `samples`: Number of diffusion samples to generate

**Returns:**
List of goal dictionaries with:
- `index`: Sample index from diffusion model
- `goal_center`: World coordinates `[x, y]`
- `final_quat`: Quaternion for object rotation `[w, x, y, z]`
- `x`, `y`, `theta`: SE(2) pose components
- `goal_sample`: Raw diffusion output array

## Input Format

### JSON Message Structure
```python
json_message = {
    "xml_path": "custom_walled_envs/jun22/env_12345.xml",
    "robot_goal": [1.5, 2.0],  # Target position [x, y]
    "robot": {
        "position": [0.0, 0.0, 0.1]  # Robot state [x, y, z]
    },
    "objects": {
        "box1": {
            "position": [1.2, 1.8, 0.05],              # World coordinates [x, y, z]
            "quaternion": [1.0, 0.0, 0.0, 0.0]         # Orientation [w, x, y, z]
        },
        "cylinder1": {
            "position": [2.1, 0.5, 0.05],
            "quaternion": [0.9, 0.0, 0.0, 0.4]
        }
        # ... additional objects
    }
}
```

## Usage Examples

### Basic Goal Generation
```python
from ktamp_learning import GoalInferenceModel

# Initialize model
goal_model = GoalInferenceModel("outputs/rel_reach_coord_goal_dit/mse/2025-08-10_06-59-27")

# Prepare environment data
json_message = {
    "xml_path": "custom_walled_envs/jun22/env_001.xml",
    "robot_goal": [1.5, 2.0],
    "robot": {"position": [0.0, 0.0, 0.1]},
    "objects": {
        "box1": {
            "position": [1.2, 1.8, 0.05],
            "quaternion": [1.0, 0.0, 0.0, 0.0]
        }
    }
}

# Generate goals for a known object
goals = goal_model.infer(
    json_message=json_message,
    xml_path="custom_walled_envs/jun22/env_001.xml",
    robot_goal=[1.5, 2.0],
    selected_object="box1",
    samples=32
)

# Analyze results
print(f"Generated {len(goals)} goal proposals:")
for goal in goals[:5]:  # Show first 5
    print(f"  pos=({goal['x']:.3f}, {goal['y']:.3f}), θ={goal['theta']:.3f}")
```

### Multiple Objects Comparison
```python
from ktamp_learning import GoalInferenceModel

goal_model = GoalInferenceModel("outputs/goal_model_path")

# Generate goals for different objects
goals_box = goal_model.infer(json_message, xml_path, robot_goal, "box1")
goals_cyl = goal_model.infer(json_message, xml_path, robot_goal, "cylinder1")

print(f"Box goals: {len(goals_box)}")
print(f"Cylinder goals: {len(goals_cyl)}")
```

## Performance Considerations

### Memory Usage
- **Goal Model**: ~2-4GB GPU memory (depending on architecture)
- **Tip**: Use CPU device if GPU memory is limited

### Inference Speed
- **Goal Inference**: ~100-200ms for 32 samples
- **Flow Matching**: ~20 steps (faster than DDPM)
- **DDPM Diffusion**: ~100 steps (more established)

### Sample Count Guidelines
- **Development/Testing**: 8-16 samples for faster iteration
- **Production**: 32 samples (default)
- **High Precision**: 64+ samples for better coverage

## Error Handling

```python
try:
    goals = goal_model.infer(json_message, xml_path, robot_goal, selected_object)

except FileNotFoundError as e:
    print(f"Model or config file not found: {e}")

except ValueError as e:
    print(f"Invalid input data: {e}")

except RuntimeError as e:
    print(f"Model inference error (GPU/memory?): {e}")
```

### Validation Checks
```python
# Validate goal generation
if len(goals) < 3:
    print("⚠️ Few goals generated - increase sample count?")

# Check goal quality
goal_spread = max(g['x'] for g in goals) - min(g['x'] for g in goals)
if goal_spread < 0.1:
    print("⚠️ Goals are very clustered - model may be overconfident")
```

## Technical Notes

### Coordinate Systems
- **Input**: World coordinates from MuJoCo simulation
- **Processing**: Pixel coordinates in 224x224 images
- **Output**: World coordinates converted back via ImageConverter

### Model Architecture
- Supports both Flow Matching and DDPM Diffusion
- DiT (Diffusion Transformer) backbone
- Single-channel output: target_goal mask in SE(2) space

### Input Channels
- robot: Robot position (channel 1)
- goal: Robot goal position (channel 2)
- movable: All movable objects (channel 3)
- static: Walls/static obstacles (channel 4)
- target_object: The selected object mask (channel 5)

## Troubleshooting

### Model Loading Issues
- Verify model path exists and contains `.hydra/config.yaml`
- Check checkpoint files exist in `checkpoints/` directory
- Ensure sufficient GPU memory available

### Inference Issues
- Validate JSON message format matches expected structure
- Check XML file exists at specified path
- Verify object names in JSON match XML model

### Performance Issues
- Reduce sample count for faster inference
- Use CPU device if GPU memory limited
- Consider Flow Matching for faster inference (20 steps vs 100)
