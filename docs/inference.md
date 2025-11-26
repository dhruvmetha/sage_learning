# Inference Guide

## Overview

The `GoalInferenceModel` class provides a high-level API for generating goal predictions from trained models.

## Quick Start

```python
from ktamp_learning import GoalInferenceModel

# Load model
model = GoalInferenceModel("outputs/run_name/", device="cuda")

# Prepare scene data
json_message = {
    "robot": {"position": [0.0, 0.0, 0.1]},
    "objects": {
        "box1": {
            "position": [1.2, 1.8, 0.05],
            "quaternion": [1.0, 0.0, 0.0, 0.0]
        }
    }
}

# Generate goals
goals = model.infer(
    json_message=json_message,
    xml_path="path/to/env.xml",
    robot_goal=[2.0, 3.0],
    selected_object="box1",
    samples=32
)

# Use results
for goal in goals:
    print(f"Goal: x={goal['x']:.3f}, y={goal['y']:.3f}, θ={goal['theta']:.3f}")
```

## API Reference

### GoalInferenceModel

```python
class GoalInferenceModel:
    def __init__(self, model_path: str, device: str = "cuda"):
        """
        Initialize goal inference model.

        Args:
            model_path: Path to training output directory containing:
                - .hydra/config.yaml
                - checkpoints/*.ckpt
            device: "cuda" or "cpu"
        """
```

### infer()

```python
def infer(
    self,
    json_message: dict,
    xml_path: str,
    robot_goal: list,
    selected_object: str,
    samples: int = 32
) -> list:
    """
    Generate goal proposals for a selected object.

    Args:
        json_message: Scene state dictionary
        xml_path: Path to MuJoCo XML (relative to resources)
        robot_goal: Target position [x, y]
        selected_object: Name of object to push
        samples: Number of samples to generate

    Returns:
        List of goal dictionaries, each containing:
        - 'x': float - X coordinate in world frame
        - 'y': float - Y coordinate in world frame
        - 'theta': float - Rotation angle (radians)
        - 'goal_center': [x, y] - World coordinates
        - 'final_quat': [w, x, y, z] - Quaternion
        - 'goal_sample': np.array - Raw mask output
    """
```

## Input Format

### JSON Message Structure

```python
json_message = {
    # Robot state
    "robot": {
        "position": [x, y, z]  # World coordinates
    },

    # All objects in scene
    "objects": {
        "object_name": {
            "position": [x, y, z],           # World coordinates
            "quaternion": [w, x, y, z]       # Rotation (scalar-first)
        },
        # ... more objects
    },

    # Optional: reachable objects (not used in goal-only)
    "reachable_objects": ["obj1", "obj2"]
}
```

### XML Path

The `xml_path` parameter should be a relative path from the MuJoCo resources directory:
```python
xml_path = "custom_walled_envs/jun22/env_001.xml"
# Resolved to: /path/to/ml4kp_ktamp/resources/models/custom_walled_envs/jun22/env_001.xml
```

## Output Format

Each goal in the returned list contains:

| Key | Type | Description |
|-----|------|-------------|
| `x` | float | X coordinate in world frame |
| `y` | float | Y coordinate in world frame |
| `theta` | float | Rotation angle in radians |
| `goal_center` | [float, float] | World coordinates [x, y] |
| `final_quat` | [float, float, float, float] | Quaternion [w, x, y, z] |
| `index` | int | Sample index |
| `goal_sample` | np.array | Raw mask output (H, W, 1) |

## Usage Patterns

### Single Goal Selection

```python
goals = model.infer(json_message, xml_path, robot_goal, "box1", samples=32)

if goals:
    # Take first valid goal
    best_goal = goals[0]
    target_pose = (best_goal['x'], best_goal['y'], best_goal['theta'])
```

### Multiple Samples Analysis

```python
goals = model.infer(json_message, xml_path, robot_goal, "box1", samples=64)

# Analyze distribution
import numpy as np
xs = [g['x'] for g in goals]
ys = [g['y'] for g in goals]
print(f"Mean: ({np.mean(xs):.3f}, {np.mean(ys):.3f})")
print(f"Std:  ({np.std(xs):.3f}, {np.std(ys):.3f})")
```

### Comparing Objects

```python
# Generate goals for different objects
goals_box = model.infer(json_message, xml_path, robot_goal, "box1")
goals_cyl = model.infer(json_message, xml_path, robot_goal, "cylinder1")

print(f"Box1: {len(goals_box)} valid goals")
print(f"Cylinder1: {len(goals_cyl)} valid goals")
```

### Error Handling

```python
try:
    goals = model.infer(json_message, xml_path, robot_goal, selected_object)

    if not goals:
        print("No valid goals generated")
        # Fallback strategy
    else:
        goal = goals[0]

except FileNotFoundError as e:
    print(f"Model or config not found: {e}")
except ValueError as e:
    print(f"Invalid object: {e}")
except RuntimeError as e:
    print(f"Inference error: {e}")
```

## Performance Tips

### Sample Count Guidelines

| Use Case | Samples | Latency |
|----------|---------|---------|
| Real-time | 8-16 | ~50ms |
| Standard | 32 | ~100ms |
| High precision | 64-128 | ~200-400ms |

### Memory Usage

- Model: ~2-4GB GPU memory
- Per-batch: ~100MB for 32 samples at 64×64

### Batching

For multiple objects, process sequentially:
```python
all_goals = {}
for obj in objects_to_check:
    all_goals[obj] = model.infer(json_message, xml_path, robot_goal, obj, samples=16)
```

## Troubleshooting

### Empty Results

1. Check selected_object name matches scene objects
2. Increase sample count
3. Verify model was trained on similar scenes

### Slow Inference

1. Reduce sample count
2. Use GPU (`device="cuda"`)
3. Flow Matching models are faster than Diffusion

### Invalid Poses

1. Check mask quality (save `goal_sample` for debugging)
2. Verify coordinate system alignment
3. Check object geometry extraction

## Integration Example

```python
# Full integration with NAMO planner
from ktamp_learning import GoalInferenceModel

class MLGoalStrategy:
    def __init__(self, model_path):
        self.model = GoalInferenceModel(model_path)

    def get_goal(self, env_state, selected_object, robot_goal):
        json_message = self._state_to_json(env_state)
        xml_path = env_state.xml_path

        goals = self.model.infer(
            json_message, xml_path, robot_goal,
            selected_object, samples=32
        )

        if not goals:
            return None

        # Return best goal
        return goals[0]['x'], goals[0]['y'], goals[0]['theta']
```
