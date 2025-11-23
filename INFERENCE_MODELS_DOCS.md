# Split Inference Models Documentation

## Overview

This package provides two specialized inference models for robotic manipulation tasks:

- **`ObjectInferenceModel`**: Selects which object to manipulate using diffusion-based voting
- **`GoalInferenceModel`**: Generates goal poses in SE(2) space for a selected object

Both models are extracted from the original ZMQ-based inference server and provide clean, reusable interfaces without networking dependencies.

## Architecture

```
json_message + xml_path + robot_goal
                    ↓
         [ObjectInferenceModel] ← Independent
                    ↓
            selected_object
                    ↓
json_message + xml_path + robot_goal + selected_object  
                    ↓
          [GoalInferenceModel] ← Independent
                    ↓
              goal_proposals
```

## Installation

### Prerequisites
- **Environment**: mjxrl conda environment
- **Models**: Trained diffusion models in outputs/ directory
- **Dependencies**: PyTorch, Hydra, OpenCV, NumPy

### Setup
```bash
# Activate environment
conda activate /common/users/dm1487/envs/mjxrl

# Install learning package
cd /path/to/learning
pip install -e .
```

## API Reference

### ObjectInferenceModel

#### Constructor
```python
ObjectInferenceModel(model_path: str, device: str = "cuda")
```

**Parameters:**
- `model_path`: Path to trained object model (e.g., `"outputs/rel_reach_coord_object_dit/mse/2025-08-10_05-33-43"`)
- `device`: PyTorch device ("cuda" or "cpu")

#### infer() Method
```python
infer(json_message: dict, xml_path: str, robot_goal: list, samples: int = 32) -> dict
```

**Parameters:**
- `json_message`: Environment state data (see Input Format below)
- `xml_path`: Relative path to MuJoCo XML file
- `robot_goal`: Target position as `[x, y]` coordinates
- `samples`: Number of diffusion samples to generate

**Returns:**
- `object_votes`: Counter with vote distribution across objects
- `selected_object`: Object name with most votes (str or None)
- `total_valid_samples`: Number of valid diffusion samples processed
- `reachable_selections`: Count of reachable object selections

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
    "reachable_objects": ["box1", "cylinder1", "block2"],  # Optional
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

### Complete Pipeline
```python
from learning.ktamp_learning.object_inference_model import ObjectInferenceModel
from learning.ktamp_learning.goal_inference_model import GoalInferenceModel

# Initialize models
object_model = ObjectInferenceModel("outputs/rel_reach_coord_object_dit/mse/2025-08-10_05-33-43")
goal_model = GoalInferenceModel("outputs/rel_reach_coord_goal_dit/mse/2025-08-10_06-59-27")

# Prepare environment data
json_message = {
    "xml_path": "custom_walled_envs/jun22/env_001.xml",
    "robot_goal": [1.5, 2.0],
    "reachable_objects": ["box1", "cylinder1"],
    "robot": {"position": [0.0, 0.0, 0.1]},
    "objects": {
        "box1": {
            "position": [1.2, 1.8, 0.05],
            "quaternion": [1.0, 0.0, 0.0, 0.0]
        },
        "cylinder1": {
            "position": [2.1, 0.5, 0.05],
            "quaternion": [0.9, 0.0, 0.0, 0.4]
        }
    }
}

# Object selection
try:
    object_result = object_model.infer(
        json_message=json_message,
        xml_path="custom_walled_envs/jun22/env_001.xml",
        robot_goal=[1.5, 2.0],
        samples=32
    )
    
    if not object_result['selected_object']:
        print("❌ No valid object found")
        exit(1)
    
    print(f"🎯 Selected: {object_result['selected_object']}")
    print(f"📊 Votes: {dict(object_result['object_votes'])}")
    print(f"✅ Valid samples: {object_result['total_valid_samples']}")
    print(f"🎪 Reachable selections: {object_result['reachable_selections']}")
    
    # Goal generation - now independent!
    goals = goal_model.infer(
        json_message=json_message,
        xml_path="custom_walled_envs/jun22/env_001.xml",
        robot_goal=[1.5, 2.0],
        selected_object=object_result['selected_object'],
        samples=32
    )
    
    if not goals:
        print("❌ No valid goals generated")
        exit(1)
        
    print(f"🎯 Generated {len(goals)} goal proposals:")
    for i, goal in enumerate(goals[:5]):  # Show first 5
        print(f"  Goal {i}: pos=({goal['x']:.3f}, {goal['y']:.3f}), θ={goal['theta']:.3f}")
        
except Exception as e:
    print(f"❌ Error: {e}")
```

### Object Selection Only
```python
from learning.ktamp_learning.object_inference_model import ObjectInferenceModel

object_model = ObjectInferenceModel("outputs/object_model_path")

object_result = object_model.infer(json_message, xml_path, robot_goal)

# Analyze object selection results
votes = object_result['object_votes']
total_votes = sum(votes.values())

print("Object Selection Results:")
for obj, count in votes.most_common():
    percentage = (count / total_votes) * 100
    print(f"  {obj}: {count} votes ({percentage:.1f}%)")
```

### Goal Generation Only (with known object)
```python
from learning.ktamp_learning.goal_inference_model import GoalInferenceModel

goal_model = GoalInferenceModel("outputs/goal_model_path")

# Generate goals for specific object - now fully independent!
goals = goal_model.infer(
    json_message=json_message,
    xml_path="custom_walled_envs/jun22/env_001.xml", 
    robot_goal=[1.5, 2.0],
    selected_object="box1",  # Known object
    samples=64  # More samples for better coverage
)

# Analyze goals
print(f"Generated {len(goals)} goals for box1:")
for goal in goals:
    print(f"  x={goal['x']:.3f}, y={goal['y']:.3f}, θ={goal['theta']:.3f}")
```

## Key Benefits of Independent Design

### True Model Independence
- ✅ **ObjectInferenceModel**: Works standalone for object selection experiments
- ✅ **GoalInferenceModel**: Works standalone when you already know which object to use  
- ✅ **Same Interface**: Both models take the same raw inputs (json_message, xml_path, robot_goal)
- ✅ **No Coupling**: No need to run object inference to use goal inference

### Flexible Usage Patterns
```python
# Pattern 1: Full pipeline
object_result = object_model.infer(json_message, xml_path, robot_goal)
goals = goal_model.infer(json_message, xml_path, robot_goal, object_result['selected_object'])

# Pattern 2: Only object selection
selected_object = object_model.infer(json_message, xml_path, robot_goal)['selected_object']

# Pattern 3: Only goal generation (with known object)
goals = goal_model.infer(json_message, xml_path, robot_goal, selected_object="box1")

# Pattern 4: Different objects for same environment
goals_box = goal_model.infer(json_message, xml_path, robot_goal, "box1") 
goals_cyl = goal_model.infer(json_message, xml_path, robot_goal, "cylinder1")
```

## Performance Considerations

### Memory Usage
- **Object Model**: ~2-4GB GPU memory (depending on architecture)
- **Goal Model**: ~2-4GB GPU memory (depending on architecture)
- **Combined**: ~4-8GB GPU memory
- **Tip**: Load only needed model to save memory

### Inference Speed
- **Object Inference**: ~100-200ms for 32 samples
- **Goal Inference**: ~100-200ms for 32 samples
- **Total Pipeline**: ~200-400ms end-to-end

### Sample Count Guidelines
- **Development/Testing**: 8-16 samples for faster iteration
- **Production**: 32 samples (matches original ZMQ server)
- **High Precision**: 64+ samples for better coverage

## Error Handling

### Common Errors
```python
try:
    object_result = object_model.infer(json_message, xml_path, robot_goal)
    goals = goal_model.infer(...)
    
except FileNotFoundError as e:
    print(f"Model or config file not found: {e}")
    
except ValueError as e:
    print(f"Invalid input data: {e}")
    
except RuntimeError as e:
    print(f"Model inference error (GPU/memory?): {e}")
    
except Exception as e:
    print(f"Unexpected error: {e}")
```

### Validation Checks
```python
# Validate object selection
if not object_result['selected_object']:
    print("⚠️ No object selected - check environment setup")
    
if object_result['total_valid_samples'] < 5:
    print("⚠️ Very few valid samples - model may be struggling")
    
# Validate goal generation  
if len(goals) < 3:
    print("⚠️ Few goals generated - increase sample count?")

# Check goal quality
goal_spread = max(g['x'] for g in goals) - min(g['x'] for g in goals)
if goal_spread < 0.1:
    print("⚠️ Goals are very clustered - model may be overconfident")
```

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
- Consider batch processing for multiple environments

## Technical Notes

### Coordinate Systems
- **Input**: World coordinates from MuJoCo simulation
- **Processing**: Pixel coordinates in 224x224 images
- **Output**: World coordinates converted back via ImageConverter

### Model Architecture
- Both models use diffusion-based generation
- Object model outputs object masks
- Goal model outputs goal masks in SE(2) space
- Vote-based selection for robust object choice

### Differences from Original ZMQ Server
- ✅ Removed SE(2) clustering bug (was calculated but ignored)
- ✅ Clean error handling with informative messages
- ✅ Modular design allows independent model usage
- ✅ Better memory management with model splitting
- ✅ Simplified interface without ZMQ dependencies

## Support

For issues or questions:
1. Check that environment variables and paths are correct
2. Verify model checkpoints are compatible with current code version
3. Ensure mjxrl conda environment is properly activated
4. Review CLAUDE.md for additional project-specific guidance