# Automated Evaluation Pipeline Usage

This document explains how to use the automated evaluation pipeline for your diffusion models.

## Overview

The pipeline automates the complete evaluation workflow:
1. **Auto-detect** completed training runs or use specified models
2. **Start inference servers** with automatic GPU/port allocation
3. **Coordinate evaluations** on westeros.cs.rutgers.edu (SSH or manual)
4. **Aggregate results** and generate comparison reports

## Quick Start

### 1. Auto-detect and evaluate latest models
```bash
python src/auto_eval.py
```

### 2. Evaluate specific model runs
```bash
python src/auto_eval.py model_runs=["outputs/2025-07-17/00-16-11","outputs/2025-07-16/23-57-38"]
```

### 3. Use manual coordination (generate instruction files)
```bash
python src/auto_eval.py coordination=manual
```

### 4. Customize evaluation settings
```bash
python src/auto_eval.py num_trials=100 max_parallel_servers=2 start_port=5560
```

## Configuration

### Main Config: `config/auto_eval.yaml`
- `model_runs`: List of specific model runs to evaluate (empty = auto-detect)
- `auto_detect_latest`: Enable auto-detection of latest training runs
- `max_models_to_detect`: Maximum number of models to auto-detect
- `num_trials`: Number of trials per environment (default: 50)
- `coordination`: "ssh" for automatic execution, "manual" for instruction files
- `max_parallel_servers`: Maximum number of inference servers to run simultaneously
- `available_gpus`: Available GPU IDs on arrakis
- `max_gpus_to_use`: Maximum GPUs to use for inference

### Environment Config: `config/eval_environments/standard.yaml`
- `current`: Current test set (from eval_namo.py)
- `easy`, `medium`, `hard`: Environment difficulty categories
- `ood`: Out-of-distribution test environments
- `default_sets`: Which environment sets to run by default

## Coordination Modes

### SSH Mode (Automatic)
- Pipeline automatically executes evaluations on westeros
- Requires SSH access: `ssh westeros.cs.rutgers.edu` (no password needed on *.cs.rutgers.edu)
- Monitors evaluation progress and waits for completion

### Manual Mode 
- Pipeline generates instruction files with commands to run
- You manually execute commands on westeros
- Pipeline waits for results files to appear

## Results

Results are saved to `eval_results/YYYY-MM-DD_HH-MM-SS_evaluation_batch/`:
- `summary.json`: Complete aggregated statistics
- `model_comparison.csv`: Cross-model comparison table
- `summary_report.md`: Human-readable summary
- `{model_name}/`: Per-model detailed results

### Key Metrics
- **Success Rate**: Percentage of successful trials
- **Average Action Steps**: Mean steps for successful trials
- **Reachable Selection Rate**: How often model selects reachable objects

## Example Workflows

### Evaluate after training completes
```bash
# Training completes, then run:
python src/auto_eval.py auto_detect_latest=true max_models_to_detect=1
```

### Compare multiple models
```bash
python src/auto_eval.py model_runs=["outputs/2025-07-17/00-16-11","outputs/2025-07-16/23-57-38","outputs/2025-07-14/01-59-56"]
```

### Run on specific environment sets
```bash
# Edit config/eval_environments/standard.yaml to change default_sets
python src/auto_eval.py
```

### Test with fewer trials for quick validation
```bash
python src/auto_eval.py num_trials=10
```

## Resource Management

The pipeline automatically:
- **Allocates GPUs**: Checks memory usage and assigns free GPUs
- **Manages ports**: Finds free ports in range 5556-5570
- **Prevents conflicts**: Limits concurrent inference servers
- **Cleanup**: Terminates processes and releases resources on exit

## Troubleshooting

### SSH Connection Issues
```bash
# Test SSH manually
ssh westeros.cs.rutgers.edu echo "test"

# Check ML4KP path
ssh westeros.cs.rutgers.edu ls /common/home/dm1487/robotics_research/ktamp/ml4kp_ktamp
```

### GPU Allocation Issues
- Check `nvidia-smi` for GPU memory usage
- Reduce `max_gpus_to_use` if needed
- Ensure no training processes are using GPUs

### No Models Found
- Check `outputs/` directory structure
- Ensure training runs have `checkpoints/epoch=*.ckpt` files
- Use explicit `model_runs` instead of auto-detection

### Evaluation Failures
- Check inference server logs in pipeline output
- Verify westeros can reach inference servers
- Check that environment configs exist in XML files

## Advanced Usage

### Adding New Environment Sets
1. Edit `config/eval_environments/standard.yaml`
2. Add new environment list (e.g., `custom_hard: [...]`)
3. Update `default_sets: ["current", "custom_hard"]`

### Custom Result Directories
```bash
python src/auto_eval.py results_dir=my_custom_results
```

### Different ML4KP Path
```bash
python src/auto_eval.py ssh.ml4kp_path=/path/to/ml4kp_ktamp
```

## Integration with Training

To automatically run evaluations after training:
1. Add to your training script:
```python
# At end of train_diffusion.py
if training_completed:
    subprocess.run(["python", "src/auto_eval.py", "auto_detect_latest=true", "max_models_to_detect=1"])
```

2. Or use a separate monitoring script that watches for new training completions.