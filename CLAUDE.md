# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a robotics research codebase for learning-based knowledge transfer in task and motion planning (KTAMP), specifically focused on diffusion models for robotic manipulation tasks. The system uses PyTorch Lightning, Hydra configuration management, and ZMQ for distributed inference.

## Infrastructure Notes

**Shared Filesystem**: All *.cs.rutgers.edu machines (arrakis, westeros, etc.) share the same filesystem. Files saved on one machine are immediately accessible on all other machines at the same paths.

## Key Commands

### Training
```bash
# Always activate mjxrl environment first, then run commands
source /koko/system/anaconda/etc/profile.d/conda.sh && conda activate /common/users/dm1487/envs/mjxrl

# Train diffusion model with default configuration
python src/train_diffusion.py

# Train with specific GPU
python src/train_diffusion.py gpu_id=2

# Resume training from checkpoint
python src/train_diffusion.py +run_path=outputs/2025-07-17/00-16-11

# Train VAE model
python src/train_vae.py
```

### Inference and Evaluation
```bash
# Always activate mjxrl environment first, then run commands
source /koko/system/anaconda/etc/profile.d/conda.sh && conda activate /common/users/dm1487/envs/mjxrl

# Start ZMQ inference server
python src/inference_diffusion_zmq.py run_path=outputs/2025-07-17/00-16-11

# Split inference with multiple workers
python src/split_inference_diffusion_zmq.py

# Automated evaluation pipeline (auto-detect latest models)
python src/auto_eval.py

# Evaluate specific model runs with custom trial count
python src/auto_eval.py model_runs=["outputs/2025-07-16/22-39-31"] num_trials=1

# Manual coordination mode (generates instruction files)
python src/auto_eval.py coordination=manual
```

### Testing and Validation
```bash
# Always activate mjxrl environment first, then run commands
source /koko/system/anaconda/etc/profile.d/conda.sh && conda activate /common/users/dm1487/envs/mjxrl

# Test pipeline components
python test_pipeline.py

# Setup check for arrakis server
bash run_on_arrakis.sh
```

### Package Installation
```bash
# Install in development mode
pip install -e .
```

### Environment Setup
```bash
# Activate required conda environment
source /koko/system/anaconda/etc/profile.d/conda.sh && conda activate /common/users/dm1487/envs/mjxrl
```

## Architecture Overview

### Core Components

**Data Pipeline (`src/data/`)**
- `mask_diffusion_data.py`: Dataset handling for diffusion training with mask processing
- `mask_vae_data.py`: VAE-specific data processing
- Supports coordinate grids and various data augmentations

**Models (`src/model/`)**
- `diffusion_unetcond_module.py`: Conditional U-Net diffusion model with encoder
- `diffusion_unet_module.py`: Standard U-Net implementation
- `diffusion_module.py`: Base diffusion model class
- `dit/`: Diffusion Transformer models including relational variants
- `vae_module.py`: Variational autoencoder for latent space learning

**Inference System**
- `inference_diffusion_zmq.py`: ZMQ-based inference server for distributed evaluation
- `split_inference_diffusion_zmq.py`: Multi-worker inference for better throughput
- Uses JSON-to-image conversion for robotics task representations

**Automation Pipeline (`src/utils/`)**
- `gpu_manager.py`: GPU allocation and resource management
- `ssh_coordinator.py`: Remote execution coordination on evaluation servers
- `results_aggregator.py`: Results collection and analysis
- `auto_eval.py`: End-to-end automated evaluation workflow

### Configuration System

Uses Hydra with YAML configs in `config/`:
- `train_diffusion.yaml`: Main training configuration
- `model/`: Model architecture configs (diffusion_dit, diffusion_unet, diffusion_unetcond, vae)
- `data/`: Dataset configurations
- `trainer/`: Lightning trainer settings (default, gpu)
- `eval_environments/`: Environment sets for evaluation (standard, test)
- `auto_eval.yaml`: Automated evaluation pipeline settings

### Key Data Flow

1. **Training**: Data loaded via mask_diffusion_data → Model training with Lightning → Checkpoints saved to `outputs/YYYY-MM-DD/HH-MM-SS/`
2. **Inference**: Load checkpoint → Start ZMQ server → Process JSON requests → Return generated samples
3. **Evaluation**: Auto-detect models → Start inference servers → Coordinate remote evaluation → Aggregate results

## Environment Setup

**CRITICAL**: The `mjxrl` conda environment MUST be activated before running ANY Python script in this project.

**Required Environment**: `mjxrl` conda environment at `/common/users/dm1487/envs/mjxrl`
**GPU Requirements**: CUDA-compatible GPUs (automatically managed via gpu_manager)
**Dependencies**: PyTorch Lightning, Hydra, ZMQ, NumPy, OpenCV

**Always activate environment first**:
```bash
source /koko/system/anaconda/etc/profile.d/conda.sh && conda activate /common/users/dm1487/envs/mjxrl
```

## Output Structure

- `outputs/YYYY-MM-DD/HH-MM-SS/`: Training run outputs
  - `checkpoints/`: Model checkpoints (epoch=*.ckpt, last.ckpt)
  - `logs/`: TensorBoard logs
  - `*_results/`: Evaluation results with images and metrics
- `eval_results/`: Aggregated evaluation reports

## Development Notes

- Configuration uses Hydra's hierarchical composition
- GPU allocation is automatic but can be overridden with `gpu_id` parameter
- ZMQ inference supports concurrent requests with port management (5556-5570 range)
- Evaluation pipeline coordinates between arrakis (inference) and westeros (evaluation)
- Model architectures support both standard and relational attention mechanisms
- All *.cs.rutgers.edu servers share the same filesystem, so files are accessible across machines
- **CRITICAL**: Each evaluation server can only talk to ONE inference server (1:1 mapping). Only when evaluation server A finishes with an inference server can evaluation server B use that inference server
- **EVALUATION PIPELINE ANALYSIS** (Pinned for later discussion): Current auto_eval.py works correctly with 1:1 mapping. Uses 3 parallel inference servers, each processes one model's full evaluation sequence (current→easy→medium envs). Task queue approach considered but deemed unnecessary - current sequential-per-model approach is efficient and simpler. Each server stays busy with its assigned model's evaluations.
- **HYDRA EXPERIMENT NAMING SETUP** (Ready to test): Implemented better experiment organization with Hydra. Modified train_diffusion.yaml and model configs to automatically name experiments by architecture and loss type. Output structure: `outputs/{model_name}/{loss_type}/{timestamp}/`. Usage: `python src/train_diffusion.py model=diffusion_unetcond loss_type=dice`. NEEDS TESTING - suggest testing this in future sessions.