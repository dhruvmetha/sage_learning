# CLAUDE.md - Sage Learning Codebase Guide

## Purpose

Goal prediction for NAMO (Navigation Among Movable Obstacles) using generative models. Given a scene and a selected object, predicts **where to push the object** to help the robot reach its goal.

**For detailed documentation, see [docs/](docs/)**

## Core Architecture

```
Input (5 channels)              Output (1 channel)
─────────────────              ─────────────────
robot position      ─┐
robot goal          ─┤
movable objects     ─┼──► GenerativeModule ──► target_goal mask
static obstacles    ─┤         (DiT)              (where to push)
selected object     ─┘
```

## Generative Approaches

### Flow Matching (Facebook)

Uses Facebook's `flow-matching` library. Model learns velocity field `v(x,t)`.

```bash
# Install (first time)
pip install flow-matching torchdyn

# Train
python src/train_generative.py --config-name=train_fb_flow_matching

# With adaptive ODE solver
python src/train_generative.py --config-name=train_fb_flow_matching model.sampler.method=dopri5
```

**How it works:**
- **Training**: Interpolate noise→data via `x_t = (1-t)*x_0 + t*x_1`, model predicts velocity `v = dx/dt`
- **Inference**: Integrate velocity field with ODE solver (euler/midpoint/rk4/dopri5)
- **Steps**: ~20 (fast)

**Schedulers available** (via `model.path.scheduler=`):
- `condot` - Conditional Optimal Transport (default)
- `cosine` - Cosine annealing
- `vp` - Variance Preserving
- `linear_vp` - Linear Variance Preserving

### Diffusion (HuggingFace)

Uses HuggingFace's `diffusers` library. Model learns to predict noise `ε`.

```bash
# Install (first time)
pip install diffusers

# DDIM (deterministic, faster) - default
python src/train_generative.py --config-name=train_hf_diffusion

# DDPM (stochastic)
python src/train_generative.py --config-name=train_hf_diffusion model.sampler.sampler_type=ddpm

# Fewer inference steps (DDIM can use 20-50)
python src/train_generative.py --config-name=train_hf_diffusion sampling_steps=50
```

**How it works:**
- **Training**: Add noise via `x_t = √ᾱ*x_1 + √(1-ᾱ)*ε`, model predicts noise `ε`
- **Inference**: Iteratively denoise (DDPM=stochastic, DDIM=deterministic)
- **Steps**: ~50 for DDIM, ~100 for DDPM

**Beta schedules available** (via `model.path.beta_schedule=`):
- `squaredcos_cap_v2` - Squared cosine with cap (default, recommended)
- `linear` - Linear schedule
- `scaled_linear` - Scaled linear (Stable Diffusion style)
- `sigmoid` - Sigmoid schedule

**DDPM vs DDIM:**
| | DDPM | DDIM |
|---|---|---|
| Sampling | Stochastic | Deterministic |
| Config | `sampler_type: ddpm` | `sampler_type: ddim` |
| Steps | ~100 | ~50 (can skip steps) |

## Training Configurations

| Config | Method | Library | Inference Steps |
|--------|--------|---------|-----------------|
| `train_fb_flow_matching.yaml` | Flow Matching | Facebook | ~20 |
| `train_hf_diffusion.yaml` | DDPM/DDIM | HuggingFace | ~50 |

## File Structure

```
sage_learning/
├── src/
│   ├── train_generative.py        # Training script
│   └── model/
│       ├── generative_module.py   # Unified training module
│       ├── paths/
│       │   ├── fb_flow_matching_path.py  # Facebook flow matching
│       │   └── hf_diffusion_path.py      # HuggingFace diffusion
│       ├── samplers/
│       │   ├── fb_ode_sampler.py         # Facebook ODE solver
│       │   └── hf_diffusion_sampler.py   # HuggingFace DDPM/DDIM
│       └── dit/dit.py                    # DiT network
├── config/
│   ├── train_*.yaml                # Training configs
│   ├── model/                      # Model configs
│   ├── trainer/{single,multi}_gpu.yaml
│   ├── path/                       # Path configs
│   └── sampler/                    # Sampler configs
└── ktamp_learning/
    └── goal_inference_model.py     # Inference API
```

## Common Commands

```bash
# Flow Matching (recommended)
python src/train_generative.py --config-name=train_fb_flow_matching

# Diffusion
python src/train_generative.py --config-name=train_hf_diffusion

# Multi-GPU
python src/train_generative.py --config-name=train_fb_flow_matching trainer=multi_gpu

# Override scheduler (flow matching)
python src/train_generative.py --config-name=train_fb_flow_matching model.path.scheduler=cosine

# Override sampler method (flow matching)
python src/train_generative.py --config-name=train_fb_flow_matching model.sampler.method=dopri5

# Override sampler type (diffusion: ddim or ddpm)
python src/train_generative.py --config-name=train_hf_diffusion model.sampler.sampler_type=ddpm
```

## Inference API

```python
from ktamp_learning import GoalInferenceModel

model = GoalInferenceModel("outputs/run_path/")
goals = model.infer(json_message, xml_path, robot_goal, selected_object="box1", samples=32)
# Returns: [{'x': 1.5, 'y': 2.0, 'theta': 0.3, ...}, ...]
```

## Dependencies

**Core:**
- torch, torchvision, lightning, hydra-core, wandb

**Setup:**
```bash
# Logging
pip install wandb
wandb login

# For Flow Matching
pip install flow-matching torchdyn

# For Diffusion
pip install diffusers
```

## See Also

- [docs/training.md](docs/training.md) - Training guide
- [docs/inference.md](docs/inference.md) - Inference guide
- [docs/architecture.md](docs/architecture.md) - Architecture details
