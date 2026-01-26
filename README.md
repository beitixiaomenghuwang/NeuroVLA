# NeuroVLA: A Lego-like Codebase for Vision-Language-Action Models

<div align="center">

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.8+](https://img.shields.io/badge/pytorch-2.8+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**A modular and extensible framework for developing Vision-Language-Action models for robotic manipulation**

[Installation](#installation) • [Quick Start](#quick-start) • [Documentation](#documentation) • [Examples](#examples) • [Citation](#citation)

</div>

---

## Overview

NeuroVLA is a flexible, modular codebase for building and training Vision-Language-Action (VLA) models for robotic manipulation. It features:

- 🧩 **Modular Design**: Lego-like architecture for easy component swapping and experimentation
- 🧠 **Spiking Neural Networks**: Novel SNN-based action prediction with temporal processing
- 🎯 **State-Aware Conditioning**: FiLM and GRU-based state modulation for precise control
- 🚀 **Multiple Backends**: Support for various VLM backbones (Qwen-VL, InternVL, etc.)
- 📊 **Comprehensive Evaluation**: Built-in support for LIBERO, SimplerEnv, and real robot deployment
- ⚡ **Efficient Training**: DeepSpeed integration with ZeRO optimization

## Architecture

NeuroVLA combines vision-language models with action prediction through a modular pipeline:

```
Images + Instructions → VLM Encoder → Q-Former → State Modulator → Action Predictor → Robot Actions
                                                        ↑
                                                   Robot States
```

### Key Components

- **VLM Interface**: Qwen-VL, Qwen2.5-VL for vision-language understanding
- **Q-Former**: Extracts action-relevant features from VLM hidden states
- **State Modulator**: FiLM or GRU-gated modulation conditioned on robot states
- **Action Predictor**: SNN-based temporal action prediction with LIF neurons

## Installation

### Prerequisites

- Python >= 3.10
- CUDA >= 11.8 (for GPU support)
- PyTorch >= 2.8

### Install from Source

```bash
# Clone the repository
git clone https://github.com/yourusername/NeuroVLA.git
cd NeuroVLA

# Install dependencies
pip install -r requirements.txt

# Install NeuroVLA in editable mode
pip install -e .
```

### Key Dependencies

```
torch>=2.8.0
transformers>=4.57.0
deepspeed>=0.16.9
snntorch>=0.9.1
qwen-vl-utils
```

## Quick Start

### Training

Train NeuroVLA on LIBERO benchmark:

```bash
# Single GPU training
python NeuroVLA/training/train_NeuroVLA.py \
    --config NeuroVLA/config/training/internvla_cotrain_libero.yaml

# Multi-GPU training with DeepSpeed
bash scripts/run_scripts/run_libero_train_yibu.sh
```

### Inference

Load a pretrained model and predict actions:

```python
import torch
from NeuroVLA.model.framework.NeuroVLA_yibu import NeuroVLA

# Load model
device = torch.device("cuda:0")
model = NeuroVLA.from_pretrained("path/to/checkpoint.pt").to(device)

# Prepare inputs
images = [...]  # List of PIL Images
instructions = ["pick up the red block"]
states = [...]  # Robot state history [B, T, 8]

# Predict actions
with torch.inference_mode():
    result = model.predict_action(
        batch_images=images,
        instructions=instructions,
        states=states,
    )
    actions = result["normalized_actions"]
```

## Project Structure

```
NeuroVLA/
├── NeuroVLA/
│   ├── config/              # Configuration files
│   │   ├── training/        # Training configs (LIBERO, OXE, etc.)
│   │   └── deepseeds/       # DeepSpeed configs
│   ├── dataloader/          # Data loading modules
│   │   ├── lerobot_datasets.py
│   │   └── gr00t_lerobot/   # GR00T/LeRobot integration
│   ├── model/
│   │   ├── framework/       # Main model frameworks
│   │   │   ├── NeuroVLA_yibu.py      # Main NeuroVLA model
│   │   │   └── base_framework.py     # Base framework class
│   │   └── modules/
│   │       ├── vlm/         # Vision-language models
│   │       ├── action_model/  # Action prediction heads
│   │       │   └── spike_action_model_multitimestep.py
│   │       ├── projector/   # Q-Former and projectors
│   │       └── dino_model/  # DINO vision encoder
│   └── training/            # Training scripts
├── deployment/              # Model server and deployment
│   └── model_server/        # WebSocket policy server
├── examples/
│   ├── LIBERO/             # LIBERO evaluation
│   ├── SimplerEnv/         # SimplerEnv evaluation
│   └── real_robot/         # Real robot deployment
└── scripts/                # Training and evaluation scripts
```

## Examples

### LIBERO Evaluation

Evaluate on LIBERO benchmark tasks:

```bash
cd examples/LIBERO

# Start model server
bash run_server.sh

# Run evaluation
bash eval_libero.sh
```

See [examples/LIBERO/README.md](examples/LIBERO/README.md) for detailed instructions.

### SimplerEnv Evaluation

Evaluate on SimplerEnv simulation:

```bash
cd examples/SimplerEnv
python start_simpler_env.py --checkpoint path/to/checkpoint.pt
```

See [examples/SimplerEnv/README.md](examples/SimplerEnv/README.md) for more details.

### Real Robot Deployment

Deploy on physical robots:

```bash
cd examples/real_robot
# Follow setup instructions in README.md
```

See [examples/real_robot/README.md](examples/real_robot/README.md) for hardware setup and deployment guide.

## Model Zoo

| Model | Dataset | Success Rate | Checkpoint |
|-------|---------|--------------|------------|
| NeuroVLA-Base | LIBERO-90 | TBD | Coming soon |
| NeuroVLA-Large | OXE | TBD | Coming soon |

## Configuration

NeuroVLA uses YAML configuration files for training. Key configuration options:

```yaml
framework:
  name: "NeuroVLA_yibu"
  layer_qformer:
    qformer_start_layer: -6
    qformer_end_layer: -1
    num_query_tokens: 8

training:
  batch_size: 32
  learning_rate: 1e-4
  num_epochs: 100

deepspeed:
  config: "config/deepseeds/deepspeed_zero2.yaml"
```

See [NeuroVLA/config/training/](NeuroVLA/config/training/) for example configurations.

## Development

### Code Style

We use `black` and `ruff` for code formatting:

```bash
# Format code
make autoformat

# Check code style
make check

# Clean cache files
make clean
```

### Design Philosophy

NeuroVLA follows a "Lego-like" design philosophy with:

- **Modularity**: Easy to swap components (VLM, action head, etc.)
- **Extensibility**: Simple to add new models and datasets
- **Clarity**: Clear separation of concerns and well-documented code

See [assets/intro_v1.md](assets/intro_v1.md) for detailed design principles.

## Documentation

- [Design Philosophy](assets/intro_v1.md) - Detailed architecture and conventions
- [LIBERO Evaluation](examples/LIBERO/README.md) - LIBERO benchmark guide
- [SimplerEnv Evaluation](examples/SimplerEnv/README.md) - SimplerEnv setup
- [Real Robot Deployment](examples/real_robot/README.md) - Hardware deployment
- [Model Server](deployment/model_server/README.md) - WebSocket server usage

## Contributing

We welcome contributions! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please ensure your code follows our style guidelines and includes appropriate tests.

## Citation

If you find NeuroVLA useful in your research, please cite:

```bibtex
@software{neurovla2024,
  title={NeuroVLA: A Lego-like Codebase for Vision-Language-Action Models},
  author={Ye, Jinhui and Wang, Fangjing and Yu, Junqiu},
  year={2024},
  url={https://github.com/yourusername/NeuroVLA}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built upon [StarVLA](https://github.com/yourusername/StarVLA) codebase
- Inspired by [OpenVLA](https://github.com/openvla/openvla) and [GR00T](https://github.com/NVlabs/GR00T)
- Uses [Qwen-VL](https://github.com/QwenLM/Qwen-VL) for vision-language understanding
- Evaluation on [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) and [SimplerEnv](https://github.com/simpler-env/SimplerEnv)

## Contact

- **Jinhui Ye** - jinhuiyes@gmail.com
- **Fangjing Wang** - fangjing_wang@outlook.com
- **Junqiu Yu** - michaelyu1101@163.com

For questions and discussions, please open an issue on GitHub.

---

<div align="center">
Made with ❤️ by the NeuroVLA Team
</div>
