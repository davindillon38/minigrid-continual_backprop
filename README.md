# Continual Backpropagation for Transfer Learning in MiniGrid

Implementation of continual backpropagation ([Dohare et al., 2024](https://www.nature.com/articles/s41586-024-07711-7)) applied to multi-stage reinforcement learning in MiniGrid environments.

## Key Results

Continual backpropagation dramatically improves transfer learning performance:
- **16x16 transfer**: 5% → 85% success rate (17x improvement)
- Multi-stage training (5x5→6x6→8x8) essential for transfer
- Continual backprop adds +8pp on hard transfer tasks

See [RESULTS.md](RESULTS.md) for full details.

## Installation

```bash
pip install -r requirements.txt
```

Requires: PyTorch, gym-minigrid, tensorboardX

## Quick Start

### Train with continual backpropagation (multi-stage):
```bash
# Stage 1: 5x5
python -m scripts.train --algo ppo --env MiniGrid-DoorKey-5x5-v0 --model multistage_cb --frames 500000 --continual-backprop

# Stage 2: 6x6
python -m scripts.train --algo ppo --env MiniGrid-DoorKey-6x6-v0 --model multistage_cb --frames 1000000 --continual-backprop

# Stage 3: 8x8
python -m scripts.train --algo ppo --env MiniGrid-DoorKey-8x8-v0 --model multistage_cb --frames 2000000 --continual-backprop
```

### Evaluate transfer:
```bash
python -m scripts.evaluate --env MiniGrid-DoorKey-16x16-v0 --model multistage_cb --episodes 100
```

## Method

Continual backpropagation maintains network plasticity by:
1. Tracking neuron utility (activation × outgoing weight magnitude)
2. Periodically reinitializing low-utility neurons
3. Protecting recently-reset neurons (maturity threshold)

Key hyperparameters:
- `replacement_rate=1e-4` (0.01% of neurons per update)
- `maturity_threshold=100` (protection period)
- `decay_rate=0.99` (utility exponential moving average)

## Files

- `continual_backprop.py` - Continual backpropagation implementation
- `model.py` - Modified ACModel with activation tracking
- `scripts/train.py` - Training script with CB integration
- `scripts/evaluate.py` - Evaluation script

## Citation

Based on:
```
Dohare et al. (2024). Loss of Plasticity in Deep Continual Learning. Nature.
```

## Repository Structure

```
minigrid-continual-backprop/
├── README.md
├── RESULTS.md
├── requirements.txt
├── continual_backprop.py
├── model.py
├── scripts/
│   ├── train.py
│   └── evaluate.py
├── utils/
└── torch-ac/
```
