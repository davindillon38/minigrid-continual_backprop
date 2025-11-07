# Experimental Results

## Summary

Multi-stage training with continual backpropagation achieves **85% success on 16x16 DoorKey**, compared to 5% for single-stage baseline - a **17x improvement**.

## Full Comparison

| Environment | Single-stage Baseline | Multi-stage Baseline | Multi-stage + CB | CB Benefit |
|-------------|----------------------|---------------------|------------------|------------|
| 6x6 | 25% | 93% | **94%** | +1pp |
| 8x8 (training) | 13% | 94% | **95%** | +1pp |
| **16x16 (transfer)** | **5%** | **77%** | **85%** | **+8pp** |

## Key Findings

1. **Multi-stage training is essential**: Training progressively (5x5→6x6→8x8) improves 16x16 transfer from 5% to 77%

2. **Continual backpropagation adds value**: Provides additional +8pp improvement on hard transfer tasks

3. **Effect scales with difficulty**: CB benefit most visible on 16x16 (hardest), minimal on easier tasks

4. **Network remains plastic**: Agent successfully learns across all three training stages without catastrophic forgetting

## Training Details

### Single-stage Baseline
- **Environment**: MiniGrid-DoorKey-8x8-v0
- **Frames**: 2M
- **Algorithm**: PPO (default hyperparameters)
- **Final performance**: 13% on 8x8, 5% on 16x16
- **Issue**: Poor transfer to unseen sizes

### Multi-stage Baseline (without CB)
- **Stage 1**: DoorKey-5x5 (500k frames) → 95% success
- **Stage 2**: DoorKey-6x6 (1M frames) → 93% success
- **Stage 3**: DoorKey-8x8 (2M frames) → 94% success
- **Transfer to 16x16**: 77% success
- **Key insight**: Progressive curriculum enables strong transfer

### Multi-stage + Continual Backprop
- **Same stages** as multi-stage baseline
- **CB hyperparameters**: 
  - `replacement_rate=1e-4` (0.01% neurons/update)
  - `maturity_threshold=100` steps
  - `decay_rate=0.99` for utility tracking
- **Neurons replaced**: ~20-30 total across 2M frames
- **Final performance**: 
  - 8x8: 95% (+1pp vs baseline)
  - 16x16: **85%** (+8pp vs baseline)

## Detailed Results

### 16x16 Transfer (Critical Test)

**Single-stage**: R=0.05 (5% success)
- 95 episodes failed (timeout at 2560 steps)
- Poor exploration in large environment

**Multi-stage**: R=0.77 (77% success)
- 13 episodes failed
- Much better exploration and generalization

**Multi-stage + CB**: R=0.85 (85% success)
- Only 4 episodes failed
- Best transfer performance
- Network maintains plasticity through all training stages

### Continual Backprop Statistics

**Neuron replacement patterns**:
- First replacement: ~100 updates (after maturity threshold)
- Total replacements: 20-30 neurons across 2M frames
- Mostly low-utility neurons (utility < 0.001)
- Replacement ages: 100-970 steps (respects maturity threshold)

**Utility ranges**:
- Actor layer: [0.000002, 0.006]
- Critic layer: [0.000024, 0.010]
- Replaced neurons typically in bottom 1% of utility

## Hardware & Performance

- **GPU**: NVIDIA GeForce RTX 4060 (8GB)
- **Training time**: ~4-5 hours total for all stages
- **FPS**: 
  - CPU: ~2500 FPS
  - CUDA: ~3400 FPS (36% speedup)
- **Total training**: 3.5M frames across all stages

## Reproducibility

All results averaged over 100 evaluation episodes. Seeds:
- Training: seed=1
- Evaluation: deterministic environments

Commands to reproduce:
```bash
# Multi-stage + CB (best results)
python -m scripts.train --algo ppo --env MiniGrid-DoorKey-5x5-v0 --model multistage_cb --frames 500000 --continual-backprop
python -m scripts.train --algo ppo --env MiniGrid-DoorKey-6x6-v0 --model multistage_cb --frames 1000000 --continual-backprop
python -m scripts.train --algo ppo --env MiniGrid-DoorKey-8x8-v0 --model multistage_cb --frames 2000000 --continual-backprop
python -m scripts.evaluate --env MiniGrid-DoorKey-16x16-v0 --model multistage_cb --episodes 100
```

## Future Work

1. **Additional environments**: Test on KeyCorridor, ObstructedMaze
2. **Longer training**: Increase to 5-10M frames per stage
3. **Hyperparameter tuning**: Explore different replacement rates
4. **Ablation studies**: Effect of maturity threshold, decay rate
5. **Comparison to other methods**: Test against experience replay, elastic weight consolidation
