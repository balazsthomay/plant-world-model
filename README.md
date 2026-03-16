# Learned World Model for Industrial Process Control

Given only historical sensor data from an industrial process, this project learns a neural dynamics model that predicts how the plant state evolves under control actions, wraps it as a standard Gymnasium environment, and trains an RL agent entirely inside the learned simulator. The key result: the agent transfers to the ground truth environment with near-zero performance gap, demonstrating that logged data alone can replace first-principles modeling for RL-based process control.

## Architecture

```
Historical Sensor Data          Learned Dynamics Model            RL Agent Training
(state, action, next_state)     (Ensemble of Probabilistic MLPs)  (SAC via Stable-Baselines3)
         |                               |                                |
         v                               v                                v
   Data Pipeline ──────> Train Ensemble ──────> Gymnasium Env ──────> Sim-to-Real Eval
   (normalize,            (bootstrap,          (step = model          (train in learned,
    temporal split)        early stopping)       prediction)            test in ground truth)
```

## Key Results

### The learned model faithfully reproduces CSTR dynamics over 100-step rollouts

The ensemble of 5 probabilistic MLPs learns to predict concentration (Ca) and temperature (T) transitions in a Continuously Stirred Tank Reactor. Over 100-step open-loop rollouts, the learned model (red) tracks the ground truth simulator (blue) with high fidelity. Uncertainty bands grow over time as expected from compounding prediction errors.

![Rollout Comparison](figures/rollout_comparison.png)

### Prediction error stays low even at long horizons

Multi-step prediction error (MSE) is plotted against rollout horizon. Concentration prediction remains extremely accurate (~10<sup>-7</sup> MSE at 50 steps). Temperature shows more error growth due to the exothermic reaction dynamics, but stays manageable (~10<sup>-3</sup> MSE).

![Multi-Step Error](figures/multistep_error.png)

### The ensemble decomposes uncertainty into epistemic and aleatoric components

Epistemic uncertainty (model uncertainty, reducible with more data) and aleatoric uncertainty (inherent noise) are separately tracked. This decomposition lets the agent know when it's in well-modeled vs. unexplored regions of state space.

![Uncertainty Decomposition](figures/uncertainty_decomposition.png)

### An RL agent trained in the learned environment transfers to the ground truth

The sim-to-real transfer gap is the central evaluation: can a policy learned entirely in the neural simulator perform well in the real system? The answer is yes. The agent trained in the learned environment achieves equivalent performance when evaluated in the ground truth CSTR.

![Sim-to-Real Transfer Gap](figures/sim_to_real.png)

| Condition | Mean Episode Reward |
|-----------|-------------------|
| GT-trained, evaluated in GT | -0.53 |
| Learned-trained, evaluated in Learned | -0.55 |
| Learned-trained, evaluated in GT | -0.52 |

## Technical Details

**Dynamics Model**: Ensemble of 5 probabilistic MLPs (4 hidden layers of 200 units, SiLU activation). Each network outputs a Gaussian distribution over state deltas (next_state - current_state). Trained with Gaussian negative log-likelihood loss on bootstrap samples with early stopping. Epistemic uncertainty = variance of ensemble means; aleatoric uncertainty = mean of ensemble variances.

**Ground Truth**: PC-Gym CSTR environment (ODE-based reactor simulation with Arrhenius kinetics). State = [Ca, T], Action = [Tc (jacket temperature)]. Data collected with random, sinusoidal, and step-change action strategies for excitation diversity.

**RL**: SAC (Soft Actor-Critic) via Stable-Baselines3 with default hyperparameters. Reward = negative squared setpoint tracking error.

## How to Run

```bash
# Install dependencies
uv sync

# Run full pipeline (data collection -> training -> evaluation -> figures)
uv run python -m scripts.run_pipeline

# Run tests
uv run pytest --cov=src
```

## Project Structure

```
src/
  configs.py            # Dataclass configurations
  data_collection.py    # PC-Gym CSTR data collection
  dataset.py            # Normalization and PyTorch datasets
  dynamics_model.py     # Probabilistic MLP ensemble
  training.py           # Ensemble training with bootstrap + early stopping
  learned_env.py        # Gymnasium wrapper around learned model
  rl_evaluation.py      # SB3 training and sim-to-real comparison
  figures.py            # Visualization
tests/                  # 41 tests, 92% coverage
scripts/
  run_pipeline.py       # End-to-end pipeline
```

## Limitations and Next Steps

**What works well**: On the 2-state CSTR (a standard benchmark), the ensemble learns accurate dynamics and produces a simulator faithful enough for zero-gap policy transfer. The uncertainty decomposition correctly identifies when the model is extrapolating.

**What would need work for real deployment**:

- **Scale**: Industrial processes have 20-50+ state variables. The CSTR has 2. The architecture scales linearly in parameters but the data requirements and evaluation complexity grow significantly.
- **Partial observability**: Real plants have unmeasured state variables. A latent dynamics model (encoder + recurrent state) would be needed.
- **Non-stationarity**: Equipment degrades, feedstock changes. The model would need online adaptation or periodic retraining.
- **Data quality**: The CSTR data has perfect action variation because we control the simulator. Real plant data from operators running near steady-state may lack the excitation needed to learn dynamics — this is the fundamental data requirements question for learned world models.
- **Safety**: The learned environment should flag when the RL agent visits states far from the training distribution (the uncertainty estimates support this) and the policy should be validated against known constraints before deployment.
