# TRC: Trust Region Conditional Value at Risk for Safe Reinforcement Learning

This is an official GitHub Repository for paper ([link](https://doi.org/10.1109/LRA.2022.3141829)):

- Dohyeong Kim and Songhwai Oh, “TRC: Trust region conditional value at risk for
safe reinforcement learning,” IEEE Robotics and Automation Letters,
vol. 7, no. 2, pp. 2621–2628, Apr. 2022.

- This paper proposes a CVaR-constrained safe RL method (called TRC).

## License

Distributed under the MIT License. See `LICENSE` for more information.

## requirement

- python 3.7 or greater
- gym
- mujoco-py (https://github.com/openai/mujoco-py)
- safety-gym (https://github.com/openai/safety-gym)
- stable-baselines3
- tensorflow-gpu==1.13.1
- tensorflow-probability==0.6.0
- torch==1.10.0
- requests
- wandb

## How to use

### tf1

- training:

  - ```bash
    cd tf1
    bash train_{env_name}.sh
    ```

- test:

  - ```bash
    cd tf1
    bash test.sh
    ```

### torch

- training:

  - ```bash
    cd torch
    bash train_{env_name}.sh
    ```

- test:

  - ```bash
    cd torch
    bash test.sh
    ```

### Supported environments

- Safety-Gym: `Safexp-PointGoal1-v0`, `Safexp-PointGoal1-v0`, `Doggo-v0` (which is a hierarchical version of `Safexp-DoggoGoal1-v0`)

- MuJoCo: `Jackal-v0`
