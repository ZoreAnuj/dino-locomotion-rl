# Dino Locomotion RL

An open-source research platform for studying dinosaur biomechanics and robotic locomotion using reinforcement learning. This project provides MuJoCo-based Gymnasium environments to simulate and train locomotion policies, with a focus on curriculum learning and experiment tracking.

## Key Features
- Custom MuJoCo environments for dinosaur and robotic locomotion
- Curriculum learning for progressive skill acquisition
- Weights & Biases integration for experiment tracking
- Modular architecture for easy environment and algorithm extension

## Tech Stack
- Python, MuJoCo, Gymnasium
- Stable-Baselines3 (RL algorithms)
- Weights & Biases (experiment tracking)
- NumPy, SciPy, Matplotlib

## Getting Started
```bash
git clone https://github.com/zoreanuj/dino-locomotion-rl.git
cd dino-locomotion-rl
pip install -r requirements.txt
python train.py --env DinoBipedal-v0
```