# 🧠 Neuron Runner - Advanced AI-Themed Endless Runner

**Repository Name:** `neuron-runner-ai-game`

**"Train. Dodge. Optimize. Learn."**

An innovative educational endless runner game that combines thrilling gameplay with deep learning concepts. Players control a neural network navigating through a world of AI challenges, learning about machine learning while having fun!

## � Key Features

### 🎨 **Model Skin System**
- **4 Unique AI Models**: CNN, RNN, Transformer, and GAN
- Each model has distinct visuals, abilities, and gameplay mechanics
- Special abilities: Speed Burst, Echo Jump, Teleportation, and Phase Shift

### 🏆 **Challenge Modes**
- **Data Shift**: Adapt to changing environments
- **Adversarial**: Survive hostile AI attacks
- **Limited Epoch**: Race against training time
- **Gradient Instability**: Navigate chaotic gradients

### 📊 **Live ML Statistics**
- Real-time accuracy, loss, and gradient health monitoring
- Visual ML metrics that affect gameplay
- Overfitting risk and batch normalization indicators

### 🔧 **Custom Model Builder**
- Drag-and-drop neural network architecture designer
- Choose layers: Dense, Conv2D, LSTM, Dropout, BatchNorm, Attention
- Configure optimizers, learning rates, and batch sizes
- Your custom model affects gameplay performance!

### 📚 **Educational AI Fact Cards**
- 20+ educational cards covering ML basics to advanced concepts
- Categories: ML Basics, Deep Learning, AI History, Modern AI, Ethics
- Interactive trivia system that appears during gameplay

### 📈 **Training Journal & Achievements**
- Persistent progress tracking and statistics
- 15+ unlockable achievements with different rarities
- Session history with detailed performance metrics
- Daily/weekly activity tracking

### 🌍 **Dynamic World Themes**
- Visual environment changes every 1000 points
- 6 unique AI domains: Training Lab, Neural Network, Deep Learning, etc.
- Theme-specific visual effects and particle systems
- Immersive atmosphere that evolves with your progress

### 🎵 **Evolving Synthwave Soundtrack**
- Dynamic music system that adapts to gameplay intensity
- Multiple layered tracks that blend based on game state
- Theme-specific musical elements and transitions
- Synthwave aesthetic with futuristic AI-themed sounds

### 📹 **Instant Replay + Share GIF**
- Automatic recording of last 10 seconds of gameplay
- One-click GIF generation with stats overlay
- Shareable highlights with watermarks and progress bars
- Multiple quality settings for different file sizes

### 🤖 **AI Agent Competition Mode**
- Train your own Q-learning bot to play the game
- Human vs AI challenge modes
- Bot leaderboard and performance tracking
- Watch AI agents learn and improve over time

### 🎓 **Educational Mode**
- Step-by-step tutorials for AI/ML concepts
- Interactive tooltips and guided learning
- Perfect for students and educators
- Learn while you play!

## 🎮 Game Concept
You are an AI neural network avatar running through the digital landscape of machine learning training. Navigate obstacles, collect powerups, and learn about AI while having fun!

## 🚀 Quick Start
1. Install dependencies: `pip install -r requirements.txt`
2. Run the game: `python main.py`

## 🎯 Controls

### Basic Gameplay:
- **SPACE**: Jump over obstacles
- **DOWN**: Duck/slide under barriers
- **L**: Layer Boost (special ability)
- **X**: Model-specific special ability

### Advanced Features:
- **B**: Open Model Builder
- **J**: View Training Journal
- **R**: Toggle Replay Recording
- **G**: Generate Replay GIF (when game over)
- **A**: Toggle AI Mode (human vs bot)
- **T**: Create AI Tournament
- **C**: Human vs AI Challenge
- **E**: Toggle Educational Mode
- **M**: Toggle Music Visualization
- **V**: Cycle ML Stats Dashboard View
- **F1**: Show/Hide Help Overlay

## 📊 Game Features
- Real-time loss curve visualization
- Epoch tracking (increases every 60 seconds)
- ML-themed obstacles and powerups
- Neural network avatar with layer system

## 🏗️ Project Structure
```
neuron-runner-ai-game/
├── main.py                 # Main game entry point
├── game/
│   ├── player.py          # Neural network player with model skins
│   ├── model_skins.py     # AI model theming system
│   ├── challenge_modes.py # Special gameplay challenges
│   ├── ml_stats.py        # Live ML statistics display
│   ├── model_builder.py   # Custom model creation tool
│   ├── fact_cards.py      # Educational AI trivia system
│   ├── training_journal.py# Progress tracking & achievements
│   ├── world_themes.py    # Dynamic visual environment system
│   ├── music_system.py    # Evolving synthwave soundtrack
│   ├── educational_mode.py# Tutorial and learning system
│   ├── replay_system.py   # Instant replay and GIF generation
│   ├── ai_competition.py  # AI bot training and competition
│   ├── obstacles.py       # AI-themed obstacles
│   ├── powerups.py        # ML powerups and bonuses
│   ├── particles.py       # Visual effects system
│   ├── utils.py           # Game utilities and UI
│   └── config.py          # Game configuration
├── assets/                # Game assets (when added)
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🎓 **Educational Value**

This game serves as an interactive introduction to:
- Neural network architectures and their characteristics
- Machine learning concepts like overfitting, gradient descent
- AI model types: CNNs, RNNs, Transformers, GANs
- Training processes and optimization techniques
- AI ethics and future considerations

Perfect for:
- Students learning about AI/ML
- Educators teaching neural networks
- Anyone curious about how AI models work
- Gamers who enjoy educational content

## 🛠️ **Technical Implementation**

**Built with:**
- **Python 3.8+** 
- **Pygame 2.6+** for graphics and game engine
- **JSON** for persistent data storage
- **Math** for advanced visual effects and animations

**Requirements:**
```
pygame>=2.6.0
numpy>=1.21.0
matplotlib>=3.5.0
imageio>=2.9.0     # For GIF generation
pillow>=8.0.0      # For image processing
```

## 🧩 Game Elements

### 🎨 AI Model Skins
| Model | Special Ability | Characteristics |
|-------|----------------|-----------------|
| CNN | Speed Burst | +20% speed, geometric visuals |
| RNN | Echo Jump | Memory trails, sequence processing |
| Transformer | Teleport | Attention rings, forward teleport |
| GAN | Phase Shift | Dual nature, alternating modes |

### 🚧 Obstacles
| Obstacle | Effect |
|----------|--------|
| Overfitting | Slows down movement |
| Vanishing Gradient | Disables input for 2 seconds |
| Noisy Data | Screen distortion + loss spike |
| Dead Neuron | Disables a powerup slot |

### ⚡ Powerups
| Powerup | Benefit |
|---------|---------|
| Dataset | Restores health + speed boost |
| Optimizer | Increases learning rate |
| Dropout | Temporary immunity |
| Batch Norm | Stabilizes training |

## 🏆 **Achievements System**

Unlock achievements by:
- Reaching score milestones
- Completing challenge modes
- Using different model skins
- Playing consistently over days
- Dodging obstacles and collecting powerups

## 🤝 **Contributing**

We welcome contributions! Whether you want to:
- Add new AI model types
- Create more educational fact cards
- Improve visual effects
- Add new challenge modes
- Enhance the model builder

Please feel free to fork the repository and submit pull requests!

## 📜 **License**

This project is licensed under the MIT License - see the LICENSE file for details.

## 🎉 **Acknowledgments**

- Inspired by the fascinating world of artificial intelligence
- Educational content sourced from various AI/ML resources
- Built with love for both gaming and learning communities

---

**Ready to train your neural network? Let's run! 🏃‍♂️🧠**

*"In Neuron Runner, every obstacle dodged is a lesson learned, every powerup collected is knowledge gained!"*
# neuron-runner
# neuron-runner
