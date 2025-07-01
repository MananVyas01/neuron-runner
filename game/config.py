# Game Configuration Constants
import pygame

# 🎮 Display Settings
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
FPS = 60
TITLE = "🧠 Neuron Runner - Train. Dodge. Optimize."

# 🎨 Colors (ML/AI themed)
COLORS = {
    'BACKGROUND': (15, 20, 35),        # Dark neural network background
    'GROUND': (45, 55, 75),            # Slightly lighter ground
    'NEURON_BLUE': (64, 156, 255),     # Player neuron color
    'ACTIVATION': (0, 255, 127),       # Activation function green
    'GRADIENT': (255, 107, 107),       # Gradient red
    'LOSS': (255, 71, 87),             # Loss/error red
    'DATASET': (255, 193, 7),          # Dataset gold
    'OPTIMIZER': (156, 39, 176),       # Optimizer purple
    'WHITE': (255, 255, 255),
    'BLACK': (0, 0, 0),
    'GRAY': (128, 128, 128),
    'DARK_GRAY': (64, 64, 64),          # Dark gray for backgrounds
    'DARK_BLUE': (25, 35, 60),         # Dark blue for overlays
    'LIGHT_GRAY': (200, 200, 200),     # Light gray for UI elements
    'PURPLE': (156, 39, 176),          # Purple for special elements
    'ORANGE': (255, 165, 0),           # Orange for highlights
    'YELLOW': (255, 255, 0),           # Yellow for attention
    'RED': (255, 0, 0),                # Pure red
    'GREEN': (0, 255, 0),              # Pure green
    'BLUE': (0, 0, 255),               # Pure blue
    'CYAN': (0, 255, 255),             # Cyan for effects
    'MAGENTA': (255, 0, 255),          # Magenta for effects
    'BROWN': (139, 69, 19),            # Brown for earth tones
    'PINK': (255, 192, 203),           # Pink for highlights
    'LIME': (50, 205, 50),             # Lime green
    'NAVY': (0, 0, 128),               # Navy blue
    'MAROON': (128, 0, 0),             # Maroon red
    'OLIVE': (128, 128, 0),            # Olive green
    'SILVER': (192, 192, 192),         # Silver
    'GOLD': (255, 215, 0),             # Gold
    'TEAL': (0, 128, 128),             # Teal
    'FUCHSIA': (255, 0, 255),          # Fuchsia
    'AQUA': (0, 255, 255)              # Aqua
}

# 🏃 Player Settings
PLAYER_START_X = 100
PLAYER_START_Y = SCREEN_HEIGHT - 150
PLAYER_SIZE = 40
PLAYER_SPEED = 8
JUMP_STRENGTH = -18
GRAVITY = 1
GROUND_Y = SCREEN_HEIGHT - 100

# 🚧 Obstacle Settings
OBSTACLE_BASE_SPEED = 6  # Base speed that increases over time
OBSTACLE_MAX_SPEED = 12  # Maximum obstacle speed
OBSTACLE_SPAWN_RATE = 0.02  # Increased spawn rate for more action
OBSTACLE_TYPES = {
    'OVERFITTING': {'color': COLORS['LOSS'], 'effect': 'slow', 'damage': 15},
    'VANISHING_GRADIENT': {'color': COLORS['GRADIENT'], 'effect': 'freeze', 'damage': 20},
    'NOISY_DATA': {'color': COLORS['GRAY'], 'effect': 'distort', 'damage': 10},
    'DEAD_NEURON': {'color': COLORS['BLACK'], 'effect': 'disable', 'damage': 25}
}

# ⚡ Powerup Settings
POWERUP_BASE_SPEED = 6  # Powerups should move at same speed as obstacles
POWERUP_SPAWN_RATE = 0.012  # Increased powerup spawn rate
POWERUP_TYPES = {
    'DATASET': {'color': COLORS['DATASET'], 'effect': 'boost'},
    'OPTIMIZER': {'color': COLORS['OPTIMIZER'], 'effect': 'temp_boost'},
    'DROPOUT': {'color': COLORS['ACTIVATION'], 'effect': 'immunity'}
}

# 🎮 Game Speed Progression
SPEED_INCREASE_RATE = 0.001  # How fast the game speeds up
DIFFICULTY_SPIKE_INTERVAL = 1000  # Every 1000 points, add difficulty spike
EXCITEMENT_THRESHOLD = 500  # Score needed for excitement features

# 📊 Game Mechanics
EPOCH_DURATION = 60  # seconds
LOSS_UPDATE_FREQUENCY = 10  # frames
LAYER_BOOST_DURATION = 10  # seconds

# 🔊 Audio Settings
SOUND_VOLUME = 0.7
MUSIC_VOLUME = 0.5

# 📈 UI Settings
UI_FONT_SIZE = 24
SMALL_FONT_SIZE = 18
GRAPH_WIDTH = 200
GRAPH_HEIGHT = 100
GRAPH_X = SCREEN_WIDTH - 220
GRAPH_Y = 20

# 🎯 Game States
GAME_STATES = {
    'MENU': 0,
    'PLAYING': 1,
    'PAUSED': 2,
    'GAME_OVER': 3
}

# ⌨️ Input Keys
KEYS = {
    'JUMP': pygame.K_SPACE,
    'DUCK': pygame.K_DOWN,
    'LAYER_BOOST': pygame.K_l,
    'PAUSE': pygame.K_p,
    'RESTART': pygame.K_r
}
