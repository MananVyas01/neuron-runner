"""
🧠 Model Skin System - Choose Your Neural Network Architecture
Advanced theming system with different AI model types and abilities.
"""

import pygame
import math
import random
from .config import *

class ModelSkin:
    def __init__(self, name, display_name, description, color_scheme, abilities, visual_style):
        self.name = name
        self.display_name = display_name
        self.description = description
        self.color_scheme = color_scheme
        self.abilities = abilities
        self.visual_style = visual_style
        
        # Performance characteristics
        self.base_speed = abilities.get('speed_multiplier', 1.0)
        self.jump_power = abilities.get('jump_multiplier', 1.0)
        self.special_cooldown = abilities.get('special_cooldown', 30)
        self.special_ability = abilities.get('special_ability', None)
        
    def get_themed_colors(self):
        """Get color scheme for this model"""
        return self.color_scheme
    
    def apply_passive_effects(self, player):
        """Apply passive model effects to player"""
        player.optimizer_speed *= self.base_speed
        # Additional passive effects can be added here
    
    def can_use_special(self, player):
        """Check if special ability can be used"""
        return hasattr(player, 'special_cooldown_timer') and player.special_cooldown_timer <= 0
    
    def use_special_ability(self, player):
        """Execute special ability"""
        if not self.can_use_special(player):
            return False
            
        if self.special_ability == 'teleport':
            # Transformer teleport
            player.rect.x += 200  # Teleport forward
            player.special_cooldown_timer = self.special_cooldown * FPS
            return True
        elif self.special_ability == 'speed_burst':
            # CNN speed burst
            player.cnn_speed_timer = 3 * FPS  # 3 seconds
            player.special_cooldown_timer = self.special_cooldown * FPS
            return True
        elif self.special_ability == 'echo_jump':
            # RNN echo effect
            player.rnn_echo_timer = 2 * FPS  # 2 seconds
            player.special_cooldown_timer = self.special_cooldown * FPS
            return True
        
        return False

# Define available model skins
MODEL_SKINS = {
    'CNN': ModelSkin(
        name='CNN',
        display_name='Convolutional Neural Network',
        description='🏃‍♂️ SPRINTER: Fast processing, excellent at pattern recognition\n• +20% movement speed\n• Special: Speed Burst (3s boost)\n• Visual: Clean geometric patterns',
        color_scheme={
            'primary': (64, 156, 255),      # Bright blue
            'secondary': (0, 255, 127),     # Green accent
            'glow': (100, 200, 255),        # Light blue glow
            'trail': (64, 156, 255, 100),   # Semi-transparent blue
        },
        abilities={
            'speed_multiplier': 1.2,
            'jump_multiplier': 1.0,
            'special_ability': 'speed_burst',
            'special_cooldown': 25,
        },
        visual_style={
            'shape': 'geometric',
            'pattern': 'grid',
            'animation': 'sharp',
            'trail_type': 'lines',
        }
    ),
    
    'RNN': ModelSkin(
        name='RNN',
        display_name='Recurrent Neural Network',
        description='🔄 MEMORY MASTER: Processes sequences, slight input delay\n• Looping visual trail\n• Special: Echo Jump (double jump effect)\n• Visual: Matrix-style cascading patterns',
        color_scheme={
            'primary': (156, 64, 255),      # Purple
            'secondary': (255, 64, 156),    # Pink accent
            'glow': (200, 100, 255),        # Light purple glow
            'trail': (156, 64, 255, 80),    # Semi-transparent purple
        },
        abilities={
            'speed_multiplier': 0.95,
            'jump_multiplier': 1.1,
            'special_ability': 'echo_jump',
            'special_cooldown': 20,
            'input_delay': 0.1,  # Slight delay for realism
        },
        visual_style={
            'shape': 'flowing',
            'pattern': 'matrix',
            'animation': 'wave',
            'trail_type': 'spiral',
        }
    ),
    
    'TRANSFORMER': ModelSkin(
        name='TRANSFORMER',
        display_name='Transformer Network',
        description='⚡ ATTENTION MASTER: Self-attention mechanism, teleportation\n• Special: Forward Teleport (200px)\n• Visual: Multi-head attention rings\n• Balanced stats with unique mobility',
        color_scheme={
            'primary': (255, 193, 7),       # Gold
            'secondary': (255, 87, 34),     # Orange accent
            'glow': (255, 235, 59),         # Bright yellow glow
            'trail': (255, 193, 7, 120),    # Semi-transparent gold
        },
        abilities={
            'speed_multiplier': 1.0,
            'jump_multiplier': 1.0,
            'special_ability': 'teleport',
            'special_cooldown': 30,
        },
        visual_style={
            'shape': 'multi_ring',
            'pattern': 'attention',
            'animation': 'pulse',
            'trail_type': 'particles',
        }
    ),
    
    'GAN': ModelSkin(
        name='GAN',
        display_name='Generative Adversarial Network',
        description='🎭 DUAL NATURE: Generator and Discriminator in one\n• Alternating visual modes\n• Special: Phase Shift (immunity + speed)\n• Visual: Dual-tone shifting appearance',
        color_scheme={
            'primary': (76, 175, 80),       # Green
            'secondary': (244, 67, 54),     # Red
            'glow': (139, 195, 74),         # Light green
            'trail': (76, 175, 80, 90),     # Semi-transparent green
        },
        abilities={
            'speed_multiplier': 1.1,
            'jump_multiplier': 0.9,
            'special_ability': 'phase_shift',
            'special_cooldown': 35,
        },
        visual_style={
            'shape': 'dual',
            'pattern': 'adversarial',
            'animation': 'alternating',
            'trail_type': 'dual_color',
        }
    )
}

class ModelSelector:
    def __init__(self):
        self.current_model = 'CNN'  # Default
        self.selection_screen_active = False
        self.selected_index = 0
        self.models_list = list(MODEL_SKINS.keys())
        
        # UI elements (fonts will be initialized when needed)
        self.font_large = None
        self.font_medium = None
        self.font_small = None
        
        # Animation
        self.preview_timer = 0
        self.selection_pulse = 0
        
    def show_selection_screen(self):
        """Activate model selection screen"""
        self.selection_screen_active = True
        self.preview_timer = 0
        
    def hide_selection_screen(self):
        """Hide model selection screen"""
        self.selection_screen_active = False
        
    def handle_input(self, keys, events):
        """Handle input for model selection"""
        if not self.selection_screen_active:
            return False
            
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP or event.key == pygame.K_w:
                    self.selected_index = (self.selected_index - 1) % len(self.models_list)
                elif event.key == pygame.K_DOWN or event.key == pygame.K_s:
                    self.selected_index = (self.selected_index + 1) % len(self.models_list)
                elif event.key == pygame.K_RETURN or event.key == pygame.K_SPACE:
                    self.current_model = self.models_list[self.selected_index]
                    self.hide_selection_screen()
                    return True  # Model selected
                elif event.key == pygame.K_ESCAPE:
                    self.hide_selection_screen()
                    return False
        
        return False
    
    def update(self):
        """Update animation timers"""
        self.preview_timer += 0.1
        self.selection_pulse += 0.15
        
    def _init_fonts(self):
        """Initialize fonts if not already done"""
        if self.font_large is None:
            self.font_large = pygame.font.Font(None, 48)
            self.font_medium = pygame.font.Font(None, 32)
            self.font_small = pygame.font.Font(None, 24)
    
    def draw_selection_screen(self, screen):
        """Draw the model selection interface"""
        if not self.selection_screen_active:
            return
        
        self._init_fonts()  # Ensure fonts are initialized
        
        # Semi-transparent overlay
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        overlay.set_alpha(200)
        overlay.fill(COLORS['BLACK'])
        screen.blit(overlay, (0, 0))
        
        # Title
        title_text = self.font_large.render("🧠 SELECT YOUR AI MODEL", True, COLORS['WHITE'])
        title_rect = title_text.get_rect(center=(SCREEN_WIDTH//2, 80))
        screen.blit(title_text, title_rect)
        
        # Model cards
        card_width = 300
        card_height = 200
        cards_per_row = 2
        start_x = (SCREEN_WIDTH - (cards_per_row * card_width + (cards_per_row - 1) * 50)) // 2
        start_y = 150
        
        for i, model_name in enumerate(self.models_list):
            model = MODEL_SKINS[model_name]
            
            # Calculate card position
            row = i // cards_per_row
            col = i % cards_per_row
            card_x = start_x + col * (card_width + 50)
            card_y = start_y + row * (card_height + 30)
            
            # Selection highlight
            is_selected = (i == self.selected_index)
            if is_selected:
                pulse = abs(math.sin(self.selection_pulse)) * 10
                highlight_rect = pygame.Rect(card_x - 5 - pulse, card_y - 5 - pulse, 
                                           card_width + 10 + pulse*2, card_height + 10 + pulse*2)
                pygame.draw.rect(screen, COLORS['ACTIVATION'], highlight_rect, 3)
            
            # Card background
            card_rect = pygame.Rect(card_x, card_y, card_width, card_height)
            card_bg = pygame.Surface((card_width, card_height))
            card_bg.set_alpha(180)
            card_bg.fill(model.color_scheme['primary'])
            screen.blit(card_bg, card_rect)
            
            # Card border
            pygame.draw.rect(screen, COLORS['WHITE'], card_rect, 2)
            
            # Model preview (animated)
            self.draw_model_preview(screen, model, card_x + card_width//2, card_y + 60)
            
            # Model name
            name_text = self.font_medium.render(model.display_name, True, COLORS['WHITE'])
            name_rect = name_text.get_rect(center=(card_x + card_width//2, card_y + 120))
            screen.blit(name_text, name_rect)
            
            # Model stats preview
            stats_y = card_y + 145
            stats_text = [
                f"Speed: {model.base_speed:.1f}x",
                f"Special: {model.abilities.get('special_ability', 'None').replace('_', ' ').title()}"
            ]
            
            for j, stat in enumerate(stats_text):
                stat_surface = self.font_small.render(stat, True, COLORS['WHITE'])
                stat_rect = stat_surface.get_rect(center=(card_x + card_width//2, stats_y + j * 20))
                screen.blit(stat_surface, stat_rect)
        
        # Selected model detailed info
        if self.selected_index < len(self.models_list):
            selected_model = MODEL_SKINS[self.models_list[self.selected_index]]
            
            # Description panel
            desc_panel_y = start_y + ((len(self.models_list) + 1) // cards_per_row) * (card_height + 30) + 50
            desc_width = 600
            desc_height = 120
            desc_x = (SCREEN_WIDTH - desc_width) // 2
            
            # Panel background
            desc_bg = pygame.Surface((desc_width, desc_height))
            desc_bg.set_alpha(200)
            desc_bg.fill(COLORS['BLACK'])
            screen.blit(desc_bg, (desc_x, desc_panel_y))
            pygame.draw.rect(screen, selected_model.color_scheme['primary'], 
                           (desc_x, desc_panel_y, desc_width, desc_height), 3)
            
            # Description text
            desc_lines = selected_model.description.split('\n')
            for i, line in enumerate(desc_lines):
                if line.strip():
                    desc_surface = self.font_small.render(line.strip(), True, COLORS['WHITE'])
                    screen.blit(desc_surface, (desc_x + 20, desc_panel_y + 20 + i * 22))
        
        # Controls help
        controls_text = "↑↓ Navigate | ENTER Select | ESC Cancel"
        controls_surface = self.font_small.render(controls_text, True, COLORS['GRAY'])
        controls_rect = controls_surface.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT - 30))
        screen.blit(controls_surface, controls_rect)
    
    def draw_model_preview(self, screen, model, x, y):
        """Draw animated preview of the model"""
        size = 30
        
        if model.visual_style['shape'] == 'geometric':
            # CNN - clean geometric shape
            points = []
            for i in range(6):
                angle = (i * 60) * math.pi / 180
                px = x + math.cos(angle + self.preview_timer) * size
                py = y + math.sin(angle + self.preview_timer) * size
                points.append((px, py))
            pygame.draw.polygon(screen, model.color_scheme['glow'], points)
            
        elif model.visual_style['shape'] == 'flowing':
            # RNN - flowing circular pattern
            for i in range(3):
                radius = size - i * 8
                offset = self.preview_timer + i * math.pi / 3
                glow_x = x + math.cos(offset) * 5
                glow_y = y + math.sin(offset) * 5
                pygame.draw.circle(screen, model.color_scheme['glow'], 
                                 (int(glow_x), int(glow_y)), radius, 2)
                
        elif model.visual_style['shape'] == 'multi_ring':
            # Transformer - multiple attention rings
            for i in range(4):
                radius = size - i * 6
                offset = self.preview_timer * (1 + i * 0.2)
                ring_alpha = 150 - i * 30
                ring_color = (*model.color_scheme['primary'], ring_alpha)
                
                ring_surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(ring_surface, ring_color, (radius, radius), radius, 2)
                screen.blit(ring_surface, (x - radius, y - radius))
                
        elif model.visual_style['shape'] == 'dual':
            # GAN - dual nature visualization
            generator_color = model.color_scheme['primary']
            discriminator_color = model.color_scheme['secondary']
            
            # Switch colors based on timer
            if int(self.preview_timer * 2) % 2:
                generator_color, discriminator_color = discriminator_color, generator_color
            
            pygame.draw.circle(screen, generator_color, (x - 15, y), size // 2)
            pygame.draw.circle(screen, discriminator_color, (x + 15, y), size // 2)
    
    def get_current_model(self):
        """Get currently selected model skin"""
        return MODEL_SKINS[self.current_model]
    
    def is_selection_active(self):
        """Check if selection screen is active"""
        return self.selection_screen_active
    
    def get_selected_model(self):
        """Get the currently selected model skin"""
        return self.current_model
    
    def get_current_model_data(self):
        """Get the current model skin data"""
        return MODEL_SKINS.get(self.current_model, MODEL_SKINS['CNN'])

# Global model selector instance
model_selector = ModelSelector()
