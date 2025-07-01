"""
🧠 Neural Network Player Avatar
The player represents a neural network that's being trained.
"""

import pygame
import math
from .config import *
from .model_skins import model_selector

class NeuronPlayer(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        
        # 🎨 Visual properties
        self.image = pygame.Surface((PLAYER_SIZE, PLAYER_SIZE), pygame.SRCALPHA)
        self.rect = self.image.get_rect()
        self.rect.x = PLAYER_START_X
        self.rect.y = PLAYER_START_Y
        
        # 🧠 Neural Network Properties
        self.activation_level = 100  # Health/activation strength
        self.layers = ['input', 'hidden']  # Current network architecture
        self.optimizer_speed = 1.0  # Movement speed multiplier
        self.learning_rate = 0.01  # Affects various game mechanics
        
        # 🏃 Movement properties
        self.velocity_y = 0
        self.is_jumping = False
        self.is_ducking = False
        self.on_ground = True
        
        # ⚡ Special abilities
        self.layer_boost_timer = 0
        self.immunity_timer = 0
        self.freeze_timer = 0  # For vanishing gradient effect
        
        # 🎨 Animation properties
        self.pulse_timer = 0
        self.base_size = PLAYER_SIZE
        
        # 📊 Stats
        self.training_progress = 0
        self.epochs_survived = 0
        
        # 🎮 Excitement Features
        self.combo_count = 0
        self.combo_timer = 0
        self.last_action_time = 0
        self.adrenaline_mode = False
        self.adrenaline_timer = 0
        self.perfect_dodges = 0
        self.close_call_timer = 0
        
        # 🧠 Model Skin Features
        self.current_model = None
        self.model_colors = COLORS.copy()  # Default colors
        self.special_cooldown_timer = 0
        self.cnn_speed_timer = 0
        self.rnn_echo_timer = 0
        self.input_delay_buffer = []
        self.special_abilities_used = 0  # Track special ability usage
        
        # Apply model skin
        self.apply_model_skin()
        
        self.update_sprite()
    
    def update_sprite(self):
        """Update the visual representation of the neuron"""
        # Create a circular neuron with pulsing effect
        self.pulse_timer += 0.15
        pulse_intensity = 4 + int(self.activation_level / 25)  # Pulse more when healthy
        pulse_size = self.base_size + int(math.sin(self.pulse_timer) * pulse_intensity)
        
        # Recreate surface with larger size for effects
        self.image = pygame.Surface((pulse_size + 20, pulse_size + 20), pygame.SRCALPHA)
        
        # Draw the main neuron body
        center = ((pulse_size + 20) // 2, (pulse_size + 20) // 2)
        radius = pulse_size // 2 - 2
        
        # Color based on activation level with more vibrant colors
        activation_ratio = self.activation_level / 100
        if activation_ratio > 0.7:
            base_color = self.model_colors.get('primary', COLORS['NEURON_BLUE'])
            color = self.interpolate_color(base_color, self.model_colors.get('glow', COLORS['ACTIVATION']), (activation_ratio - 0.7) / 0.3)
        elif activation_ratio > 0.3:
            color = self.model_colors.get('primary', COLORS['NEURON_BLUE'])
        else:
            color = self.interpolate_color(COLORS['LOSS'], self.model_colors.get('primary', COLORS['NEURON_BLUE']), activation_ratio / 0.3)
        
        # Draw outer glow effect
        for i in range(3):
            glow_radius = radius + 3 + i * 2
            glow_alpha = 60 - i * 20
            glow_color = (*color, glow_alpha)
            glow_surface = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surface, glow_color, (glow_radius, glow_radius), glow_radius)
            self.image.blit(glow_surface, (center[0] - glow_radius, center[1] - glow_radius))
        
        # Draw main neuron body
        pygame.draw.circle(self.image, color, center, radius)
        pygame.draw.circle(self.image, COLORS['WHITE'], center, radius, 2)
        
        # Draw inner core
        core_radius = max(3, radius // 3)
        core_color = self.interpolate_color(color, COLORS['WHITE'], 0.5)
        pygame.draw.circle(self.image, core_color, center, core_radius)
        
        # Draw adrenaline effect if active
        if self.adrenaline_mode:
            adrenaline_pulse = math.sin(self.pulse_timer * 4) * 5
            adrenaline_color = (*COLORS['LOSS'], 180)
            for i in range(3):
                adrenaline_radius = radius + 15 + i * 5 + int(adrenaline_pulse)
                pygame.draw.circle(self.image, adrenaline_color, center, adrenaline_radius, 2)
        
        # Draw combo indicator
        if self.combo_count > 0:
            combo_glow = math.sin(self.pulse_timer * 3) * 3
            combo_color = (*COLORS['DATASET'], 200)
            pygame.draw.circle(self.image, combo_color, center, radius + 10 + int(combo_glow), 3)
        
        # Draw activation ring if boosted (more prominent)
        if self.layer_boost_timer > 0:
            ring_pulse = math.sin(self.pulse_timer * 2) * 2
            pygame.draw.circle(self.image, COLORS['ACTIVATION'], center, radius + 8 + int(ring_pulse), 4)
            pygame.draw.circle(self.image, COLORS['WHITE'], center, radius + 12 + int(ring_pulse), 2)
        
        # Draw immunity shield if active (more visible)
        if self.immunity_timer > 0:
            shield_pulse = math.sin(self.pulse_timer * 1.5) * 3
            shield_color = (*COLORS['DATASET'], 150)
            shield_surface = pygame.Surface((radius * 3, radius * 3), pygame.SRCALPHA)
            pygame.draw.circle(shield_surface, shield_color, (radius * 3 // 2, radius * 3 // 2), radius + 15 + int(shield_pulse))
            pygame.draw.circle(shield_surface, COLORS['DATASET'], (radius * 3 // 2, radius * 3 // 2), radius + 15 + int(shield_pulse), 3)
            self.image.blit(shield_surface, (center[0] - radius * 3 // 2, center[1] - radius * 3 // 2))
        
        # Draw health indicator bars around the neuron
        self.draw_health_indicator(center, radius)
        
        # Draw neural connections (improved visibility)
        self.draw_neural_connections(center, radius)
        
        # Draw model-specific themed elements
        self.draw_model_themed_sprite(center, radius)
        
        # Update rect to maintain position
        old_center = self.rect.center
        self.rect = self.image.get_rect()
        self.rect.center = old_center
    
    def draw_health_indicator(self, center, radius):
        """Draw health bars around the neuron"""
        bar_count = 8
        bar_length = 12
        bar_width = 3
        bar_distance = radius + 25
        
        for i in range(bar_count):
            angle = (360 / bar_count) * i
            rad = math.radians(angle)
            
            # Calculate bar position
            bar_x = center[0] + math.cos(rad) * bar_distance
            bar_y = center[1] + math.sin(rad) * bar_distance
            
            # Calculate bar endpoints
            bar_start_x = bar_x + math.cos(rad + math.pi/2) * (bar_length // 2)
            bar_start_y = bar_y + math.sin(rad + math.pi/2) * (bar_length // 2)
            bar_end_x = bar_x - math.cos(rad + math.pi/2) * (bar_length // 2)
            bar_end_y = bar_y - math.sin(rad + math.pi/2) * (bar_length // 2)
            
            # Determine bar color based on health
            health_ratio = self.activation_level / 100
            bar_threshold = (i + 1) / bar_count
            
            if health_ratio >= bar_threshold:
                if health_ratio > 0.6:
                    bar_color = COLORS['ACTIVATION']
                elif health_ratio > 0.3:
                    bar_color = COLORS['DATASET']
                else:
                    bar_color = COLORS['GRADIENT']
            else:
                bar_color = COLORS['GRAY']
            
            pygame.draw.line(self.image, bar_color, (bar_start_x, bar_start_y), (bar_end_x, bar_end_y), bar_width)
    
    def draw_neural_connections(self, center, radius):
        """Draw improved neural connections"""
        connection_alpha = min(255, int(self.activation_level * 2.55))
        layer_count = min(len(self.layers), 4)  # Max 4 connections
        
        for i in range(layer_count):
            # Create dynamic connection pattern
            base_angle = -90 + (i - layer_count/2 + 0.5) * 25  # Spread connections
            angle_variation = math.sin(self.pulse_timer + i) * 5  # Slight movement
            angle = base_angle + angle_variation
            rad = math.radians(angle)
            
            # Connection points
            start_x = center[0] + math.cos(rad) * (radius - 3)
            start_y = center[1] + math.sin(rad) * (radius - 3)
            end_x = center[0] + math.cos(rad) * (radius + 20)
            end_y = center[1] + math.sin(rad) * (radius + 20)
            
            # Draw connection with pulsing effect
            connection_width = 2 + int(math.sin(self.pulse_timer + i) * 1)
            connection_color = (*COLORS['ACTIVATION'], connection_alpha)
            
            # Create connection surface with alpha
            connection_surface = pygame.Surface((self.image.get_width(), self.image.get_height()), pygame.SRCALPHA)
            pygame.draw.line(connection_surface, connection_color, (start_x, start_y), (end_x, end_y), connection_width)
            
            # Draw connection node at the end
            node_color = (*COLORS['NEURON_BLUE'], connection_alpha)
            pygame.draw.circle(connection_surface, node_color, (int(end_x), int(end_y)), 3)
            
            self.image.blit(connection_surface, (0, 0))
    
    def interpolate_color(self, color1, color2, t):
        """Interpolate between two colors"""
        t = max(0, min(1, t))  # Clamp t to [0, 1]
        return tuple(int(c1 + (c2 - c1) * t) for c1, c2 in zip(color1, color2))
    
    def handle_input(self, keys):
        """Handle player input"""
        if self.freeze_timer > 0:
            return  # Vanishing gradient effect - no input response
        
        # Jump
        if keys[KEYS['JUMP']] and self.on_ground and not self.is_ducking:
            self.jump()
        
        # Duck/Slide
        if keys[KEYS['DUCK']]:
            self.duck()
        else:
            self.stop_ducking()
        
        # Layer boost ability
        if keys[KEYS['LAYER_BOOST']] and self.layer_boost_timer <= 0:
            self.activate_layer_boost()
        
        # Special ability (X key)
        if keys[pygame.K_x]:
            self.use_special_ability()
    
    def jump(self):
        """Make the neuron jump"""
        if self.on_ground:
            self.velocity_y = JUMP_STRENGTH
            self.is_jumping = True
            self.on_ground = False
            self.register_action('jump')
            
            # Play jump sound
            try:
                from .sound import sound_system
                sound_system.play_jump()
            except ImportError:
                pass
    
    def duck(self):
        """Make the neuron duck/slide"""
        if not self.is_ducking:
            self.is_ducking = True
            # Store original rect for restoration
            self.original_rect = self.rect.copy()
            # Reduce hitbox size when ducking
            new_height = PLAYER_SIZE // 2
            self.rect.height = new_height
            self.rect.y += (PLAYER_SIZE - new_height)
            self.register_action('duck')
    
    def stop_ducking(self):
        """Stop ducking"""
        if self.is_ducking:
            self.is_ducking = False
            # Restore normal hitbox
            if hasattr(self, 'original_rect'):
                self.rect.height = PLAYER_SIZE
                self.rect.y = self.original_rect.y
    
    def register_action(self, action_type):
        """Register player action for combo system"""
        current_time = pygame.time.get_ticks()
        
        # Check for combo timing (within 1 second of last action)
        if current_time - self.last_action_time < 1000:
            self.combo_count += 1
            self.combo_timer = 3 * FPS  # 3 seconds to maintain combo
        else:
            self.combo_count = 1
        
        self.last_action_time = current_time
        
        # Activate adrenaline mode with high combos
        if self.combo_count >= 5 and not self.adrenaline_mode:
            self.activate_adrenaline_mode()
        elif self.combo_count % 5 == 0 and self.combo_count > 0:
            # Play combo sound for every 5th combo
            try:
                from .sound import sound_system
                sound_system.play_combo()
            except ImportError:
                pass
    
    def activate_adrenaline_mode(self):
        """Activate high-intensity adrenaline mode"""
        self.adrenaline_mode = True
        self.adrenaline_timer = 5 * FPS  # 5 seconds of adrenaline
        self.optimizer_speed = min(3.0, self.optimizer_speed + 0.5)  # Speed boost
        
        # Create particle effect for adrenaline activation
        from .particles import particle_system
        particle_system.create_combo_burst(self.rect.centerx, self.rect.centery)
    
    def activate_layer_boost(self):
        """Activate special layer boost ability"""
        self.layer_boost_timer = LAYER_BOOST_DURATION * FPS
        # Add a temporary layer to the network
        if 'dropout' not in self.layers:
            self.layers.append('dropout')
        self.immunity_timer = LAYER_BOOST_DURATION * FPS // 2  # Half duration immunity
    
    def update(self):
        """Update player state"""
        # Apply gravity
        if not self.on_ground:
            self.velocity_y += GRAVITY
        
        # Update position
        self.rect.y += self.velocity_y * self.optimizer_speed
        
        # Ground collision
        if self.rect.bottom >= GROUND_Y:
            self.rect.bottom = GROUND_Y
            self.velocity_y = 0
            self.is_jumping = False
            self.on_ground = True
        
        # Update timers
        if self.layer_boost_timer > 0:
            self.layer_boost_timer -= 1
            if self.layer_boost_timer <= 0:
                # Remove temporary layers
                if 'dropout' in self.layers:
                    self.layers.remove('dropout')
        
        if self.immunity_timer > 0:
            self.immunity_timer -= 1
        
        if self.freeze_timer > 0:
            self.freeze_timer -= 1
        
        # Update combo system
        if self.combo_timer > 0:
            self.combo_timer -= 1
        else:
            self.combo_count = 0
        
        # Update adrenaline mode
        if self.adrenaline_timer > 0:
            self.adrenaline_timer -= 1
            if self.adrenaline_timer <= 0:
                self.adrenaline_mode = False
                self.optimizer_speed = max(1.0, self.optimizer_speed - 0.5)
            else:
                # Add continuous adrenaline particles
                from .particles import particle_system
                particle_system.create_adrenaline_particles(self.rect.centerx, self.rect.centery)
        
        # Update close call timer
        if self.close_call_timer > 0:
            self.close_call_timer -= 1
        
        # Update model-specific timers
        if self.special_cooldown_timer > 0:
            self.special_cooldown_timer -= 1
        if self.cnn_speed_timer > 0:
            self.cnn_speed_timer -= 1
        if self.rnn_echo_timer > 0:
            self.rnn_echo_timer -= 1
        
        # Update training progress
        self.training_progress += 0.1
        
        # Update visual representation
        self.update_sprite()
        
        # Keep player on screen
        self.rect.clamp_ip(pygame.Rect(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT))
    
    def take_damage(self, damage_type):
        """Handle different types of damage/effects"""
        if self.immunity_timer > 0:
            return False  # Immune to damage
        
        # More severe damage in adrenaline mode for higher stakes
        damage_multiplier = 1.5 if self.adrenaline_mode else 1.0
        
        if damage_type == 'overfitting':
            self.optimizer_speed = max(0.3, self.optimizer_speed - 0.3)
            self.activation_level = max(0, self.activation_level - int(15 * damage_multiplier))
        
        elif damage_type == 'vanishing_gradient':
            self.freeze_timer = int(2.5 * FPS)  # 2.5 seconds of no input
            self.activation_level = max(0, self.activation_level - int(20 * damage_multiplier))
        
        elif damage_type == 'noisy_data':
            self.activation_level = max(0, self.activation_level - int(10 * damage_multiplier))
            # Screen distortion effect handled elsewhere
        
        elif damage_type == 'dead_neuron':
            self.activation_level = max(0, self.activation_level - int(25 * damage_multiplier))
            # Disable a layer temporarily
            if len(self.layers) > 1:
                self.layers.pop()
        
        # Reset combo on damage
        self.combo_count = 0
        self.combo_timer = 0
        
        return True  # Damage was applied
    
    def register_close_call(self):
        """Register a close call for excitement"""
        self.close_call_timer = 1 * FPS  # 1 second close call indicator
        self.perfect_dodges += 1
        
        # Bonus activation for perfect dodges
        if self.perfect_dodges % 3 == 0:
            self.activation_level = min(100, self.activation_level + 5)
    
    def apply_powerup(self, powerup_type):
        """Apply powerup effects"""
        if powerup_type == 'dataset':
            self.activation_level = min(100, self.activation_level + 20)
            self.optimizer_speed = min(2.0, self.optimizer_speed + 0.1)
        
        elif powerup_type == 'optimizer':
            self.learning_rate = min(0.1, self.learning_rate + 0.01)
            self.optimizer_speed = min(2.0, self.optimizer_speed + 0.3)
        
        elif powerup_type == 'dropout':
            self.immunity_timer = 10 * FPS  # 10 seconds immunity
            if 'dropout' not in self.layers:
                self.layers.append('dropout')
    
    def get_status(self):
        """Get current player status for UI display"""
        return {
            'activation': self.activation_level,
            'layers': len(self.layers),
            'optimizer_speed': self.optimizer_speed,
            'learning_rate': self.learning_rate,
            'progress': self.training_progress,
            'epochs': self.epochs_survived,
            'boosted': self.layer_boost_timer > 0,
            'immune': self.immunity_timer > 0,
            'frozen': self.freeze_timer > 0,
            'combo': self.combo_count,
            'adrenaline': self.adrenaline_mode,
            'close_call': self.close_call_timer > 0,
            'perfect_dodges': self.perfect_dodges
        }
    
    def reset(self):
        """Reset player to initial state"""
        self.rect.x = PLAYER_START_X
        self.rect.y = PLAYER_START_Y
        self.velocity_y = 0
        self.activation_level = 100
        self.layers = ['input', 'hidden']
        self.optimizer_speed = 1.0
        self.learning_rate = 0.01
        self.layer_boost_timer = 0
        self.immunity_timer = 0
        self.freeze_timer = 0
        self.training_progress = 0
        self.epochs_survived = 0
        self.is_jumping = False
        self.is_ducking = False
        self.on_ground = True
        # Reset excitement features
        self.combo_count = 0
        self.combo_timer = 0
        self.last_action_time = 0
        self.adrenaline_mode = False
        self.adrenaline_timer = 0
        self.perfect_dodges = 0
        self.close_call_timer = 0
        self.special_abilities_used = 0  # Reset special ability counter
    
    def apply_model_skin(self):
        """Apply the currently selected model skin"""
        self.current_model = model_selector.get_current_model()
        self.model_colors = self.current_model.get_themed_colors()
        
        # Apply passive effects
        self.current_model.apply_passive_effects(self)
        
        # Initialize model-specific timers
        self.special_cooldown_timer = 0
        
        print(f"🧠 Applied model skin: {self.current_model.display_name}")
    
    def use_special_ability(self):
        """Use model-specific special ability"""
        if self.current_model and self.current_model.can_use_special(self):
            success = self.current_model.use_special_ability(self)
            if success:
                self.special_abilities_used += 1  # Track usage
                print(f"⚡ Used {self.current_model.display_name} special ability!")
                return True
        return False
    
    def draw_model_themed_sprite(self, center, radius):
        """Draw model-specific themed visual elements"""
        if not self.current_model:
            return
            
        visual_style = self.current_model.visual_style
        colors = self.model_colors
        
        if visual_style['shape'] == 'geometric':
            # CNN - Clean geometric patterns
            self.draw_cnn_style(center, radius, colors)
        elif visual_style['shape'] == 'flowing':
            # RNN - Matrix-style flowing patterns
            self.draw_rnn_style(center, radius, colors)
        elif visual_style['shape'] == 'multi_ring':
            # Transformer - Multi-head attention rings
            self.draw_transformer_style(center, radius, colors)
        elif visual_style['shape'] == 'dual':
            # GAN - Dual nature visualization
            self.draw_gan_style(center, radius, colors)
    
    def draw_cnn_style(self, center, radius, colors):
        """Draw CNN-specific visual elements"""
        # Geometric grid pattern
        grid_size = 4
        for i in range(-grid_size, grid_size + 1):
            for j in range(-grid_size, grid_size + 1):
                if i == 0 or j == 0:  # Main grid lines
                    start_x = center[0] + i * 5
                    start_y = center[1] + j * 5
                    if abs(i * 5) < radius and abs(j * 5) < radius:
                        pixel_color = colors['secondary'] if (i + j) % 2 == 0 else colors['primary']
                        pygame.draw.circle(self.image, pixel_color, (start_x, start_y), 1)
        
        # Speed burst effect
        if self.cnn_speed_timer > 0:
            burst_intensity = self.cnn_speed_timer / (3 * FPS)
            burst_color = (*colors['glow'], int(200 * burst_intensity))
            for i in range(3):
                burst_radius = radius + 20 + i * 8
                burst_surface = pygame.Surface((burst_radius * 2, burst_radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(burst_surface, burst_color, (burst_radius, burst_radius), burst_radius, 2)
                self.image.blit(burst_surface, (center[0] - burst_radius, center[1] - burst_radius))
    
    def draw_rnn_style(self, center, radius, colors):
        """Draw RNN-specific visual elements"""
        # Matrix-style cascading effect
        matrix_density = 8
        for i in range(matrix_density):
            angle = (360 / matrix_density) * i + self.pulse_timer * 50
            rad = math.radians(angle)
            
            # Cascading points
            for j in range(3):
                cascade_radius = radius + 15 + j * 8
                cascade_x = center[0] + math.cos(rad) * cascade_radius
                cascade_y = center[1] + math.sin(rad) * cascade_radius
                
                alpha = 255 - j * 80
                cascade_color = (*colors['primary'], alpha)
                cascade_surface = pygame.Surface((6, 6), pygame.SRCALPHA)
                pygame.draw.circle(cascade_surface, cascade_color, (3, 3), 3)
                self.image.blit(cascade_surface, (cascade_x - 3, cascade_y - 3))
        
        # Echo effect for RNN
        if self.rnn_echo_timer > 0:
            echo_intensity = self.rnn_echo_timer / (2 * FPS)
            echo_color = (*colors['secondary'], int(150 * echo_intensity))
            echo_surface = pygame.Surface((radius * 4, radius * 4), pygame.SRCALPHA)
            pygame.draw.circle(echo_surface, echo_color, (radius * 2, radius * 2), int(radius * 1.5), 3)
            self.image.blit(echo_surface, (center[0] - radius * 2, center[1] - radius * 2))
    
    def draw_transformer_style(self, center, radius, colors):
        """Draw Transformer-specific visual elements"""
        # Multi-head attention rings
        num_heads = 4
        for head in range(num_heads):
            head_angle = (360 / num_heads) * head + self.pulse_timer * 20
            head_radius = radius + 12 + head * 3
            
            # Attention ring
            attention_color = (*colors['primary'], 180 - head * 30)
            attention_surface = pygame.Surface((head_radius * 2, head_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(attention_surface, attention_color, (head_radius, head_radius), head_radius, 2)
            
            # Rotate the ring
            rotated_surface = pygame.transform.rotate(attention_surface, head_angle)
            rotated_rect = rotated_surface.get_rect(center=center)
            self.image.blit(rotated_surface, rotated_rect)
        
        # Teleport preparation effect
        if self.special_cooldown_timer < 2 * FPS:  # Show when ready
            teleport_glow = abs(math.sin(self.pulse_timer * 5)) * 100
            teleport_color = (*colors['glow'], int(teleport_glow))
            teleport_surface = pygame.Surface((radius * 3, radius * 3), pygame.SRCALPHA)
            pygame.draw.circle(teleport_surface, teleport_color, (radius * 3 // 2, radius * 3 // 2), radius + 10, 4)
            self.image.blit(teleport_surface, (center[0] - radius * 3 // 2, center[1] - radius * 3 // 2))
    
    def draw_gan_style(self, center, radius, colors):
        """Draw GAN-specific visual elements"""
        # Dual nature - alternating between generator and discriminator
        phase = int(self.pulse_timer * 2) % 2
        
        if phase == 0:  # Generator phase
            primary_color = colors['primary']
            secondary_color = colors['secondary']
        else:  # Discriminator phase
            primary_color = colors['secondary']
            secondary_color = colors['primary']
        
        # Draw dual circles
        offset = 8
        pygame.draw.circle(self.image, primary_color, 
                          (center[0] - offset, center[1]), radius // 2)
        pygame.draw.circle(self.image, secondary_color, 
                          (center[0] + offset, center[1]), radius // 2)
        
        # Adversarial connection
        pygame.draw.line(self.image, COLORS['WHITE'], 
                        (center[0] - offset, center[1]), 
                        (center[0] + offset, center[1]), 3)
