"""
🧠 Custom Model Builder Mini-Game - INDUSTRIAL LEVEL
Interactive drag-and-drop neural network architecture builder with real ML configuration
"""

import pygame
import math
import json
import os
from collections import OrderedDict
from .config import *

class LayerProperty:
    """Represents a configurable property of a layer"""
    def __init__(self, name, value, prop_type, min_val=None, max_val=None, options=None):
        self.name = name
        self.value = value
        self.type = prop_type  # 'int', 'float', 'bool', 'choice'
        self.min_val = min_val
        self.max_val = max_val
        self.options = options or []
        self.editing = False

class LayerComponent:
    """Represents a draggable neural network layer with real ML properties"""
    
    def __init__(self, layer_type, x, y):
        self.layer_type = layer_type
        self.x = x
        self.y = y
        self.width = 140
        self.height = 60
        self.dragging = False
        self.drag_offset_x = 0
        self.drag_offset_y = 0
        self.connected_to = None
        self.connected_from = []
        self.selected = False
        self.layer_id = f"{layer_type}_{id(self)}"
        
        # Advanced properties with real ML parameters
        self.properties = self.get_advanced_properties()
        self.performance_impact = self.calculate_performance_impact()
        
        # Visual feedback
        self.pulse_timer = 0
        self.error_state = None
        
    def get_advanced_properties(self):
        """Get comprehensive ML properties for each layer type"""
        properties = OrderedDict()
        
        if self.layer_type == 'Dense':
            properties['neurons'] = LayerProperty('neurons', 128, 'int', 1, 2048)
            properties['activation'] = LayerProperty('activation', 'relu', 'choice', 
                options=['relu', 'sigmoid', 'tanh', 'leaky_relu', 'swish', 'gelu'])
            properties['use_bias'] = LayerProperty('use_bias', True, 'bool')
            properties['kernel_init'] = LayerProperty('kernel_init', 'glorot_uniform', 'choice',
                options=['glorot_uniform', 'he_normal', 'xavier_normal', 'random_normal'])
            properties['l1_reg'] = LayerProperty('l1_reg', 0.0, 'float', 0.0, 0.1)
            properties['l2_reg'] = LayerProperty('l2_reg', 0.0, 'float', 0.0, 0.1)
            
        elif self.layer_type == 'Conv2D':
            properties['filters'] = LayerProperty('filters', 32, 'int', 1, 512)
            properties['kernel_size'] = LayerProperty('kernel_size', 3, 'int', 1, 11)
            properties['strides'] = LayerProperty('strides', 1, 'int', 1, 4)
            properties['padding'] = LayerProperty('padding', 'same', 'choice', options=['same', 'valid'])
            properties['activation'] = LayerProperty('activation', 'relu', 'choice',
                options=['relu', 'sigmoid', 'tanh', 'leaky_relu', 'swish'])
            properties['dilation_rate'] = LayerProperty('dilation_rate', 1, 'int', 1, 5)
            
        elif self.layer_type == 'LSTM':
            properties['units'] = LayerProperty('units', 64, 'int', 1, 512)
            properties['return_sequences'] = LayerProperty('return_sequences', True, 'bool')
            properties['return_state'] = LayerProperty('return_state', False, 'bool')
            properties['dropout'] = LayerProperty('dropout', 0.0, 'float', 0.0, 0.9)
            properties['recurrent_dropout'] = LayerProperty('recurrent_dropout', 0.0, 'float', 0.0, 0.9)
            properties['activation'] = LayerProperty('activation', 'tanh', 'choice',
                options=['tanh', 'sigmoid', 'relu', 'hard_sigmoid'])
            
        elif self.layer_type == 'Dropout':
            properties['rate'] = LayerProperty('rate', 0.2, 'float', 0.0, 0.9)
            properties['noise_shape'] = LayerProperty('noise_shape', 'None', 'choice',
                options=['None', 'spatial', 'channel'])
            
        elif self.layer_type == 'BatchNorm':
            properties['momentum'] = LayerProperty('momentum', 0.99, 'float', 0.0, 1.0)
            properties['epsilon'] = LayerProperty('epsilon', 0.001, 'float', 1e-8, 1e-2)
            properties['center'] = LayerProperty('center', True, 'bool')
            properties['scale'] = LayerProperty('scale', True, 'bool')
            
        elif self.layer_type == 'Attention':
            properties['heads'] = LayerProperty('heads', 8, 'int', 1, 32)
            properties['key_dim'] = LayerProperty('key_dim', 64, 'int', 8, 512)
            properties['value_dim'] = LayerProperty('value_dim', 64, 'int', 8, 512)
            properties['dropout'] = LayerProperty('dropout', 0.0, 'float', 0.0, 0.5)
            properties['use_bias'] = LayerProperty('use_bias', True, 'bool')
            
        elif self.layer_type == 'Embedding':
            properties['input_dim'] = LayerProperty('input_dim', 10000, 'int', 1, 100000)
            properties['output_dim'] = LayerProperty('output_dim', 128, 'int', 1, 1024)
            properties['mask_zero'] = LayerProperty('mask_zero', False, 'bool')
            properties['input_length'] = LayerProperty('input_length', 100, 'int', 1, 1000)
            
        elif self.layer_type == 'MaxPool2D':
            properties['pool_size'] = LayerProperty('pool_size', 2, 'int', 2, 8)
            properties['strides'] = LayerProperty('strides', 2, 'int', 1, 4)
            properties['padding'] = LayerProperty('padding', 'valid', 'choice', options=['valid', 'same'])
            
        elif self.layer_type == 'GlobalAvgPool2D':
            properties['keepdims'] = LayerProperty('keepdims', False, 'bool')
            
        elif self.layer_type == 'Flatten':
            # Flatten has no configurable parameters
            pass
            
        return properties
    
    def calculate_performance_impact(self):
        """Calculate the performance impact of this layer on gameplay"""
        impact = {
            'speed_modifier': 1.0,
            'accuracy_bonus': 0.0,
            'memory_cost': 1.0,
            'training_time': 1.0
        }
        
        if self.layer_type == 'Dense':
            neurons = self.properties.get('neurons', LayerProperty('neurons', 128, 'int')).value
            impact['memory_cost'] = neurons / 128.0
            impact['training_time'] = neurons / 128.0
            impact['accuracy_bonus'] = min(neurons / 512.0, 0.1)
            
        elif self.layer_type == 'Conv2D':
            filters = self.properties.get('filters', LayerProperty('filters', 32, 'int')).value
            kernel = self.properties.get('kernel_size', LayerProperty('kernel_size', 3, 'int')).value
            impact['memory_cost'] = (filters * kernel * kernel) / 288.0  # 32 * 3 * 3
            impact['speed_modifier'] = 1.0 + (filters / 128.0)
            
        elif self.layer_type == 'LSTM':
            units = self.properties.get('units', LayerProperty('units', 64, 'int')).value
            impact['memory_cost'] = units / 64.0 * 4  # LSTM has 4 gates
            impact['training_time'] = units / 64.0 * 2
            impact['accuracy_bonus'] = min(units / 256.0, 0.15)
            
        elif self.layer_type == 'Dropout':
            rate = self.properties.get('rate', LayerProperty('rate', 0.2, 'float')).value
            impact['accuracy_bonus'] = rate * 0.1  # Dropout helps generalization
            impact['speed_modifier'] = 1.0 - (rate * 0.1)
            
        elif self.layer_type == 'BatchNorm':
            impact['speed_modifier'] = 1.1  # BatchNorm speeds up training
            impact['accuracy_bonus'] = 0.05
            
        elif self.layer_type == 'Attention':
            heads = self.properties.get('heads', LayerProperty('heads', 8, 'int')).value
            key_dim = self.properties.get('key_dim', LayerProperty('key_dim', 64, 'int')).value
            impact['memory_cost'] = (heads * key_dim) / 512.0
            impact['accuracy_bonus'] = min(heads / 16.0, 0.2)
            impact['training_time'] = heads / 8.0
            
        return impact
    
    def get_rect(self):
        """Get the rectangle for collision detection"""
        return pygame.Rect(self.x, self.y, self.width, self.height)
    
    def handle_mouse_down(self, pos):
        """Handle mouse down event"""
        if self.get_rect().collidepoint(pos):
            self.dragging = True
            self.drag_offset_x = pos[0] - self.x
            self.drag_offset_y = pos[1] - self.y
            return True
        return False
    
    def handle_mouse_up(self):
        """Handle mouse up event"""
        self.dragging = False
    
    def handle_mouse_motion(self, pos):
        """Handle mouse motion when dragging"""
        if self.dragging:
            self.x = pos[0] - self.drag_offset_x
            self.y = pos[1] - self.drag_offset_y
    
    def draw(self, screen):
        """Draw the layer component"""
        rect = self.get_rect()
        
        # Layer colors based on type
        colors = {
            'Dense': COLORS['NEURON_BLUE'],
            'Conv2D': COLORS['ACTIVATION'],
            'LSTM': COLORS['DATASET'],
            'Dropout': COLORS['GRADIENT'],
            'BatchNorm': COLORS['LOSS'],
            'Attention': COLORS['PURPLE'],
            'Embedding': COLORS['ORANGE']
        }
        
        base_color = colors.get(self.layer_type, COLORS['WHITE'])
        
        # Draw shadow if dragging
        if self.dragging:
            shadow_rect = rect.copy()
            shadow_rect.x += 3
            shadow_rect.y += 3
            pygame.draw.rect(screen, COLORS['DARK_GRAY'], shadow_rect, border_radius=8)
        
        # Draw main rectangle
        pygame.draw.rect(screen, base_color, rect, border_radius=8)
        pygame.draw.rect(screen, COLORS['WHITE'], rect, 2, border_radius=8)
        
        # Draw layer type text
        font = pygame.font.Font(None, 24)
        text = font.render(self.layer_type, True, COLORS['WHITE'])
        text_rect = text.get_rect(center=rect.center)
        screen.blit(text, text_rect)
        
        # Draw connection points
        connection_color = COLORS['YELLOW'] if self.dragging else COLORS['LIGHT_GRAY']
        # Input connection (top)
        pygame.draw.circle(screen, connection_color, (rect.centerx, rect.top), 6)
        # Output connection (bottom)
        pygame.draw.circle(screen, connection_color, (rect.centerx, rect.bottom), 6)

class ModelBuilder:
    """Custom Model Builder mini-game"""
    
    def __init__(self):
        self.active = False
        self.layers = []
        self.available_layers = ['Dense', 'Conv2D', 'LSTM', 'Dropout', 'BatchNorm', 'Attention', 'Embedding']
        self.palette_x = 50
        self.palette_y = 100
        self.canvas_area = pygame.Rect(250, 50, 700, 500)
        self.selected_layer = None
        
        # Model configuration
        self.optimizer = 'Adam'
        self.learning_rate = 0.001
        self.batch_size = 32
        self.epochs = 10
        
        # Available options
        self.optimizers = ['Adam', 'SGD', 'RMSprop', 'Adagrad']
        self.optimizer_index = 0
        
        # UI elements
        self.build_button = pygame.Rect(950, 450, 100, 40)
        self.clear_button = pygame.Rect(950, 500, 100, 40)
        self.close_button = pygame.Rect(1000, 20, 60, 30)
        
        # Animation
        self.pulse_timer = 0
        
        # Results
        self.built_model = None
        self.model_stats = None
    
    def show(self):
        """Show the model builder interface"""
        self.active = True
        self.reset_builder()
    
    def hide(self):
        """Hide the model builder interface"""
        self.active = False
    
    def reset_builder(self):
        """Reset the builder to initial state"""
        self.layers = []
        self.selected_layer = None
        self.built_model = None
        self.model_stats = None
    
    def handle_event(self, event):
        """Handle pygame events"""
        if not self.active:
            return False
        
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left click
                pos = event.pos
                
                # Check close button
                if self.close_button.collidepoint(pos):
                    self.hide()
                    return True
                
                # Check build button
                if self.build_button.collidepoint(pos):
                    self.build_model()
                    return True
                
                # Check clear button
                if self.clear_button.collidepoint(pos):
                    self.reset_builder()
                    return True
                
                # Check palette (create new layer)
                if self.is_in_palette_area(pos):
                    layer_type = self.get_layer_from_palette(pos)
                    if layer_type:
                        new_layer = LayerComponent(layer_type, pos[0], pos[1])
                        self.layers.append(new_layer)
                        new_layer.dragging = True
                        new_layer.drag_offset_x = 10
                        new_layer.drag_offset_y = 10
                        return True
                
                # Check existing layers
                for layer in reversed(self.layers):  # Check from top to bottom
                    if layer.handle_mouse_down(pos):
                        # Move to front
                        self.layers.remove(layer)
                        self.layers.append(layer)
                        self.selected_layer = layer
                        return True
        
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:  # Left click release
                for layer in self.layers:
                    layer.handle_mouse_up()
        
        elif event.type == pygame.MOUSEMOTION:
            for layer in self.layers:
                layer.handle_mouse_motion(event.pos)
        
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.hide()
                return True
            elif event.key == pygame.K_DELETE and self.selected_layer:
                self.layers.remove(self.selected_layer)
                self.selected_layer = None
                return True
            elif event.key == pygame.K_TAB:
                # Cycle through optimizers
                self.optimizer_index = (self.optimizer_index + 1) % len(self.optimizers)
                self.optimizer = self.optimizers[self.optimizer_index]
                return True
        
        return False
    
    def is_in_palette_area(self, pos):
        """Check if position is in the layer palette area"""
        palette_rect = pygame.Rect(self.palette_x, self.palette_y, 150, len(self.available_layers) * 50)
        return palette_rect.collidepoint(pos)
    
    def get_layer_from_palette(self, pos):
        """Get layer type from palette position"""
        relative_y = pos[1] - self.palette_y
        layer_index = relative_y // 50
        if 0 <= layer_index < len(self.available_layers):
            return self.available_layers[layer_index]
        return None
    
    def build_model(self):
        """Build the custom model and calculate stats"""
        if len(self.layers) < 2:
            return  # Need at least 2 layers
        
        # Sort layers by Y position (top to bottom)
        sorted_layers = sorted(self.layers, key=lambda l: l.y)
        
        # Calculate model complexity
        total_params = 0
        model_depth = len(sorted_layers)
        
        for i, layer in enumerate(sorted_layers):
            layer_params = self.calculate_layer_params(layer, i)
            total_params += layer_params
        
        # Calculate model performance based on architecture
        self.model_stats = self.evaluate_model_architecture(sorted_layers, total_params)
        
        # Create built model info
        self.built_model = {
            'layers': [layer.layer_type for layer in sorted_layers],
            'params': total_params,
            'depth': model_depth,
            'optimizer': self.optimizer,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'stats': self.model_stats
        }
        
        print(f"🧠 Built custom model: {model_depth} layers, {total_params:,} parameters")
    
    def calculate_layer_params(self, layer, index):
        """Calculate approximate parameters for a layer"""
        base_params = {
            'Dense': 128 * 128,
            'Conv2D': 32 * 3 * 3 * 32,
            'LSTM': 64 * 64 * 4,
            'Dropout': 0,
            'BatchNorm': 128 * 2,
            'Attention': 64 * 64 * 3,
            'Embedding': 10000 * 128
        }
        return base_params.get(layer.layer_type, 1000)
    
    def evaluate_model_architecture(self, layers, total_params):
        """Evaluate the custom model architecture"""
        stats = {
            'accuracy_boost': 0,
            'speed_boost': 0,
            'memory_usage': 0,
            'training_stability': 0,
            'special_abilities': []
        }
        
        layer_types = [layer.layer_type for layer in layers]
        
        # Accuracy boost based on architecture
        if 'Conv2D' in layer_types and 'Dense' in layer_types:
            stats['accuracy_boost'] += 15  # Good for image processing
        if 'LSTM' in layer_types:
            stats['accuracy_boost'] += 10  # Good for sequences
        if 'Attention' in layer_types:
            stats['accuracy_boost'] += 20  # Transformer power
        if 'BatchNorm' in layer_types:
            stats['accuracy_boost'] += 8   # Training stability
            stats['training_stability'] += 15
        
        # Speed considerations
        if 'Dropout' in layer_types:
            stats['speed_boost'] += 5  # Regularization efficiency
        if total_params < 100000:
            stats['speed_boost'] += 10  # Lightweight model
        elif total_params > 1000000:
            stats['speed_boost'] -= 5   # Heavy model
        
        # Memory usage
        stats['memory_usage'] = min(100, total_params // 10000)
        
        # Special abilities based on architecture
        if 'Conv2D' in layer_types:
            stats['special_abilities'].append('Pattern Recognition')
        if 'LSTM' in layer_types:
            stats['special_abilities'].append('Memory Retention')
        if 'Attention' in layer_types:
            stats['special_abilities'].append('Focus Beam')
        if 'Dropout' in layer_types:
            stats['special_abilities'].append('Noise Immunity')
        
        # Clamp values
        stats['accuracy_boost'] = max(0, min(50, stats['accuracy_boost']))
        stats['speed_boost'] = max(-20, min(30, stats['speed_boost']))
        stats['training_stability'] = max(0, min(50, stats['training_stability']))
        
        return stats
    
    def get_gameplay_bonuses(self):
        """Get gameplay bonuses from the built model"""
        if not self.built_model:
            return {}
        
        stats = self.built_model['stats']
        bonuses = {}
        
        # Convert stats to gameplay effects
        if stats['accuracy_boost'] > 10:
            bonuses['score_multiplier'] = 1 + (stats['accuracy_boost'] / 100)
        
        if stats['speed_boost'] > 5:
            bonuses['movement_speed'] = 1 + (stats['speed_boost'] / 100)
        
        if stats['training_stability'] > 10:
            bonuses['damage_resistance'] = stats['training_stability'] / 100
        
        if 'Pattern Recognition' in stats['special_abilities']:
            bonuses['obstacle_prediction'] = True
        
        if 'Memory Retention' in stats['special_abilities']:
            bonuses['powerup_duration'] = 1.5
        
        if 'Focus Beam' in stats['special_abilities']:
            bonuses['precision_mode'] = True
        
        if 'Noise Immunity' in stats['special_abilities']:
            bonuses['noise_resistance'] = True
        
        return bonuses
    
    def update(self):
        """Update the model builder"""
        if not self.active:
            return
        
        self.pulse_timer += 0.1
    
    def draw(self, screen):
        """Draw the model builder interface"""
        if not self.active:
            return
        
        # Draw semi-transparent background
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        overlay.set_alpha(220)
        overlay.fill(COLORS['DARK_BLUE'])
        screen.blit(overlay, (0, 0))
        
        # Draw title
        title_font = pygame.font.Font(None, 48)
        title_text = title_font.render("🧠 Custom Model Builder", True, COLORS['WHITE'])
        screen.blit(title_text, (50, 20))
        
        # Draw canvas area
        pygame.draw.rect(screen, COLORS['DARK_GRAY'], self.canvas_area)
        pygame.draw.rect(screen, COLORS['WHITE'], self.canvas_area, 2)
        
        # Draw grid in canvas
        self.draw_grid(screen)
        
        # Draw layer palette
        self.draw_layer_palette(screen)
        
        # Draw connections between layers
        self.draw_connections(screen)
        
        # Draw layers
        for layer in self.layers:
            layer.draw(screen)
        
        # Highlight selected layer
        if self.selected_layer:
            rect = self.selected_layer.get_rect()
            pygame.draw.rect(screen, COLORS['YELLOW'], rect, 3, border_radius=8)
        
        # Draw controls
        self.draw_controls(screen)
        
        # Draw model stats if built
        if self.built_model:
            self.draw_model_stats(screen)
    
    def draw_grid(self, screen):
        """Draw grid in canvas area"""
        grid_size = 25
        grid_color = (*COLORS['LIGHT_GRAY'], 50)
        
        # Vertical lines
        for x in range(self.canvas_area.left, self.canvas_area.right, grid_size):
            pygame.draw.line(screen, grid_color, 
                           (x, self.canvas_area.top), 
                           (x, self.canvas_area.bottom))
        
        # Horizontal lines
        for y in range(self.canvas_area.top, self.canvas_area.bottom, grid_size):
            pygame.draw.line(screen, grid_color, 
                           (self.canvas_area.left, y), 
                           (self.canvas_area.right, y))
    
    def draw_layer_palette(self, screen):
        """Draw the layer palette"""
        palette_title = pygame.font.Font(None, 32).render("Layer Palette", True, COLORS['WHITE'])
        screen.blit(palette_title, (self.palette_x, self.palette_y - 30))
        
        for i, layer_type in enumerate(self.available_layers):
            y = self.palette_y + i * 50
            rect = pygame.Rect(self.palette_x, y, 150, 40)
            
            # Layer colors
            colors = {
                'Dense': COLORS['NEURON_BLUE'],
                'Conv2D': COLORS['ACTIVATION'],
                'LSTM': COLORS['DATASET'],
                'Dropout': COLORS['GRADIENT'],
                'BatchNorm': COLORS['LOSS'],
                'Attention': COLORS['PURPLE'],
                'Embedding': COLORS['ORANGE']
            }
            
            color = colors.get(layer_type, COLORS['WHITE'])
            
            # Hover effect
            mouse_pos = pygame.mouse.get_pos()
            if rect.collidepoint(mouse_pos):
                pygame.draw.rect(screen, color, rect, border_radius=8)
                pygame.draw.rect(screen, COLORS['YELLOW'], rect, 3, border_radius=8)
            else:
                pygame.draw.rect(screen, color, rect, border_radius=8)
                pygame.draw.rect(screen, COLORS['WHITE'], rect, 2, border_radius=8)
            
            # Layer type text
            font = pygame.font.Font(None, 24)
            text = font.render(layer_type, True, COLORS['WHITE'])
            text_rect = text.get_rect(center=rect.center)
            screen.blit(text, text_rect)
    
    def draw_connections(self, screen):
        """Draw connections between layers"""
        if len(self.layers) < 2:
            return
        
        # Sort layers by Y position
        sorted_layers = sorted(self.layers, key=lambda l: l.y)
        
        # Draw connections
        for i in range(len(sorted_layers) - 1):
            current = sorted_layers[i]
            next_layer = sorted_layers[i + 1]
            
            # Connection line
            start_pos = (current.x + current.width // 2, current.y + current.height)
            end_pos = (next_layer.x + next_layer.width // 2, next_layer.y)
            
            pygame.draw.line(screen, COLORS['ACTIVATION'], start_pos, end_pos, 3)
            
            # Arrow head
            self.draw_arrow_head(screen, start_pos, end_pos)
    
    def draw_arrow_head(self, screen, start, end):
        """Draw arrow head for connections"""
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.sqrt(dx*dx + dy*dy)
        
        if length == 0:
            return
        
        # Normalize
        dx /= length
        dy /= length
        
        # Arrow head points
        arrow_length = 10
        arrow_angle = 0.5
        
        x1 = end[0] - arrow_length * (dx * math.cos(arrow_angle) - dy * math.sin(arrow_angle))
        y1 = end[1] - arrow_length * (dy * math.cos(arrow_angle) + dx * math.sin(arrow_angle))
        x2 = end[0] - arrow_length * (dx * math.cos(-arrow_angle) - dy * math.sin(-arrow_angle))
        y2 = end[1] - arrow_length * (dy * math.cos(-arrow_angle) + dx * math.sin(-arrow_angle))
        
        pygame.draw.polygon(screen, COLORS['ACTIVATION'], [end, (x1, y1), (x2, y2)])
    
    def draw_controls(self, screen):
        """Draw control buttons and settings"""
        font = pygame.font.Font(None, 24)
        
        # Optimizer selection
        optimizer_text = f"Optimizer: {self.optimizer} (Tab to change)"
        optimizer_surface = font.render(optimizer_text, True, COLORS['WHITE'])
        screen.blit(optimizer_surface, (950, 100))
        
        # Learning rate
        lr_text = f"Learning Rate: {self.learning_rate}"
        lr_surface = font.render(lr_text, True, COLORS['WHITE'])
        screen.blit(lr_surface, (950, 130))
        
        # Batch size
        batch_text = f"Batch Size: {self.batch_size}"
        batch_surface = font.render(batch_text, True, COLORS['WHITE'])
        screen.blit(batch_surface, (950, 160))
        
        # Build button
        build_color = COLORS['ACTIVATION'] if len(self.layers) >= 2 else COLORS['GRAY']
        pygame.draw.rect(screen, build_color, self.build_button, border_radius=8)
        pygame.draw.rect(screen, COLORS['WHITE'], self.build_button, 2, border_radius=8)
        build_text = font.render("BUILD", True, COLORS['WHITE'])
        build_rect = build_text.get_rect(center=self.build_button.center)
        screen.blit(build_text, build_rect)
        
        # Clear button
        pygame.draw.rect(screen, COLORS['LOSS'], self.clear_button, border_radius=8)
        pygame.draw.rect(screen, COLORS['WHITE'], self.clear_button, 2, border_radius=8)
        clear_text = font.render("CLEAR", True, COLORS['WHITE'])
        clear_rect = clear_text.get_rect(center=self.clear_button.center)
        screen.blit(clear_text, clear_rect)
        
        # Close button
        pygame.draw.rect(screen, COLORS['DARK_GRAY'], self.close_button, border_radius=8)
        pygame.draw.rect(screen, COLORS['WHITE'], self.close_button, 2, border_radius=8)
        close_text = font.render("✕", True, COLORS['WHITE'])
        close_rect = close_text.get_rect(center=self.close_button.center)
        screen.blit(close_text, close_rect)
        
        # Instructions
        instructions = [
            "• Drag layers from palette to canvas",
            "• Delete selected layer with DEL key",
            "• Connect layers top to bottom",
            "• Build model when ready"
        ]
        
        for i, instruction in enumerate(instructions):
            inst_surface = font.render(instruction, True, COLORS['LIGHT_GRAY'])
            screen.blit(inst_surface, (950, 200 + i * 25))
    
    def draw_model_stats(self, screen):
        """Draw built model statistics"""
        if not self.built_model:
            return
        
        stats_area = pygame.Rect(950, 320, 200, 120)
        pygame.draw.rect(screen, COLORS['DARK_BLUE'], stats_area, border_radius=8)
        pygame.draw.rect(screen, COLORS['WHITE'], stats_area, 2, border_radius=8)
        
        font = pygame.font.Font(None, 20)
        stats = self.built_model['stats']
        
        # Title
        title = font.render("Model Performance", True, COLORS['WHITE'])
        screen.blit(title, (stats_area.x + 10, stats_area.y + 10))
        
        # Stats
        y_offset = 35
        stat_texts = [
            f"Accuracy: +{stats['accuracy_boost']}%",
            f"Speed: {stats['speed_boost']:+d}%",
            f"Memory: {stats['memory_usage']}%",
            f"Stability: +{stats['training_stability']}%"
        ]
        
        for stat_text in stat_texts:
            stat_surface = font.render(stat_text, True, COLORS['LIGHT_GRAY'])
            screen.blit(stat_surface, (stats_area.x + 10, stats_area.y + y_offset))
            y_offset += 20

# Global model builder instance
model_builder = ModelBuilder()
