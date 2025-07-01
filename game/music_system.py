"""
🎵 Evolving Synthwave Soundtrack System
Dynamic music system that adapts to gameplay intensity and world themes
"""

import pygame
import math
import random
from .config import *

class MusicLayer:
    """Represents a single layer of the dynamic soundtrack"""
    
    def __init__(self, name, base_frequency, intensity_range, theme_affinity=None):
        self.name = name
        self.base_frequency = base_frequency
        self.intensity_range = intensity_range  # (min, max) intensity levels
        self.theme_affinity = theme_affinity or []  # Themes this layer works best with
        self.current_volume = 0.0
        self.target_volume = 0.0
        self.fade_speed = 2.0  # Volume change per second
        self.frequency_offset = 0.0
        self.phase = random.random() * math.pi * 2
        
    def update(self, dt, intensity, current_theme):
        """Update the music layer based on intensity and theme"""
        # Calculate target volume based on intensity and theme affinity
        base_volume = self.calculate_base_volume(intensity)
        theme_modifier = self.get_theme_modifier(current_theme)
        self.target_volume = base_volume * theme_modifier
        
        # Smooth volume transitions
        volume_diff = self.target_volume - self.current_volume
        if abs(volume_diff) > 0.01:
            self.current_volume += volume_diff * self.fade_speed * dt
        else:
            self.current_volume = self.target_volume
        
        # Update frequency offset for pitch bending
        self.frequency_offset = math.sin(pygame.time.get_ticks() * 0.001 + self.phase) * 0.1
        
    def calculate_base_volume(self, intensity):
        """Calculate base volume based on intensity"""
        min_intensity, max_intensity = self.intensity_range
        if intensity < min_intensity:
            return 0.0
        elif intensity > max_intensity:
            return 1.0
        else:
            return (intensity - min_intensity) / (max_intensity - min_intensity)
    
    def get_theme_modifier(self, current_theme):
        """Get volume modifier based on current theme"""
        if not self.theme_affinity:
            return 1.0
        
        if current_theme in self.theme_affinity:
            return 1.2  # Boost volume for matching themes
        else:
            return 0.8  # Reduce volume for non-matching themes
    
    def get_current_frequency(self):
        """Get current frequency with offset"""
        return self.base_frequency * (1.0 + self.frequency_offset)

class SynthwavePlayer:
    """Procedural synthwave music player using pygame's sound generation"""
    
    def __init__(self):
        self.sample_rate = 22050
        self.buffer_size = 1024
        self.playing = False
        self.master_volume = 0.3
        
        # Music layers
        self.layers = {
            'bass': MusicLayer('bass', 80, (0.0, 0.4), ['Neural Processing', 'Quantum ML']),
            'lead': MusicLayer('lead', 440, (0.3, 1.0), ['Computer Vision', 'Transformer Attention']),
            'pad': MusicLayer('pad', 220, (0.1, 0.8), ['Natural Language', 'Generative AI']),
            'arp': MusicLayer('arp', 880, (0.5, 1.0), ['Computer Vision', 'Quantum ML']),
            'drums': MusicLayer('drums', 60, (0.2, 1.0)),  # Kick drum
            'hihat': MusicLayer('hihat', 8000, (0.4, 1.0))  # Hi-hat
        }
        
        # Intensity calculation
        self.base_intensity = 0.3
        self.current_intensity = 0.3
        self.target_intensity = 0.3
        self.intensity_smoothing = 1.0
        
        # Initialize audio system
        self.initialized = False
    
    def initialize(self):
        """Initialize the synthwave player"""
        try:
            # Try to initialize pygame mixer for audio
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=self.sample_rate, size=-16, channels=2, buffer=self.buffer_size)
            self.initialized = True
            print("🎵 Synthwave player initialized")
        except Exception as e:
            print(f"⚠️  Audio initialization failed: {e}")
            self.initialized = False
    
    def generate_tone(self, frequency, duration, volume=1.0, wave_type='sine'):
        """Generate a tone with specified parameters"""
        frames = int(duration * self.sample_rate)
        arr = []
        
        for i in range(frames):
            time = float(i) / self.sample_rate
            
            if wave_type == 'sine':
                wave = math.sin(frequency * 2 * math.pi * time)
            elif wave_type == 'square':
                wave = 1.0 if math.sin(frequency * 2 * math.pi * time) > 0 else -1.0
            elif wave_type == 'sawtooth':
                wave = 2 * (time * frequency - math.floor(time * frequency + 0.5))
            elif wave_type == 'triangle':
                t = time * frequency
                wave = 2 * abs(2 * (t - math.floor(t + 0.5))) - 1
            else:
                wave = 0
            
            # Apply volume and convert to 16-bit
            sample = int(wave * volume * 32767 * self.master_volume)
            arr.append([sample, sample])  # Stereo
        
        return arr
    
    def update_intensity(self, intensity):
        """Update music intensity based on game state"""
        if not self.initialized:
            return
        
        self.target_intensity = max(0.1, min(1.0, intensity))
        # Smooth intensity changes
        self.current_intensity += (self.target_intensity - self.current_intensity) * 0.1
    
    def update(self, dt, score, player_status, current_theme, challenge_active=False, noise_level=0.0):
        """Update the music system"""
        # Update intensity
        self.update_intensity(score, player_status, challenge_active)
        
        # Smooth intensity transitions
        intensity_diff = self.target_intensity - self.current_intensity
        self.current_intensity += intensity_diff * self.intensity_smoothing * dt
        
        # Update beat timer
        beat_duration = 60.0 / self.bpm
        self.beat_timer += dt
        if self.beat_timer >= beat_duration:
            self.beat_timer = 0
            self.current_beat = (self.current_beat + 1) % 16  # 16-beat cycle
        
        # Update layers
        for layer in self.layers.values():
            layer.update(dt, self.current_intensity, current_theme)
        
        # Update effects based on game state
        self.update_effects(noise_level, player_status)
    
    def update_effects(self, noise_level, player_status):
        """Update audio effects based on game state"""
        # Low-pass filter during noise/confusion
        if noise_level > 0.3:
            self.low_pass_enabled = True
            self.low_pass_cutoff = max(0.3, 1.0 - noise_level)
        else:
            self.low_pass_enabled = False
            self.low_pass_cutoff = 1.0
        
        # Reverb adjustments
        if player_status.get('adrenaline', False):
            self.reverb_amount = 0.4  # More reverb during adrenaline
        else:
            self.reverb_amount = 0.2
    
    def get_intensity_visualization(self):
        """Get data for visualizing music intensity"""
        return {
            'current_intensity': self.current_intensity,
            'target_intensity': self.target_intensity,
            'beat_progress': self.beat_timer / (60.0 / self.bpm),
            'layer_volumes': {name: layer.current_volume for name, layer in self.layers.items()},
            'effects_active': {
                'low_pass': self.low_pass_enabled,
                'reverb': self.reverb_enabled
            }
        }
    
    def start(self):
        """Start the music system"""
        self.playing = True
        print("🎵 Synthwave soundtrack started")
    
    def stop(self):
        """Stop the music system"""
        self.playing = False
        print("🎵 Synthwave soundtrack stopped")
    
    def set_master_volume(self, volume):
        """Set master volume (0.0 to 1.0)"""
        self.master_volume = max(0.0, min(1.0, volume))
    
    def cleanup(self):
        """Clean up audio resources"""
        if self.initialized:
            try:
                pygame.mixer.quit()
            except:
                pass
            self.initialized = False

class MusicVisualizer:
    """Visual representation of the dynamic music system"""
    
    def __init__(self):
        self.show_visualizer = False
        self.visualization_mode = 'bars'  # 'bars', 'waveform', 'spectrum'
        self.history_length = 60  # frames
        self.intensity_history = []
        self.beat_pulse = 0
        
    def toggle(self):
        """Toggle visualizer on/off"""
        self.show_visualizer = not self.show_visualizer
    
    def update(self, intensity, theme=None):
        """Update visualizer with current music data"""
        # Create mock music data structure
        music_data = {
            'current_intensity': intensity,
            'beat_progress': (pygame.time.get_ticks() / 500) % 1.0  # Mock beat
        }
        
        # Track intensity history
        self.intensity_history.append(music_data['current_intensity'])
        if len(self.intensity_history) > self.history_length:
            self.intensity_history.pop(0)
        
        # Beat pulse effect
        beat_progress = music_data['beat_progress']
        if beat_progress < 0.1:  # Beat just happened
            self.beat_pulse = 1.0
        else:
            self.beat_pulse *= 0.9  # Decay
    
    def draw(self, screen, music_data):
        """Draw music visualizer"""
        if not self.show_visualizer:
            return
        
        if self.visualization_mode == 'bars':
            self.draw_bars(screen, music_data)
        elif self.visualization_mode == 'waveform':
            self.draw_waveform(screen, music_data)
        elif self.visualization_mode == 'spectrum':
            self.draw_spectrum(screen, music_data)
    
    def draw_bars(self, screen, music_data):
        """Draw bar-style visualizer"""
        visualizer_rect = pygame.Rect(20, SCREEN_HEIGHT - 120, 200, 80)
        
        # Background
        bg_surface = pygame.Surface((visualizer_rect.width, visualizer_rect.height), pygame.SRCALPHA)
        bg_surface.fill((*COLORS['BLACK'], 150))
        screen.blit(bg_surface, visualizer_rect)
        
        # Layer bars
        layer_names = list(music_data['layer_volumes'].keys())
        bar_width = visualizer_rect.width // len(layer_names) - 2
        
        for i, (layer_name, volume) in enumerate(music_data['layer_volumes'].items()):
            bar_x = visualizer_rect.x + i * (bar_width + 2)
            bar_height = int(volume * visualizer_rect.height * 0.8)
            bar_y = visualizer_rect.bottom - bar_height
            
            # Bar color based on layer type
            colors = {
                'bass': COLORS['LOSS'],
                'lead': COLORS['ACTIVATION'],
                'pad': COLORS['DATASET'],
                'arp': COLORS['GRADIENT'],
                'drums': COLORS['NEURON_BLUE'],
                'hihat': COLORS['WHITE']
            }
            bar_color = colors.get(layer_name, COLORS['GRAY'])
            
            # Add beat pulse effect
            if self.beat_pulse > 0.5:
                pulse_boost = int(self.beat_pulse * 20)
                bar_color = tuple(min(255, c + pulse_boost) for c in bar_color)
            
            pygame.draw.rect(screen, bar_color, (bar_x, bar_y, bar_width, bar_height))
            
            # Layer name
            font = pygame.font.Font(None, 16)
            name_text = font.render(layer_name[:3].upper(), True, COLORS['WHITE'])
            screen.blit(name_text, (bar_x, visualizer_rect.bottom + 5))
    
    def draw_waveform(self, screen, music_data):
        """Draw waveform-style visualizer"""
        if len(self.intensity_history) < 2:
            return
        
        visualizer_rect = pygame.Rect(20, SCREEN_HEIGHT - 60, 300, 40)
        
        # Background
        bg_surface = pygame.Surface((visualizer_rect.width, visualizer_rect.height), pygame.SRCALPHA)
        bg_surface.fill((*COLORS['BLACK'], 150))
        screen.blit(bg_surface, visualizer_rect)
        
        # Waveform
        points = []
        for i, intensity in enumerate(self.intensity_history):
            x = visualizer_rect.x + (i / len(self.intensity_history)) * visualizer_rect.width
            y = visualizer_rect.centery - (intensity - 0.5) * visualizer_rect.height
            points.append((x, y))
        
        if len(points) > 1:
            pygame.draw.lines(screen, COLORS['ACTIVATION'], False, points, 2)
    
    def draw_spectrum(self, screen, music_data):
        """Draw spectrum-style visualizer"""
        visualizer_rect = pygame.Rect(20, SCREEN_HEIGHT - 100, 250, 60)
        
        # Background
        bg_surface = pygame.Surface((visualizer_rect.width, visualizer_rect.height), pygame.SRCALPHA)
        bg_surface.fill((*COLORS['BLACK'], 150))
        screen.blit(bg_surface, visualizer_rect)
        
        # Frequency bands (simulated)
        band_count = 16
        band_width = visualizer_rect.width // band_count - 1
        
        for i in range(band_count):
            # Simulate frequency response
            frequency = 50 * (2 ** (i / 2))  # Log scale
            band_intensity = music_data['current_intensity'] * random.uniform(0.3, 1.0)
            
            band_height = int(band_intensity * visualizer_rect.height)
            band_x = visualizer_rect.x + i * (band_width + 1)
            band_y = visualizer_rect.bottom - band_height
            
            # Color based on frequency
            if frequency < 200:
                color = COLORS['LOSS']  # Bass - red
            elif frequency < 2000:
                color = COLORS['ACTIVATION']  # Mids - blue
            else:
                color = COLORS['DATASET']  # Highs - green
            
            pygame.draw.rect(screen, color, (band_x, band_y, band_width, band_height))

class DynamicMusicSystem:
    """Unified music system that combines synthwave player and visualizer"""
    
    def __init__(self):
        self.player = SynthwavePlayer()
        self.visualizer = MusicVisualizer()
        self.initialized = False
    
    def initialize(self):
        """Initialize the music system"""
        self.player.initialize()
        self.initialized = True
    
    def update(self, intensity, theme=None):
        """Update music based on game state"""
        if not self.initialized:
            return
        
        self.player.update_intensity(intensity)
        self.visualizer.update(intensity, theme)
    
    def toggle_visualization(self):
        """Toggle music visualization"""
        self.visualizer.toggle()
    
    def draw(self, screen):
        """Draw music visualizer"""
        if not self.initialized:
            return
        
        # Create mock music data for visualization
        mock_music_data = {
            'current_intensity': self.player.current_intensity,
            'layer_volumes': {
                'bass': self.player.current_intensity * 0.8,
                'lead': self.player.current_intensity * 0.6,
                'pad': self.player.current_intensity * 0.4,
                'arp': self.player.current_intensity * 0.9
            },
            'beat_progress': (pygame.time.get_ticks() / 500) % 1.0
        }
        
        self.visualizer.draw(screen, mock_music_data)
    
    def cleanup(self):
        """Clean up music resources"""
        self.player.cleanup()

# Global music system instances
synthwave_player = SynthwavePlayer()
music_visualizer = MusicVisualizer()
dynamic_music_system = DynamicMusicSystem()
