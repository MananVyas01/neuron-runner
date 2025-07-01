"""
🎯 Challenge Modes - Real ML Simulation Events
Advanced challenge modes that simulate real ML scenarios like data shifts, adversarial attacks, and time constraints.
"""

import pygame
import random
import math
from .config import *

class ChallengeMode:
    def __init__(self, name, display_name, description, duration, effects, visual_changes):
        self.name = name
        self.display_name = display_name
        self.description = description
        self.duration = duration  # in seconds
        self.effects = effects
        self.visual_changes = visual_changes
        self.is_active = False
        self.timer = 0
        self.intensity = 1.0
        
    def activate(self):
        """Activate this challenge mode"""
        self.is_active = True
        self.timer = self.duration * FPS
        print(f"🚨 Challenge Activated: {self.display_name}")
        return True
    
    def deactivate(self):
        """Deactivate this challenge mode"""
        self.is_active = False
        self.timer = 0
        print(f"✅ Challenge Completed: {self.display_name}")
    
    def update(self):
        """Update challenge mode state"""
        if not self.is_active:
            return
            
        self.timer -= 1
        if self.timer <= 0:
            self.deactivate()
        else:
            # Calculate intensity (starts high, may vary)
            time_ratio = self.timer / (self.duration * FPS)
            if self.name == 'adversarial':
                # Adversarial mode gets more intense over time
                self.intensity = 1.5 - (time_ratio * 0.5)
            else:
                # Others maintain steady intensity
                self.intensity = 1.0
    
    def apply_effects(self, obstacle_manager, powerup_manager, background_manager):
        """Apply challenge-specific effects to game systems"""
        if not self.is_active:
            return
            
        if self.name == 'data_shift':
            # Change visual theme and obstacle patterns
            background_manager.set_theme(self.visual_changes['theme'])
            obstacle_manager.set_data_shift_mode(True)
            
        elif self.name == 'adversarial':
            # Make obstacle patterns tricky and unfair
            obstacle_manager.set_adversarial_mode(True, self.intensity)
            
        elif self.name == 'limited_epoch':
            # Time pressure mode - handled in main game loop
            pass

class ChallengeManager:
    def __init__(self):
        self.challenges = {
            'data_shift': ChallengeMode(
                name='data_shift',
                display_name='Data Distribution Shift',
                description='🌊 DOMAIN ADAPTATION: Environment and enemies change!\n• New visual theme\n• Different obstacle patterns\n• Tests model robustness\n• Duration: 30 seconds',
                duration=30,
                effects={
                    'visual_theme_change': True,
                    'obstacle_pattern_shift': True,
                    'difficulty_modifier': 1.2,
                },
                visual_changes={
                    'theme': 'cyber_space',
                    'color_shift': (50, -30, 100),  # RGB shifts
                    'pattern_change': 'adversarial'
                }
            ),
            
            'adversarial': ChallengeMode(
                name='adversarial',
                display_name='Adversarial Attack Mode',
                description='⚔️ ADVERSARIAL EXAMPLES: Obstacles become deceptive!\n• Tricky/unfair patterns\n• Visual noise effects\n• Simulates FGSM attacks\n• Duration: 20 seconds',
                duration=20,
                effects={
                    'adversarial_patterns': True,
                    'visual_noise': True,
                    'deceptive_spawning': True,
                    'difficulty_modifier': 1.5,
                },
                visual_changes={
                    'noise_level': 0.3,
                    'pattern_chaos': True,
                    'color_distortion': True
                }
            ),
            
            'limited_epoch': ChallengeMode(
                name='limited_epoch',
                display_name='Limited Training Time',
                description='⏰ TIME CONSTRAINT: Complete training under time limit!\n• 5 minute race mode\n• Accelerated scoring\n• Pressure mechanics\n• Duration: 300 seconds',
                duration=300,
                effects={
                    'time_pressure': True,
                    'accelerated_scoring': 2.0,
                    'bonus_powerups': True,
                    'difficulty_modifier': 0.8,  # Slightly easier to compensate for time pressure
                },
                visual_changes={
                    'timer_display': True,
                    'urgency_effects': True,
                    'speed_lines': True
                }
            ),
            
            'gradient_instability': ChallengeMode(
                name='gradient_instability',
                display_name='Gradient Instability',
                description='🌪️ TRAINING INSTABILITY: Chaotic gradient behavior!\n• Random control inversions\n• Explosive gradients\n• Vanishing effects\n• Duration: 25 seconds',
                duration=25,
                effects={
                    'control_chaos': True,
                    'explosive_gradients': True,
                    'random_freezes': True,
                    'difficulty_modifier': 1.3,
                },
                visual_changes={
                    'chaos_effects': True,
                    'gradient_visualization': True,
                    'instability_particles': True
                }
            )
        }
        
        self.active_challenge = None
        self.challenge_queue = []
        self.last_challenge_time = 0
        self.challenge_interval = 60 * FPS  # Every 60 seconds
        
        # UI elements (initialize later when pygame is ready)
        self.font_large = None
        self.font_medium = None
        self.font_small = None
        self.notification_timer = 0
        self.pending_challenge = None
        
    def _ensure_fonts(self):
        """Ensure fonts are initialized when needed"""
        if self.font_large is None:
            try:
                self.font_large = pygame.font.Font(None, 48)
                self.font_medium = pygame.font.Font(None, 32)
                self.font_small = pygame.font.Font(None, 24)
            except:
                # Fallback if fonts can't be loaded
                self.font_large = pygame.font.Font(None, 36)
                self.font_medium = pygame.font.Font(None, 24)
                self.font_small = pygame.font.Font(None, 20)
        
    def trigger_random_challenge(self, game_score):
        """Trigger a random challenge based on game progress"""
        if self.active_challenge:
            return False  # Don't trigger if one is already active
            
        # Determine available challenges based on score
        available_challenges = list(self.challenges.keys())
        
        # Higher scores unlock more challenging modes
        if game_score < 2000:
            available_challenges = ['data_shift', 'limited_epoch']
        elif game_score < 5000:
            available_challenges = ['data_shift', 'adversarial', 'limited_epoch']
        
        if available_challenges:
            challenge_name = random.choice(available_challenges)
            return self.activate_challenge(challenge_name)
        
        return False
    
    def activate_challenge(self, challenge_name):
        """Activate a specific challenge"""
        if challenge_name not in self.challenges:
            return False
            
        if self.active_challenge:
            self.active_challenge.deactivate()
        
        self.active_challenge = self.challenges[challenge_name]
        self.active_challenge.activate()
        self.notification_timer = 3 * FPS  # Show notification for 3 seconds
        return True
    
    def update(self, game_score, game_time):
        """Update challenge system"""
        # Update notification timer
        if self.notification_timer > 0:
            self.notification_timer -= 1
        
        # Update active challenge
        if self.active_challenge:
            self.active_challenge.update()
            if not self.active_challenge.is_active:
                self.active_challenge = None
        
        # Check for automatic challenge triggers
        if (game_time - self.last_challenge_time > self.challenge_interval and 
            not self.active_challenge and game_score > 1000):
            if self.trigger_random_challenge(game_score):
                self.last_challenge_time = game_time
    
    def apply_challenge_effects(self, obstacle_manager, powerup_manager, background_manager):
        """Apply effects of active challenge to game systems"""
        if self.active_challenge:
            self.active_challenge.apply_effects(obstacle_manager, powerup_manager, background_manager)
    
    def get_active_challenge(self):
        """Get currently active challenge"""
        return self.active_challenge
    
    def get_current_challenge(self):
        """Get the currently active challenge"""
        return self.active_challenge
    
    def draw_challenge_notification(self, screen):
        """Draw challenge activation notification"""
        if not self.active_challenge or self.notification_timer <= 0:
            return
        
        # Ensure fonts are loaded
        self._ensure_fonts()
        
        # Pulsing notification
        pulse = abs(math.sin(pygame.time.get_ticks() * 0.01)) * 0.3 + 0.7
        alpha = int(255 * pulse * (self.notification_timer / (3 * FPS)))
        
        # Notification background
        notif_width = 500
        notif_height = 120
        notif_x = (SCREEN_WIDTH - notif_width) // 2
        notif_y = 100
        
        # Background with challenge-specific color
        if self.active_challenge.name == 'data_shift':
            bg_color = (64, 156, 255)
        elif self.active_challenge.name == 'adversarial':
            bg_color = (255, 71, 87)
        elif self.active_challenge.name == 'limited_epoch':
            bg_color = (255, 193, 7)
        else:
            bg_color = (156, 39, 176)
        
        notif_bg = pygame.Surface((notif_width, notif_height))
        notif_bg.set_alpha(alpha)
        notif_bg.fill(bg_color)
        screen.blit(notif_bg, (notif_x, notif_y))
        
        # Border
        pygame.draw.rect(screen, COLORS['WHITE'], (notif_x, notif_y, notif_width, notif_height), 3)
        
        # Challenge icon and title
        title_text = self.font_large.render(f"🚨 {self.active_challenge.display_name.upper()}", True, COLORS['WHITE'])
        title_rect = title_text.get_rect(center=(SCREEN_WIDTH//2, notif_y + 30))
        screen.blit(title_text, title_rect)
        
        # Duration info
        remaining_time = self.active_challenge.timer // FPS
        duration_text = self.font_medium.render(f"Duration: {remaining_time}s remaining", True, COLORS['WHITE'])
        duration_rect = duration_text.get_rect(center=(SCREEN_WIDTH//2, notif_y + 70))
        screen.blit(duration_text, duration_rect)
    
    def draw_challenge_hud(self, screen):
        """Draw challenge-specific HUD elements"""
        if not self.active_challenge:
            return
        
        # Ensure fonts are loaded
        self._ensure_fonts()
        
        # Challenge status panel
        panel_x = SCREEN_WIDTH - 300
        panel_y = 200
        panel_width = 280
        panel_height = 100
        
        # Panel background
        panel_bg = pygame.Surface((panel_width, panel_height))
        panel_bg.set_alpha(150)
        panel_bg.fill(COLORS['BLACK'])
        screen.blit(panel_bg, (panel_x, panel_y))
        
        # Border with challenge color
        if self.active_challenge.name == 'data_shift':
            border_color = (64, 156, 255)
        elif self.active_challenge.name == 'adversarial':
            border_color = (255, 71, 87)
        elif self.active_challenge.name == 'limited_epoch':
            border_color = (255, 193, 7)
        else:
            border_color = (156, 39, 176)
        
        pygame.draw.rect(screen, border_color, (panel_x, panel_y, panel_width, panel_height), 3)
        
        # Challenge info
        title_text = self.font_small.render(f"ACTIVE: {self.active_challenge.display_name}", True, COLORS['WHITE'])
        screen.blit(title_text, (panel_x + 10, panel_y + 10))
        
        # Progress bar
        progress_width = panel_width - 20
        progress_height = 8
        progress_x = panel_x + 10
        progress_y = panel_y + 35
        
        # Background bar
        pygame.draw.rect(screen, COLORS['GRAY'], (progress_x, progress_y, progress_width, progress_height))
        
        # Progress fill
        progress_ratio = self.active_challenge.timer / (self.active_challenge.duration * FPS)
        fill_width = int(progress_width * progress_ratio)
        pygame.draw.rect(screen, border_color, (progress_x, progress_y, fill_width, progress_height))
        
        # Time remaining
        remaining_time = self.active_challenge.timer // FPS
        time_text = self.font_small.render(f"Time: {remaining_time}s", True, COLORS['WHITE'])
        screen.blit(time_text, (panel_x + 10, panel_y + 50))
        
        # Intensity indicator
        intensity_text = self.font_small.render(f"Intensity: {self.active_challenge.intensity:.1f}x", True, COLORS['WHITE'])
        screen.blit(intensity_text, (panel_x + 10, panel_y + 70))
    
    def handle_challenge_effects_on_player(self, player):
        """Apply challenge effects directly to player"""
        if not self.active_challenge:
            return
        
        challenge = self.active_challenge
        
        if challenge.name == 'adversarial':
            # Add visual noise to player
            if random.random() < 0.1:  # 10% chance per frame
                player.visual_noise_timer = 30  # 0.5 seconds of noise
        
        elif challenge.name == 'gradient_instability':
            # Random control effects
            if hasattr(player, 'gradient_chaos_timer'):
                player.gradient_chaos_timer -= 1
            else:
                player.gradient_chaos_timer = 0
                
            if player.gradient_chaos_timer <= 0 and random.random() < 0.05:  # 5% chance
                player.gradient_chaos_timer = random.randint(60, 180)  # 1-3 seconds
                player.control_inverted = True
    
    def get_score_multiplier(self):
        """Get score multiplier for active challenge"""
        if not self.active_challenge:
            return 1.0
        
        return self.active_challenge.effects.get('accelerated_scoring', 1.0)
    
    def should_show_selection_menu(self):
        """Check if challenge selection menu should be shown"""
        return False  # Challenges are automatic for now
    
    def reset(self):
        """Reset challenge system"""
        if self.active_challenge:
            self.active_challenge.deactivate()
        self.active_challenge = None
        self.last_challenge_time = 0
        self.notification_timer = 0

# Global challenge manager
challenge_manager = ChallengeManager()
