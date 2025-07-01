"""
🏆 Achievement System
Track and display player achievements and milestones.
"""

import pygame
from .config import *

class Achievement:
    def __init__(self, name, description, icon, threshold, stat_key):
        self.name = name
        self.description = description
        self.icon = icon
        self.threshold = threshold
        self.stat_key = stat_key
        self.unlocked = False
        self.display_timer = 0
    
    def check_unlock(self, stats, player_status):
        """Check if achievement should be unlocked"""
        if self.unlocked:
            return False
        
        current_value = 0
        if self.stat_key == 'score':
            current_value = stats.score
        elif self.stat_key == 'combo':
            current_value = player_status.get('combo', 0)
        elif self.stat_key == 'perfect_dodges':
            current_value = player_status.get('perfect_dodges', 0)
        elif self.stat_key == 'epochs':
            current_value = stats.current_epoch
        elif self.stat_key == 'time_survived':
            current_value = stats.score // 60  # Convert to seconds
        
        if current_value >= self.threshold:
            self.unlocked = True
            self.display_timer = 5 * FPS  # Show for 5 seconds
            return True
        
        return False

class AchievementSystem:
    def __init__(self):
        self.achievements = [
            Achievement("First Steps", "Survive for 30 seconds", "🎯", 30, "time_survived"),
            Achievement("Neural Apprentice", "Reach 1000 training progress", "🧠", 1000, "score"),
            Achievement("Combo Master", "Achieve a 10x combo", "🔥", 10, "combo"),
            Achievement("Perfect Dodger", "Perform 20 perfect dodges", "💫", 20, "perfect_dodges"),
            Achievement("Epoch Expert", "Complete 5 epochs", "📊", 5, "epochs"),
            Achievement("Speed Demon", "Survive for 2 minutes", "⚡", 120, "time_survived"),
            Achievement("Neural Network", "Reach 5000 training progress", "🌐", 5000, "score"),
            Achievement("Combo Legend", "Achieve a 20x combo", "🔥", 20, "combo"),
            Achievement("Dodge Master", "Perform 50 perfect dodges", "💫", 50, "perfect_dodges"),
            Achievement("Training Veteran", "Survive for 5 minutes", "🏆", 300, "time_survived"),
        ]
        
        self.active_notifications = []
    
    def update(self, stats, player_status):
        """Update achievements and check for new unlocks"""
        for achievement in self.achievements:
            if achievement.check_unlock(stats, player_status):
                self.active_notifications.append(achievement)
                print(f"🏆 Achievement Unlocked: {achievement.name} - {achievement.description}")
                
                # Play achievement sound
                try:
                    from .sound import sound_system
                    sound_system.play_achievement()
                except ImportError:
                    pass
        
        # Update display timers
        for achievement in self.active_notifications[:]:
            achievement.display_timer -= 1
            if achievement.display_timer <= 0:
                self.active_notifications.remove(achievement)
    
    def draw(self, screen):
        """Draw achievement notifications"""
        y_offset = 200
        
        for achievement in self.active_notifications:
            # Achievement notification background
            notification_width = 350
            notification_height = 80
            x = SCREEN_WIDTH - notification_width - 20
            y = y_offset
            
            # Background with fade effect
            alpha = min(255, achievement.display_timer * 3)
            bg_surface = pygame.Surface((notification_width, notification_height))
            bg_surface.set_alpha(alpha)
            bg_surface.fill(COLORS['DATASET'])
            screen.blit(bg_surface, (x, y))
            
            # Border
            pygame.draw.rect(screen, COLORS['WHITE'], (x, y, notification_width, notification_height), 3)
            
            # Achievement text
            font = pygame.font.Font(None, 28)
            small_font = pygame.font.Font(None, 20)
            
            title_text = font.render(f"{achievement.icon} {achievement.name}", True, COLORS['WHITE'])
            desc_text = small_font.render(achievement.description, True, COLORS['WHITE'])
            
            screen.blit(title_text, (x + 10, y + 10))
            screen.blit(desc_text, (x + 10, y + 40))
            
            y_offset += 90
    
    def get_unlocked_count(self):
        """Get number of unlocked achievements"""
        return sum(1 for achievement in self.achievements if achievement.unlocked)
    
    def get_total_count(self):
        """Get total number of achievements"""
        return len(self.achievements)

# Global achievement system
achievement_system = AchievementSystem()
