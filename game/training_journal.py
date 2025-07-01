"""
📊 Training Journal System
Persistent statistics, progress tracking, and achievement system
"""

import pygame
import json
import os
import time
from datetime import datetime, timedelta
from .config import *

class TrainingSession:
    """Represents a single training session (game run)"""
    
    def __init__(self):
        self.start_time = time.time()
        self.end_time = None
        self.score = 0
        self.distance = 0
        self.obstacles_dodged = 0
        self.powerups_collected = 0
        self.special_abilities_used = 0
        self.model_skin = "CNN"
        self.challenge_mode = None
        self.custom_model_bonuses = {}
        self.cause_of_termination = "collision"  # collision, quit, etc.
        
    def end_session(self, final_score, final_distance, obstacles, powerups, abilities, cause):
        """End the training session with final stats"""
        self.end_time = time.time()
        self.score = final_score
        self.distance = final_distance
        self.obstacles_dodged = obstacles
        self.powerups_collected = powerups
        self.special_abilities_used = abilities
        self.cause_of_termination = cause
    
    def get_duration(self):
        """Get session duration in seconds"""
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time
    
    def to_dict(self):
        """Convert to dictionary for serialization"""
        return {
            'start_time': self.start_time,
            'end_time': self.end_time,
            'score': self.score,
            'distance': self.distance,
            'obstacles_dodged': self.obstacles_dodged,
            'powerups_collected': self.powerups_collected,
            'special_abilities_used': self.special_abilities_used,
            'model_skin': self.model_skin,
            'challenge_mode': self.challenge_mode,
            'custom_model_bonuses': self.custom_model_bonuses,
            'cause_of_termination': self.cause_of_termination
        }

class Achievement:
    """Represents an unlockable achievement"""
    
    def __init__(self, id, name, description, condition_func, icon="🏆", rarity="common"):
        self.id = id
        self.name = name
        self.description = description
        self.condition_func = condition_func
        self.icon = icon
        self.rarity = rarity
        self.unlocked = False
        self.unlock_time = None
    
    def check_unlock(self, journal_data):
        """Check if achievement should be unlocked"""
        if not self.unlocked and self.condition_func(journal_data):
            self.unlocked = True
            self.unlock_time = time.time()
            return True
        return False
    
    def to_dict(self):
        """Convert to dictionary for serialization"""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'icon': self.icon,
            'rarity': self.rarity,
            'unlocked': self.unlocked,
            'unlock_time': self.unlock_time
        }

class TrainingJournal:
    """Persistent training journal and statistics system"""
    
    def __init__(self):
        self.save_file = "training_journal.json"
        self.current_session = None
        self.showing_journal = False
        self.journal_page = 0  # 0=stats, 1=sessions, 2=achievements
        
        # Persistent data
        self.total_sessions = 0
        self.total_score = 0
        self.total_distance = 0
        self.total_time_played = 0
        self.best_score = 0
        self.best_distance = 0
        self.longest_session = 0
        self.sessions_history = []
        self.model_skin_usage = {}
        self.challenge_mode_attempts = {}
        self.powerup_usage_stats = {}
        self.achievements = []
        
        # Daily/Weekly stats
        self.daily_stats = {}
        self.weekly_stats = {}
        
        # UI elements
        self.journal_rect = pygame.Rect(100, 50, SCREEN_WIDTH - 200, SCREEN_HEIGHT - 100)
        self.close_button = pygame.Rect(SCREEN_WIDTH - 150, 70, 60, 30)
        self.page_buttons = [
            pygame.Rect(120, 520, 100, 30),  # Stats
            pygame.Rect(240, 520, 100, 30),  # History
            pygame.Rect(360, 520, 120, 30)   # Achievements
        ]
        
        # Animation
        self.slide_progress = 0
        self.slide_speed = 4.0
        self.notification_queue = []
        
        # Load data and initialize achievements
        self.load_data()
        self.initialize_achievements()
    
    def initialize_achievements(self):
        """Initialize all achievements"""
        if not self.achievements:  # Only initialize if empty
            self.achievements = [
                # Score-based achievements
                Achievement("first_steps", "First Steps", "Complete your first training session", 
                           lambda d: d['total_sessions'] > 0, "👶", "common"),
                Achievement("novice_runner", "Novice Runner", "Score 1,000 points in a single run", 
                           lambda d: d['best_score'] >= 1000, "🏃", "common"),
                Achievement("skilled_navigator", "Skilled Navigator", "Score 5,000 points in a single run", 
                           lambda d: d['best_score'] >= 5000, "🎯", "uncommon"),
                Achievement("expert_neuron", "Expert Neuron", "Score 10,000 points in a single run", 
                           lambda d: d['best_score'] >= 10000, "🧠", "rare"),
                Achievement("ai_master", "AI Master", "Score 25,000 points in a single run", 
                           lambda d: d['best_score'] >= 25000, "👑", "legendary"),
                
                # Distance achievements
                Achievement("marathon_runner", "Marathon Runner", "Travel 10,000 units in total", 
                           lambda d: d['total_distance'] >= 10000, "🏃‍♂️", "common"),
                Achievement("ultra_marathoner", "Ultra Marathoner", "Travel 50,000 units in total", 
                           lambda d: d['total_distance'] >= 50000, "🏃‍♀️", "uncommon"),
                
                # Time-based achievements
                Achievement("dedicated_trainee", "Dedicated Trainee", "Play for 1 hour total", 
                           lambda d: d['total_time_played'] >= 3600, "⏰", "common"),
                Achievement("committed_researcher", "Committed Researcher", "Play for 5 hours total", 
                           lambda d: d['total_time_played'] >= 18000, "🔬", "uncommon"),
                
                # Session-based achievements
                Achievement("persistent_learner", "Persistent Learner", "Complete 10 training sessions", 
                           lambda d: d['total_sessions'] >= 10, "📚", "common"),
                Achievement("veteran_trainer", "Veteran Trainer", "Complete 50 training sessions", 
                           lambda d: d['total_sessions'] >= 50, "🎖️", "uncommon"),
                Achievement("training_legend", "Training Legend", "Complete 100 training sessions", 
                           lambda d: d['total_sessions'] >= 100, "🏆", "rare"),
                
                # Model skin achievements
                Achievement("skin_collector", "Skin Collector", "Use all 4 different model skins", 
                           lambda d: len(d.get('model_skin_usage', {})) >= 4, "🎨", "uncommon"),
                Achievement("transformer_master", "Transformer Master", "Complete 10 runs with Transformer skin", 
                           lambda d: d.get('model_skin_usage', {}).get('Transformer', 0) >= 10, "🤖", "uncommon"),
                
                # Special achievements
                Achievement("obstacle_master", "Obstacle Master", "Dodge 1000 obstacles total", 
                           lambda d: sum(s.get('obstacles_dodged', 0) for s in d.get('sessions_history', [])) >= 1000, "🚧", "rare"),
                Achievement("powerup_enthusiast", "Powerup Enthusiast", "Collect 500 powerups total", 
                           lambda d: sum(s.get('powerups_collected', 0) for s in d.get('sessions_history', [])) >= 500, "⚡", "uncommon"),
                Achievement("challenge_seeker", "Challenge Seeker", "Complete runs in all challenge modes", 
                           lambda d: len(d.get('challenge_mode_attempts', {})) >= 4, "⚔️", "rare"),
                
                # Streaks and consistency
                Achievement("daily_trainer", "Daily Trainer", "Play on 7 different days", 
                           lambda d: len(d.get('daily_stats', {})) >= 7, "📅", "uncommon"),
                Achievement("consistency_champion", "Consistency Champion", "Play for 30 different days", 
                           lambda d: len(d.get('daily_stats', {})) >= 30, "🔥", "legendary")
            ]
    
    def start_session(self, model_skin="CNN", challenge_mode=None):
        """Start a new training session"""
        self.current_session = TrainingSession()
        self.current_session.model_skin = model_skin
        self.current_session.challenge_mode = challenge_mode
        
        print(f"📊 Started training session with {model_skin}" + 
              (f" in {challenge_mode} mode" if challenge_mode else ""))
    
    def update_session(self, score, distance, obstacles, powerups, abilities):
        """Update current session stats"""
        if self.current_session:
            self.current_session.score = score
            self.current_session.distance = distance
            self.current_session.obstacles_dodged = obstacles
            self.current_session.powerups_collected = powerups
            self.current_session.special_abilities_used = abilities
    
    def end_session(self, cause="collision"):
        """End the current training session"""
        if not self.current_session:
            return
        
        # Finalize session data
        session_duration = self.current_session.get_duration()
        self.current_session.end_session(
            self.current_session.score,
            self.current_session.distance,
            self.current_session.obstacles_dodged,
            self.current_session.powerups_collected,
            self.current_session.special_abilities_used,
            cause
        )
        
        # Update aggregate stats
        self.total_sessions += 1
        self.total_score += self.current_session.score
        self.total_distance += self.current_session.distance
        self.total_time_played += session_duration
        
        # Update best records
        if self.current_session.score > self.best_score:
            self.best_score = self.current_session.score
        if self.current_session.distance > self.best_distance:
            self.best_distance = self.current_session.distance
        if session_duration > self.longest_session:
            self.longest_session = session_duration
        
        # Update model skin usage
        skin = self.current_session.model_skin
        self.model_skin_usage[skin] = self.model_skin_usage.get(skin, 0) + 1
        
        # Update challenge mode attempts
        if self.current_session.challenge_mode:
            mode = self.current_session.challenge_mode
            self.challenge_mode_attempts[mode] = self.challenge_mode_attempts.get(mode, 0) + 1
        
        # Update daily stats
        today = datetime.now().strftime("%Y-%m-%d")
        if today not in self.daily_stats:
            self.daily_stats[today] = {'sessions': 0, 'score': 0, 'time': 0}
        
        self.daily_stats[today]['sessions'] += 1
        self.daily_stats[today]['score'] += self.current_session.score
        self.daily_stats[today]['time'] += session_duration
        
        # Add to history (keep last 100 sessions)
        self.sessions_history.append(self.current_session.to_dict())
        if len(self.sessions_history) > 100:
            self.sessions_history.pop(0)
        
        # Check achievements
        self.check_achievements()
        
        # Save data
        self.save_data()
        
        print(f"📊 Training session ended: {self.current_session.score} points, " +
              f"{self.current_session.distance:.0f} distance, {session_duration:.1f}s")
        
        self.current_session = None
    
    def check_achievements(self):
        """Check and unlock achievements"""
        journal_data = {
            'total_sessions': self.total_sessions,
            'total_score': self.total_score,
            'total_distance': self.total_distance,
            'total_time_played': self.total_time_played,
            'best_score': self.best_score,
            'best_distance': self.best_distance,
            'longest_session': self.longest_session,
            'model_skin_usage': self.model_skin_usage,
            'challenge_mode_attempts': self.challenge_mode_attempts,
            'sessions_history': self.sessions_history,
            'daily_stats': self.daily_stats
        }
        
        for achievement in self.achievements:
            if achievement.check_unlock(journal_data):
                self.show_achievement_notification(achievement)
    
    def show_achievement_notification(self, achievement):
        """Show achievement unlock notification"""
        self.notification_queue.append({
            'type': 'achievement',
            'achievement': achievement,
            'timestamp': time.time()
        })
        print(f"🏆 Achievement Unlocked: {achievement.name}")
    
    def show_journal(self, page=0):
        """Show the training journal interface"""
        self.showing_journal = True
        self.journal_page = page
        self.slide_progress = 0
    
    def hide_journal(self):
        """Hide the training journal interface"""
        self.showing_journal = False
    
    def handle_event(self, event):
        """Handle pygame events"""
        if not self.showing_journal:
            return False
        
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left click
                pos = event.pos
                
                # Check close button
                if self.close_button.collidepoint(pos):
                    self.hide_journal()
                    return True
                
                # Check page buttons
                for i, button in enumerate(self.page_buttons):
                    if button.collidepoint(pos):
                        self.journal_page = i
                        return True
        
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.hide_journal()
                return True
            elif event.key == pygame.K_LEFT:
                self.journal_page = max(0, self.journal_page - 1)
                return True
            elif event.key == pygame.K_RIGHT:
                self.journal_page = min(2, self.journal_page + 1)
                return True
        
        return False
    
    def update(self):
        """Update the training journal"""
        if self.showing_journal:
            # Update slide animation
            if self.slide_progress < 1.0:
                self.slide_progress += self.slide_speed * (1/60)  # Assuming 60 FPS
                self.slide_progress = min(1.0, self.slide_progress)
        
        # Update notifications
        current_time = time.time()
        self.notification_queue = [n for n in self.notification_queue 
                                  if current_time - n['timestamp'] < 3.0]  # Keep for 3 seconds
    
    def draw(self, screen):
        """Draw the training journal interface"""
        if not self.showing_journal:
            return
        
        # Calculate slide position
        slide_offset = (1 - self.slide_progress) * 200
        current_x = self.journal_rect.x - slide_offset
        
        # Draw semi-transparent background
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        overlay.set_alpha(int(200 * self.slide_progress))
        overlay.fill(COLORS['DARK_BLUE'])
        screen.blit(overlay, (0, 0))
        
        # Draw journal background
        journal_rect = pygame.Rect(current_x, self.journal_rect.y, 
                                  self.journal_rect.width, self.journal_rect.height)
        
        pygame.draw.rect(screen, COLORS['WHITE'], journal_rect, border_radius=15)
        pygame.draw.rect(screen, COLORS['NEURON_BLUE'], journal_rect, 4, border_radius=15)
        
        # Draw title
        title_font = pygame.font.Font(None, 48)
        title_text = title_font.render("📊 Training Journal", True, COLORS['DARK_BLUE'])
        screen.blit(title_text, (journal_rect.x + 20, journal_rect.y + 20))
        
        # Draw close button
        close_rect = pygame.Rect(journal_rect.right - 80, journal_rect.y + 20, 60, 30)
        pygame.draw.rect(screen, COLORS['LOSS'], close_rect, border_radius=8)
        close_text = pygame.font.Font(None, 24).render("✕", True, COLORS['WHITE'])
        close_text_rect = close_text.get_rect(center=close_rect.center)
        screen.blit(close_text, close_text_rect)
        
        # Draw page buttons
        page_names = ["Stats", "History", "Achievements"]
        button_font = pygame.font.Font(None, 24)
        
        for i, (button, name) in enumerate(zip(self.page_buttons, page_names)):
            adjusted_button = pygame.Rect(button.x - slide_offset, button.y, button.width, button.height)
            
            if i == self.journal_page:
                pygame.draw.rect(screen, COLORS['NEURON_BLUE'], adjusted_button, border_radius=8)
                text_color = COLORS['WHITE']
            else:
                pygame.draw.rect(screen, COLORS['LIGHT_GRAY'], adjusted_button, border_radius=8)
                text_color = COLORS['DARK_BLUE']
            
            button_text = button_font.render(name, True, text_color)
            button_text_rect = button_text.get_rect(center=adjusted_button.center)
            screen.blit(button_text, button_text_rect)
        
        # Draw content based on current page
        content_rect = pygame.Rect(journal_rect.x + 20, journal_rect.y + 120, 
                                  journal_rect.width - 40, journal_rect.height - 180)
        
        if self.journal_page == 0:
            self.draw_stats_page(screen, content_rect)
        elif self.journal_page == 1:
            self.draw_history_page(screen, content_rect)
        elif self.journal_page == 2:
            self.draw_achievements_page(screen, content_rect)
    
    def draw_stats_page(self, screen, rect):
        """Draw the statistics page"""
        font = pygame.font.Font(None, 24)
        title_font = pygame.font.Font(None, 32)
        
        y_offset = 0
        
        # Overall stats
        title = title_font.render("📈 Overall Statistics", True, COLORS['NEURON_BLUE'])
        screen.blit(title, (rect.x, rect.y + y_offset))
        y_offset += 40
        
        stats = [
            f"Total Sessions: {self.total_sessions}",
            f"Total Score: {self.total_score:,}",
            f"Total Distance: {self.total_distance:,.0f}",
            f"Total Time Played: {self.format_time(self.total_time_played)}",
            f"Best Score: {self.best_score:,}",
            f"Best Distance: {self.best_distance:,.0f}",
            f"Longest Session: {self.format_time(self.longest_session)}",
            f"Average Score: {self.total_score // max(1, self.total_sessions):,}"
        ]
        
        for stat in stats:
            stat_surface = font.render(stat, True, COLORS['DARK_GRAY'])
            screen.blit(stat_surface, (rect.x + 20, rect.y + y_offset))
            y_offset += 25
        
        # Model skin usage
        y_offset += 20
        title = title_font.render("🎨 Model Skin Usage", True, COLORS['NEURON_BLUE'])
        screen.blit(title, (rect.x, rect.y + y_offset))
        y_offset += 40
        
        if self.model_skin_usage:
            for skin, count in self.model_skin_usage.items():
                percentage = (count / self.total_sessions) * 100 if self.total_sessions > 0 else 0
                skin_text = f"{skin}: {count} sessions ({percentage:.1f}%)"
                skin_surface = font.render(skin_text, True, COLORS['DARK_GRAY'])
                screen.blit(skin_surface, (rect.x + 20, rect.y + y_offset))
                y_offset += 25
        else:
            no_data = font.render("No model skin data yet", True, COLORS['GRAY'])
            screen.blit(no_data, (rect.x + 20, rect.y + y_offset))
        
        # Recent activity
        if rect.height > y_offset + 100:
            y_offset += 20
            title = title_font.render("📅 Recent Activity", True, COLORS['NEURON_BLUE'])
            screen.blit(title, (rect.x, rect.y + y_offset))
            y_offset += 40
            
            # Show last 7 days
            recent_days = list(self.daily_stats.keys())[-7:]
            for day in recent_days:
                stats = self.daily_stats[day]
                day_text = f"{day}: {stats['sessions']} sessions, {stats['score']:,} points"
                day_surface = font.render(day_text, True, COLORS['DARK_GRAY'])
                screen.blit(day_surface, (rect.x + 20, rect.y + y_offset))
                y_offset += 22
    
    def draw_history_page(self, screen, rect):
        """Draw the session history page"""
        font = pygame.font.Font(None, 20)
        title_font = pygame.font.Font(None, 32)
        
        title = title_font.render("📜 Session History", True, COLORS['NEURON_BLUE'])
        screen.blit(title, (rect.x, rect.y))
        
        if not self.sessions_history:
            no_data = font.render("No sessions recorded yet", True, COLORS['GRAY'])
            screen.blit(no_data, (rect.x + 20, rect.y + 50))
            return
        
        # Column headers
        headers = ["Date", "Score", "Distance", "Duration", "Model", "Mode"]
        header_x_positions = [0, 120, 200, 280, 360, 450]
        
        for header, x_pos in zip(headers, header_x_positions):
            header_surface = pygame.font.Font(None, 22).render(header, True, COLORS['NEURON_BLUE'])
            screen.blit(header_surface, (rect.x + x_pos, rect.y + 40))
        
        # Draw line under headers
        pygame.draw.line(screen, COLORS['LIGHT_GRAY'], 
                        (rect.x, rect.y + 60), (rect.x + rect.width - 20, rect.y + 60), 2)
        
        # Show last 15 sessions
        recent_sessions = self.sessions_history[-15:]
        for i, session in enumerate(reversed(recent_sessions)):
            y_pos = rect.y + 70 + i * 22
            
            # Alternate row colors
            if i % 2 == 0:
                row_rect = pygame.Rect(rect.x, y_pos - 2, rect.width - 20, 20)
                pygame.draw.rect(screen, COLORS['VERY_LIGHT_GRAY'], row_rect)
            
            # Format session data
            date_str = datetime.fromtimestamp(session['start_time']).strftime("%m/%d %H:%M")
            score_str = f"{session['score']:,}"
            distance_str = f"{session['distance']:.0f}"
            duration_str = self.format_time(session.get('end_time', session['start_time']) - session['start_time'])
            model_str = session.get('model_skin', 'Unknown')[:8]
            mode_str = session.get('challenge_mode', 'Normal')[:10]
            
            data = [date_str, score_str, distance_str, duration_str, model_str, mode_str]
            
            for data_item, x_pos in zip(data, header_x_positions):
                data_surface = font.render(data_item, True, COLORS['DARK_GRAY'])
                screen.blit(data_surface, (rect.x + x_pos, y_pos))
    
    def draw_achievements_page(self, screen, rect):
        """Draw the achievements page"""
        font = pygame.font.Font(None, 20)
        title_font = pygame.font.Font(None, 32)
        
        title = title_font.render("🏆 Achievements", True, COLORS['NEURON_BLUE'])
        screen.blit(title, (rect.x, rect.y))
        
        # Achievement stats
        unlocked_count = sum(1 for a in self.achievements if a.unlocked)
        total_count = len(self.achievements)
        progress_text = f"Unlocked: {unlocked_count}/{total_count} ({unlocked_count/total_count*100:.1f}%)"
        progress_surface = font.render(progress_text, True, COLORS['DARK_GRAY'])
        screen.blit(progress_surface, (rect.x, rect.y + 35))
        
        # Achievement list
        y_offset = 70
        for achievement in self.achievements:
            if y_offset > rect.height - 50:
                break
            
            # Achievement icon and name
            icon_color = COLORS['YELLOW'] if achievement.unlocked else COLORS['GRAY']
            icon_surface = pygame.font.Font(None, 24).render(achievement.icon, True, icon_color)
            screen.blit(icon_surface, (rect.x, rect.y + y_offset))
            
            # Achievement name
            name_color = COLORS['DARK_BLUE'] if achievement.unlocked else COLORS['GRAY']
            name_surface = pygame.font.Font(None, 22).render(achievement.name, True, name_color)
            screen.blit(name_surface, (rect.x + 30, rect.y + y_offset))
            
            # Achievement description
            desc_surface = font.render(achievement.description, True, COLORS['DARK_GRAY'])
            screen.blit(desc_surface, (rect.x + 30, rect.y + y_offset + 20))
            
            # Rarity indicator
            rarity_colors = {
                'common': COLORS['GRAY'],
                'uncommon': COLORS['GREEN'],
                'rare': COLORS['NEURON_BLUE'],
                'legendary': COLORS['PURPLE']
            }
            rarity_color = rarity_colors.get(achievement.rarity, COLORS['GRAY'])
            pygame.draw.circle(screen, rarity_color, (rect.x + rect.width - 30, rect.y + y_offset + 15), 8)
            
            y_offset += 50
    
    def draw_notifications(self, screen):
        """Draw achievement notifications"""
        for i, notification in enumerate(self.notification_queue):
            if notification['type'] == 'achievement':
                self.draw_achievement_notification(screen, notification, i)
    
    def draw_achievement_notification(self, screen, notification, index):
        """Draw a single achievement notification"""
        achievement = notification['achievement']
        elapsed = time.time() - notification['timestamp']
        
        # Animation
        if elapsed < 0.5:
            # Slide in
            progress = elapsed / 0.5
            x_offset = (1 - progress) * 300
        elif elapsed > 2.5:
            # Slide out
            progress = (elapsed - 2.5) / 0.5
            x_offset = progress * 300
        else:
            x_offset = 0
        
        # Notification position
        notification_rect = pygame.Rect(SCREEN_WIDTH - 350 + x_offset, 100 + index * 80, 300, 60)
        
        # Background
        pygame.draw.rect(screen, COLORS['DARK_BLUE'], notification_rect, border_radius=10)
        pygame.draw.rect(screen, COLORS['YELLOW'], notification_rect, 3, border_radius=10)
        
        # Icon
        icon_font = pygame.font.Font(None, 32)
        icon_surface = icon_font.render(achievement.icon, True, COLORS['YELLOW'])
        screen.blit(icon_surface, (notification_rect.x + 10, notification_rect.y + 15))
        
        # Text
        title_font = pygame.font.Font(None, 24)
        desc_font = pygame.font.Font(None, 18)
        
        title_text = title_font.render("Achievement Unlocked!", True, COLORS['WHITE'])
        screen.blit(title_text, (notification_rect.x + 50, notification_rect.y + 5))
        
        name_text = desc_font.render(achievement.name, True, COLORS['YELLOW'])
        screen.blit(name_text, (notification_rect.x + 50, notification_rect.y + 25))
        
        desc_text = desc_font.render(achievement.description[:40] + "...", True, COLORS['LIGHT_GRAY'])
        screen.blit(desc_text, (notification_rect.x + 50, notification_rect.y + 40))
    
    def format_time(self, seconds):
        """Format time in seconds to human readable format"""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds//60:.0f}m {seconds%60:.0f}s"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{hours:.0f}h {minutes:.0f}m"
    
    def save_data(self):
        """Save journal data to file"""
        try:
            data = {
                'total_sessions': self.total_sessions,
                'total_score': self.total_score,
                'total_distance': self.total_distance,
                'total_time_played': self.total_time_played,
                'best_score': self.best_score,
                'best_distance': self.best_distance,
                'longest_session': self.longest_session,
                'sessions_history': self.sessions_history,
                'model_skin_usage': self.model_skin_usage,
                'challenge_mode_attempts': self.challenge_mode_attempts,
                'powerup_usage_stats': self.powerup_usage_stats,
                'achievements': [a.to_dict() for a in self.achievements],
                'daily_stats': self.daily_stats,
                'weekly_stats': self.weekly_stats
            }
            
            with open(self.save_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            print(f"❌ Failed to save training journal: {e}")
    
    def load_data(self):
        """Load journal data from file"""
        try:
            if os.path.exists(self.save_file):
                with open(self.save_file, 'r') as f:
                    data = json.load(f)
                
                self.total_sessions = data.get('total_sessions', 0)
                self.total_score = data.get('total_score', 0)
                self.total_distance = data.get('total_distance', 0)
                self.total_time_played = data.get('total_time_played', 0)
                self.best_score = data.get('best_score', 0)
                self.best_distance = data.get('best_distance', 0)
                self.longest_session = data.get('longest_session', 0)
                self.sessions_history = data.get('sessions_history', [])
                self.model_skin_usage = data.get('model_skin_usage', {})
                self.challenge_mode_attempts = data.get('challenge_mode_attempts', {})
                self.powerup_usage_stats = data.get('powerup_usage_stats', {})
                self.daily_stats = data.get('daily_stats', {})
                self.weekly_stats = data.get('weekly_stats', {})
                
                # Load achievements
                achievement_data = data.get('achievements', [])
                if achievement_data:
                    # Match loaded achievements with initialized ones
                    for loaded_achievement in achievement_data:
                        for achievement in self.achievements:
                            if achievement.id == loaded_achievement['id']:
                                achievement.unlocked = loaded_achievement.get('unlocked', False)
                                achievement.unlock_time = loaded_achievement.get('unlock_time')
                                break
                
                print(f"📊 Loaded training journal: {self.total_sessions} sessions, {self.best_score} best score")
                
        except Exception as e:
            print(f"⚠️ Failed to load training journal: {e}")
    
    def get_summary_stats(self):
        """Get summary statistics for display"""
        return {
            'total_sessions': self.total_sessions,
            'best_score': self.best_score,
            'total_time': self.format_time(self.total_time_played),
            'achievements_unlocked': sum(1 for a in self.achievements if a.unlocked),
            'total_achievements': len(self.achievements)
        }

# Global training journal instance
training_journal = TrainingJournal()
