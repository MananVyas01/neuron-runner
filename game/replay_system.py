"""
📹 Instant Replay + Share GIF System
Captures last 10 seconds of gameplay and creates shareable GIFs with stats overlay
"""

import pygame
import time
import os
from collections import deque
from threading import Thread
from .config import *

try:
    import imageio
    from PIL import Image, ImageDraw, ImageFont
    REPLAY_AVAILABLE = True
except ImportError:
    REPLAY_AVAILABLE = False
    print("⚠️ Replay system dependencies not available. Install imageio and Pillow for GIF recording.")

class GameFrame:
    """Represents a single frame of gameplay"""
    def __init__(self, surface, timestamp, stats, player_pos):
        self.surface = surface.copy()
        self.timestamp = timestamp
        self.stats = stats.copy() if hasattr(stats, 'copy') else stats
        self.player_pos = player_pos

class ReplaySystem:
    """Manages gameplay recording and GIF generation"""
    
    def __init__(self):
        self.recording = False
        self.frames = deque(maxlen=600)  # 10 seconds at 60 FPS
        self.max_recording_time = 10.0  # seconds
        self.fps = 60
        self.last_capture = 0
        self.capture_interval = 1.0 / 30  # Capture at 30 FPS to save memory
        self.recording_quality = "medium"  # low, medium, high
        
        # Create replays directory
        self.replay_dir = "replays"
        os.makedirs(self.replay_dir, exist_ok=True)
        
        print("📹 Replay System initialized!")
        if not REPLAY_AVAILABLE:
            print("   ⚠️  GIF export disabled - missing dependencies")
    
    def start_recording(self):
        """Start recording gameplay"""
        if not REPLAY_AVAILABLE:
            return False
        
        self.recording = True
        self.frames.clear()
        print("🔴 Recording started...")
        return True
    
    def stop_recording(self):
        """Stop recording gameplay"""
        self.recording = False
        print("⏹️ Recording stopped.")
    
    def capture_frame(self, screen, stats, player_pos):
        """Capture a frame of gameplay"""
        if not self.recording or not REPLAY_AVAILABLE:
            return
        
        current_time = time.time()
        if current_time - self.last_capture < self.capture_interval:
            return
        
        # Capture frame based on quality setting
        if self.recording_quality == "low":
            # Scale down for smaller file size
            scaled_surface = pygame.transform.scale(screen, (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
        elif self.recording_quality == "medium":
            # Moderate scaling
            scaled_surface = pygame.transform.scale(screen, (int(SCREEN_WIDTH * 0.75), int(SCREEN_HEIGHT * 0.75)))
        else:  # high quality
            scaled_surface = screen.copy()
        
        frame = GameFrame(
            surface=scaled_surface,
            timestamp=current_time,
            stats={
                'score': stats.score,
                'distance': stats.distance_traveled,
                'obstacles_dodged': stats.obstacles_dodged,
                'powerups_collected': stats.powerups_collected
            },
            player_pos=player_pos
        )
        
        self.frames.append(frame)
        self.last_capture = current_time
    
    def generate_replay_gif(self, stats, player_model="Default", challenge_mode=None):
        """Generate a GIF from recorded frames with stats overlay"""
        if not REPLAY_AVAILABLE or len(self.frames) == 0:
            return None
        
        print("🎬 Generating replay GIF...")
        
        # Create filename with timestamp and stats
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"neuron_runner_replay_{timestamp}_score_{stats.score}.gif"
        filepath = os.path.join(self.replay_dir, filename)
        
        # Generate GIF in a separate thread to avoid blocking
        Thread(target=self._create_gif_async, args=(filepath, stats, player_model, challenge_mode)).start()
        
        return filepath
    
    def _create_gif_async(self, filepath, stats, player_model, challenge_mode):
        """Create GIF asynchronously"""
        try:
            frames_for_gif = []
            
            for i, frame in enumerate(self.frames):
                # Convert pygame surface to PIL Image
                pil_image = self._pygame_to_pil(frame.surface)
                
                # Add stats overlay
                annotated_image = self._add_stats_overlay(
                    pil_image, frame.stats, player_model, challenge_mode, 
                    frame_number=i, total_frames=len(self.frames)
                )
                
                frames_for_gif.append(annotated_image)
            
            # Create GIF
            if frames_for_gif:
                duration = max(50, int(1000 / 30))  # 30 FPS, minimum 50ms per frame
                frames_for_gif[0].save(
                    filepath,
                    save_all=True,
                    append_images=frames_for_gif[1:],
                    duration=duration,
                    loop=0,
                    optimize=True
                )
                
                print(f"✅ Replay GIF saved: {filepath}")
                print(f"   📊 Final Score: {stats.score}")
                print(f"   🏃 Distance: {stats.distance_traveled:.0f}m")
                print(f"   🚧 Obstacles Dodged: {stats.obstacles_dodged}")
                print(f"   ⚡ Powerups Collected: {stats.powerups_collected}")
            
        except Exception as e:
            print(f"❌ Error creating replay GIF: {e}")
    
    def _pygame_to_pil(self, surface):
        """Convert pygame surface to PIL Image"""
        # Convert surface to RGB array
        rgb_array = pygame.surfarray.array3d(surface)
        # Transpose to get correct orientation
        rgb_array = rgb_array.transpose((1, 0, 2))
        # Convert to PIL Image
        return Image.fromarray(rgb_array)
    
    def _add_stats_overlay(self, image, stats, player_model, challenge_mode, frame_number, total_frames):
        """Add stats overlay to image"""
        # Create a copy to draw on
        overlay_image = image.copy()
        draw = ImageDraw.Draw(overlay_image)
        
        # Try to load a font, fall back to default if not available
        try:
            font = ImageFont.truetype("arial.ttf", 16)
            small_font = ImageFont.truetype("arial.ttf", 12)
        except:
            font = ImageFont.load_default()
            small_font = font
        
        # Background for stats overlay
        overlay_width = 200
        overlay_height = 120
        margin = 10
        
        # Draw semi-transparent background
        stats_rect = (margin, margin, margin + overlay_width, margin + overlay_height)
        draw.rectangle(stats_rect, fill=(0, 0, 0, 180))
        
        # Draw stats text
        y_offset = margin + 10
        text_color = (255, 255, 255)
        
        draw.text((margin + 10, y_offset), f"Score: {stats['score']}", fill=text_color, font=font)
        y_offset += 20
        
        draw.text((margin + 10, y_offset), f"Distance: {stats['distance']:.0f}m", fill=text_color, font=font)
        y_offset += 20
        
        draw.text((margin + 10, y_offset), f"Dodged: {stats['obstacles_dodged']}", fill=text_color, font=font)
        y_offset += 20
        
        draw.text((margin + 10, y_offset), f"Powerups: {stats['powerups_collected']}", fill=text_color, font=font)
        y_offset += 20
        
        # Add model and challenge info
        if player_model != "Default":
            draw.text((margin + 10, y_offset), f"Model: {player_model}", fill=(100, 255, 100), font=small_font)
            y_offset += 15
        
        if challenge_mode:
            draw.text((margin + 10, y_offset), f"Challenge: {challenge_mode}", fill=(255, 100, 100), font=small_font)
        
        # Add watermark
        watermark_text = "🧠 Neuron Runner"
        bbox = draw.textbbox((0, 0), watermark_text, font=small_font)
        watermark_width = bbox[2] - bbox[0]
        draw.text((image.width - watermark_width - 10, image.height - 25), 
                 watermark_text, fill=(128, 128, 128), font=small_font)
        
        # Progress bar at bottom
        progress = frame_number / max(1, total_frames - 1)
        bar_width = image.width - 20
        bar_height = 4
        bar_y = image.height - 10
        
        # Background bar
        draw.rectangle((10, bar_y, 10 + bar_width, bar_y + bar_height), fill=(64, 64, 64))
        # Progress bar
        draw.rectangle((10, bar_y, 10 + int(bar_width * progress), bar_y + bar_height), fill=(0, 255, 128))
        
        return overlay_image
    
    def create_highlight_reel(self, best_moments):
        """Create a highlight reel from best moments"""
        if not REPLAY_AVAILABLE:
            return None
        
        # Implementation for creating highlight reels from specific moments
        # This could include high scores, near misses, perfect runs, etc.
        pass
    
    def get_recording_status(self):
        """Get current recording status"""
        return {
            'recording': self.recording,
            'frame_count': len(self.frames),
            'duration': len(self.frames) / 30 if self.frames else 0,
            'quality': self.recording_quality,
            'available': REPLAY_AVAILABLE
        }
    
    def set_quality(self, quality):
        """Set recording quality: 'low', 'medium', 'high'"""
        if quality in ['low', 'medium', 'high']:
            self.recording_quality = quality
            print(f"📹 Recording quality set to: {quality}")
    
    def toggle_recording(self):
        """Toggle recording on/off"""
        if self.recording:
            self.stop_recording()
        else:
            self.start_recording()
        return self.recording

# Global replay system instance
replay_system = ReplaySystem()
