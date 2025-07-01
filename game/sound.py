"""
🔊 Sound Effects System
Audio feedback for enhanced gameplay experience.
"""

import pygame
import random
from .config import *


class SoundSystem:
    def __init__(self):
        self.sounds = {}
        self.enabled = True
        self.volume = SOUND_VOLUME

        # Try to load sounds (gracefully handle missing files)
        self.load_sounds()

    def load_sounds(self):
        """Load sound effects"""
        # For now, we'll create placeholder sound effects
        # In a full implementation, you would load actual audio files
        try:
            # Create simple beep sounds programmatically
            self.create_beep_sounds()
        except:
            print("Sound system: Using silent mode")
            self.enabled = False

    def create_beep_sounds(self):
        """Create simple beep sounds programmatically"""
        # Create different frequency sounds for different actions
        sample_rate = 22050
        duration = 0.1

        # Jump sound - ascending tone
        self.sounds["jump"] = self.create_tone(440, duration, sample_rate)

        # Powerup sound - happy tone
        self.sounds["powerup"] = self.create_tone(880, duration, sample_rate)

        # Damage sound - low tone
        self.sounds["damage"] = self.create_tone(220, duration, sample_rate)

        # Combo sound - high tone
        self.sounds["combo"] = self.create_tone(1320, duration * 0.5, sample_rate)

        # Achievement sound - chord
        self.sounds["achievement"] = self.create_chord(
            [440, 554, 659], duration, sample_rate
        )

    def create_tone(self, frequency, duration, sample_rate):
        """Create a simple tone"""
        import numpy as np

        try:
            frames = int(duration * sample_rate)
            arr = np.zeros(frames)

            for i in range(frames):
                arr[i] = 0.3 * np.sin(2 * np.pi * frequency * i / sample_rate)

            # Convert to pygame sound
            arr = (arr * 32767).astype(np.int16)
            sound = pygame.sndarray.make_sound(arr)
            return sound
        except:
            return None

    def create_chord(self, frequencies, duration, sample_rate):
        """Create a chord from multiple frequencies"""
        import numpy as np

        try:
            frames = int(duration * sample_rate)
            arr = np.zeros(frames)

            for freq in frequencies:
                for i in range(frames):
                    arr[i] += 0.1 * np.sin(2 * np.pi * freq * i / sample_rate)

            # Convert to pygame sound
            arr = (arr * 32767).astype(np.int16)
            sound = pygame.sndarray.make_sound(arr)
            return sound
        except:
            return None

    def play_sound(self, sound_name):
        """Play a sound effect"""
        if not self.enabled or sound_name not in self.sounds:
            return

        sound = self.sounds[sound_name]
        if sound:
            sound.set_volume(self.volume)
            sound.play()

    def play_jump(self):
        """Play jump sound"""
        self.play_sound("jump")

    def play_powerup(self):
        """Play powerup collection sound"""
        self.play_sound("powerup")

    def play_damage(self):
        """Play damage sound"""
        self.play_sound("damage")

    def play_combo(self):
        """Play combo achievement sound"""
        self.play_sound("combo")

    def play_achievement(self):
        """Play achievement unlock sound"""
        self.play_sound("achievement")

    def set_volume(self, volume):
        """Set master volume"""
        self.volume = max(0.0, min(1.0, volume))

    def toggle_enabled(self):
        """Toggle sound effects on/off"""
        self.enabled = not self.enabled


# Global sound system
try:
    sound_system = SoundSystem()
except:
    # Fallback if sound system fails to initialize
    class DummySoundSystem:
        def play_jump(self):
            pass

        def play_powerup(self):
            pass

        def play_damage(self):
            pass

        def play_combo(self):
            pass

        def play_achievement(self):
            pass

        def set_volume(self, volume):
            pass

        def toggle_enabled(self):
            pass

    sound_system = DummySoundSystem()
    print("Sound system: Using dummy sound system")
