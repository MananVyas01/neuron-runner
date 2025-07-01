"""
⚡ ML-themed Powerups
Datasets, optimizers, and special layers to boost performance.
"""

import pygame
import random
import math
from .config import *


class Powerup(pygame.sprite.Sprite):
    def __init__(self, powerup_type, x_pos, speed_multiplier=1.0):
        super().__init__()
        self.type = powerup_type
        self.effect = POWERUP_TYPES[powerup_type]["effect"]
        self.color = POWERUP_TYPES[powerup_type]["color"]

        # Create visual representation
        self.create_sprite()
        self.rect.x = x_pos
        self.rect.y = (
            GROUND_Y - self.rect.height - random.randint(0, 100)
        )  # Vary height

        self.speed = POWERUP_BASE_SPEED * speed_multiplier
        self.animation_timer = 0
        self.float_offset = random.uniform(0, math.pi * 2)  # For floating animation
        self.base_y = self.rect.y

    def create_sprite(self):
        """Create the visual representation of the powerup"""
        if self.type == "DATASET":
            # Draw as a golden database/file icon with better visibility
            self.image = pygame.Surface((50, 60), pygame.SRCALPHA)

            # Add glow effect
            glow_surface = pygame.Surface((60, 70), pygame.SRCALPHA)
            glow_color = (*self.color, 80)
            pygame.draw.circle(glow_surface, glow_color, (30, 35), 30)
            self.image.blit(glow_surface, (-5, -5))

            # Main body
            pygame.draw.rect(self.image, self.color, (8, 8, 34, 44))
            pygame.draw.rect(self.image, COLORS["WHITE"], (8, 8, 34, 44), 3)

            # Data lines with better spacing
            for i in range(5):
                y_pos = 18 + i * 6
                pygame.draw.line(
                    self.image, COLORS["WHITE"], (12, y_pos), (38, y_pos), 2
                )

            # Corner fold
            pygame.draw.polygon(
                self.image, COLORS["WHITE"], [(32, 8), (42, 18), (32, 18)]
            )

            # Add "DATA" label
            font = pygame.font.Font(None, 16)
            text = font.render("DATA", True, COLORS["BLACK"])
            self.image.blit(text, (12, 35))

        elif self.type == "OPTIMIZER":
            # Draw as a purple gear/cog with better animation
            self.image = pygame.Surface((55, 55), pygame.SRCALPHA)
            center = (27, 27)

            # Add glow effect
            glow_surface = pygame.Surface((65, 65), pygame.SRCALPHA)
            glow_color = (*self.color, 60)
            pygame.draw.circle(glow_surface, glow_color, (32, 32), 32)
            self.image.blit(glow_surface, (-5, -5))

            # Outer gear teeth (more detailed)
            for angle in range(0, 360, 30):  # More teeth
                rad = math.radians(angle)
                x1 = center[0] + math.cos(rad) * 24
                y1 = center[1] + math.sin(rad) * 24
                x2 = center[0] + math.cos(rad) * 18
                y2 = center[1] + math.sin(rad) * 18
                pygame.draw.line(self.image, self.color, (x1, y1), (x2, y2), 5)
                pygame.draw.line(self.image, COLORS["WHITE"], (x1, y1), (x2, y2), 2)

            # Inner circle
            pygame.draw.circle(self.image, self.color, center, 15)
            pygame.draw.circle(self.image, COLORS["WHITE"], center, 15, 3)
            pygame.draw.circle(self.image, COLORS["WHITE"], center, 6)

            # Add "OPT" label
            font = pygame.font.Font(None, 16)
            text = font.render("OPT", True, COLORS["WHITE"])
            text_rect = text.get_rect(center=center)
            self.image.blit(text, text_rect)

        elif self.type == "DROPOUT":
            # Draw as a green shield with network pattern
            self.image = pygame.Surface((60, 70), pygame.SRCALPHA)

            # Add glow effect
            glow_surface = pygame.Surface((70, 80), pygame.SRCALPHA)
            glow_color = (*self.color, 70)
            pygame.draw.circle(glow_surface, glow_color, (35, 40), 35)
            self.image.blit(glow_surface, (-5, -5))

            # Shield shape
            points = [(30, 5), (50, 20), (50, 45), (30, 65), (10, 45), (10, 20)]
            pygame.draw.polygon(self.image, self.color, points)
            pygame.draw.polygon(self.image, COLORS["WHITE"], points, 4)

            # Network pattern inside (more detailed)
            nodes = [(30, 25), (20, 35), (40, 35), (30, 45), (15, 50), (45, 50)]
            for node in nodes:
                pygame.draw.circle(self.image, COLORS["WHITE"], node, 3)

            # Connections between nodes
            connections = [
                (nodes[0], nodes[1]),
                (nodes[0], nodes[2]),
                (nodes[1], nodes[3]),
                (nodes[2], nodes[3]),
                (nodes[1], nodes[4]),
                (nodes[2], nodes[5]),
            ]
            for start, end in connections:
                pygame.draw.line(self.image, COLORS["WHITE"], start, end, 2)

            # Add "SAFE" label
            font = pygame.font.Font(None, 16)
            text = font.render("SAFE", True, COLORS["WHITE"])
            self.image.blit(text, (22, 15))

        self.rect = self.image.get_rect()

    def update(self):
        """Update powerup position and animation"""
        self.rect.x -= self.speed
        self.animation_timer += 1

        # Floating animation
        float_amplitude = 5
        self.rect.y = (
            self.base_y
            + math.sin(self.animation_timer * 0.1 + self.float_offset) * float_amplitude
        )

        # Rotation effect for some powerups
        if self.type == "OPTIMIZER" and self.animation_timer % 5 == 0:
            # Slight rotation effect (recreate sprite rotated)
            angle = (self.animation_timer * 2) % 360
            self.create_rotating_sprite(angle)

        # Pulsing glow effect
        if self.animation_timer % 60 < 30:  # Pulse every 60 frames
            # Add glow effect
            glow_surface = pygame.Surface(
                (self.rect.width + 10, self.rect.height + 10), pygame.SRCALPHA
            )
            glow_color = (*self.color, 50)  # Semi-transparent
            pygame.draw.circle(
                glow_surface,
                glow_color,
                (glow_surface.get_width() // 2, glow_surface.get_height() // 2),
                max(self.rect.width, self.rect.height) // 2 + 5,
            )

        # Remove if off screen
        if self.rect.right < 0:
            self.kill()

    def create_rotating_sprite(self, angle):
        """Create a rotated version of the sprite (for optimizer gear)"""
        if self.type == "OPTIMIZER":
            self.image = pygame.Surface((45, 45), pygame.SRCALPHA)
            center = (22, 22)

            # Rotate gear teeth
            for tooth_angle in range(0, 360, 45):
                rad = math.radians(tooth_angle + angle)
                x1 = center[0] + math.cos(rad) * 20
                y1 = center[1] + math.sin(rad) * 20
                x2 = center[0] + math.cos(rad) * 15
                y2 = center[1] + math.sin(rad) * 15
                pygame.draw.line(self.image, self.color, (x1, y1), (x2, y2), 4)

            # Inner circle (doesn't rotate)
            pygame.draw.circle(self.image, self.color, center, 12)
            pygame.draw.circle(self.image, COLORS["WHITE"], center, 12, 2)
            pygame.draw.circle(self.image, COLORS["WHITE"], center, 5)

    def get_powerup_type(self):
        """Return the type of powerup for player application"""
        powerup_map = {
            "DATASET": "dataset",
            "OPTIMIZER": "optimizer",
            "DROPOUT": "dropout",
        }
        return powerup_map.get(self.type, "generic")


class PowerupManager:
    def __init__(self):
        self.powerups = pygame.sprite.Group()
        self.spawn_timer = 0
        self.current_speed_multiplier = 1.0

    def update(self, speed_multiplier):
        """Update all powerups and spawn new ones"""
        self.current_speed_multiplier = speed_multiplier
        self.powerups.update()

        # Spawn new powerups more frequently for excitement
        if random.random() < POWERUP_SPAWN_RATE:
            self.spawn_powerup()

    def spawn_powerup(self):
        """Spawn a new random powerup"""
        powerup_type = random.choice(list(POWERUP_TYPES.keys()))
        spawn_x = SCREEN_WIDTH + random.randint(50, 200)

        powerup = Powerup(powerup_type, spawn_x, self.current_speed_multiplier)
        self.powerups.add(powerup)

    def check_collections(self, player):
        """Check for powerup collections by player"""
        collected_powerup_info = None

        for powerup in self.powerups:
            if powerup.rect.colliderect(player.rect):
                # Store powerup info for particle effects
                collected_powerup_info = {
                    "type": powerup.get_powerup_type(),
                    "x": powerup.rect.centerx,
                    "y": powerup.rect.centery,
                    "color": powerup.color,
                }

                # Apply powerup to player
                player.apply_powerup(powerup.get_powerup_type())
                powerup.kill()  # Remove powerup after collection
                break  # Only one powerup per frame

        return collected_powerup_info

    def draw(self, screen):
        """Draw all powerups"""
        self.powerups.draw(screen)

        # Draw powerup labels
        for powerup in self.powerups:
            if powerup.rect.x < SCREEN_WIDTH:  # Only for visible powerups
                label = powerup.type.replace("_", " ")
                font = pygame.font.Font(None, 16)
                label_surface = font.render(label, True, COLORS["WHITE"])
                label_rect = label_surface.get_rect()
                label_rect.centerx = powerup.rect.centerx
                label_rect.bottom = powerup.rect.top - 5
                screen.blit(label_surface, label_rect)

        # Draw collection sparkles for recently collected powerups
        # (This could be enhanced with particle effects)

    def clear(self):
        """Clear all powerups"""
        self.powerups.empty()

    def get_powerup_count(self):
        """Get current number of powerups"""
        return len(self.powerups)
