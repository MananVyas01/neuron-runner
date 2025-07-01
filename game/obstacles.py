"""
🚧 ML-themed Obstacles
Overfitting, vanishing gradients, noisy data, and dead neurons.
"""

import pygame
import random
import math
from .config import *


class Obstacle(pygame.sprite.Sprite):
    def __init__(self, obstacle_type, x_pos, speed_multiplier=1.0):
        super().__init__()
        self.type = obstacle_type
        self.effect = OBSTACLE_TYPES[obstacle_type]["effect"]
        self.color = OBSTACLE_TYPES[obstacle_type]["color"]

        # Create visual representation based on type
        self.create_sprite()
        self.rect.x = x_pos
        self.rect.y = GROUND_Y - self.rect.height

        self.speed = OBSTACLE_BASE_SPEED * speed_multiplier
        self.animation_timer = 0

    def create_sprite(self):
        """Create the visual representation of the obstacle"""
        if self.type == "OVERFITTING":
            # Draw as a complex, jagged crystal-like structure
            self.image = pygame.Surface((80, 100), pygame.SRCALPHA)
            # Main crystal body
            main_points = [
                (40, 10),
                (65, 30),
                (70, 60),
                (50, 85),
                (40, 95),
                (30, 85),
                (10, 60),
                (15, 30),
            ]
            pygame.draw.polygon(self.image, self.color, main_points)
            pygame.draw.polygon(self.image, COLORS["WHITE"], main_points, 3)

            # Add internal complexity lines
            for i in range(0, len(main_points), 2):
                center = (40, 50)
                pygame.draw.line(self.image, COLORS["WHITE"], center, main_points[i], 2)

            # Add warning symbols
            font = pygame.font.Font(None, 24)
            text = font.render("!", True, COLORS["WHITE"])
            self.image.blit(text, (35, 40))

        elif self.type == "VANISHING_GRADIENT":
            # Draw as fading gradient towers with better visibility
            self.image = pygame.Surface((60, 80), pygame.SRCALPHA)
            for i in range(6):
                alpha = 255 - (i * 35)
                width = 45 - (i * 5)
                height = 12
                x = (60 - width) // 2
                y = i * 13

                # Create gradient effect
                for j in range(height):
                    fade_alpha = alpha * (1 - j / height)
                    color = (*self.color, int(fade_alpha))
                    bar_surface = pygame.Surface((width, 1))
                    bar_surface.set_alpha(int(fade_alpha))
                    bar_surface.fill(self.color)
                    self.image.blit(bar_surface, (x, y + j))

                # Add border
                pygame.draw.rect(self.image, COLORS["WHITE"], (x, y, width, height), 1)

        elif self.type == "NOISY_DATA":
            # Draw as organized chaotic pattern with better structure
            self.image = pygame.Surface((70, 70), pygame.SRCALPHA)

            # Create structured noise pattern
            for row in range(0, 70, 4):
                for col in range(0, 70, 4):
                    if random.random() < 0.7:  # 70% fill rate
                        # Create small noise blocks
                        noise_colors = [COLORS["GRAY"], COLORS["WHITE"], self.color]
                        block_color = random.choice(noise_colors)
                        pygame.draw.rect(self.image, block_color, (col, row, 3, 3))

            # Add border for better visibility
            pygame.draw.rect(self.image, COLORS["WHITE"], (0, 0, 70, 70), 2)

            # Add "NOISE" label
            font = pygame.font.Font(None, 20)
            text = font.render("NOISE", True, COLORS["WHITE"])
            text_rect = pygame.Rect(0, 0, 70, 70)
            text_pos = text.get_rect(center=text_rect.center)
            self.image.blit(text, text_pos)

        elif self.type == "DEAD_NEURON":
            # Draw as detailed dead neuron with better visibility
            self.image = pygame.Surface((60, 60), pygame.SRCALPHA)
            center = (30, 30)

            # Draw outer dead area
            pygame.draw.circle(self.image, (50, 50, 50), center, 28)
            pygame.draw.circle(self.image, self.color, center, 25)
            pygame.draw.circle(self.image, COLORS["WHITE"], center, 25, 3)

            # Draw X pattern with better visibility
            pygame.draw.line(self.image, COLORS["LOSS"], (10, 10), (50, 50), 6)
            pygame.draw.line(self.image, COLORS["LOSS"], (50, 10), (10, 50), 6)
            pygame.draw.line(self.image, COLORS["WHITE"], (10, 10), (50, 50), 2)
            pygame.draw.line(self.image, COLORS["WHITE"], (50, 10), (10, 50), 2)

            # Add warning border
            pygame.draw.circle(self.image, COLORS["LOSS"], center, 28, 2)

        self.rect = self.image.get_rect()

    def update(self):
        """Update obstacle position and animation"""
        self.rect.x -= self.speed
        self.animation_timer += 1

        # Add some animation effects
        if self.type == "NOISY_DATA":
            # Regenerate noise pattern occasionally
            if self.animation_timer % 10 == 0:
                self.create_sprite()

        elif self.type == "VANISHING_GRADIENT":
            # Pulse effect
            if self.animation_timer % 30 == 0:
                self.create_sprite()

        # Remove if off screen
        if self.rect.right < 0:
            self.kill()

    def get_damage_type(self):
        """Return the type of damage this obstacle causes"""
        damage_map = {
            "OVERFITTING": "overfitting",
            "VANISHING_GRADIENT": "vanishing_gradient",
            "NOISY_DATA": "noisy_data",
            "DEAD_NEURON": "dead_neuron",
        }
        return damage_map.get(self.type, "generic")


class ObstacleManager:
    def __init__(self):
        self.obstacles = pygame.sprite.Group()
        self.spawn_timer = 0
        self.difficulty_multiplier = 1.0
        self.last_spawn_x = 0
        self.min_gap = 150  # Reduced for more challenge
        self.max_gap = 300  # Reduced for more action
        self.spawn_cooldown = 0
        self.current_speed_multiplier = 1.0
        self.difficulty_spike_timer = 0

    def update(self, difficulty_multiplier, game_score):
        """Update all obstacles and spawn new ones"""
        self.difficulty_multiplier = difficulty_multiplier

        # Calculate current speed multiplier based on score
        self.current_speed_multiplier = 1.0 + (game_score * SPEED_INCREASE_RATE)
        self.current_speed_multiplier = min(
            OBSTACLE_MAX_SPEED / OBSTACLE_BASE_SPEED, self.current_speed_multiplier
        )

        # Check for difficulty spikes
        if game_score > 0 and game_score % DIFFICULTY_SPIKE_INTERVAL == 0:
            if self.difficulty_spike_timer <= 0:
                self.trigger_difficulty_spike()
                self.difficulty_spike_timer = 300  # 5 second cooldown

        if self.difficulty_spike_timer > 0:
            self.difficulty_spike_timer -= 1

        self.obstacles.update()

        # Update spawn cooldown
        if self.spawn_cooldown > 0:
            self.spawn_cooldown -= 1

        # More aggressive spawning for excitement
        base_spawn_rate = OBSTACLE_SPAWN_RATE * self.difficulty_multiplier

        # Increase spawn rate during difficulty spikes
        if self.difficulty_spike_timer > 240:  # First second of spike
            base_spawn_rate *= 2.0

        # Check if we can spawn
        can_spawn = self.spawn_cooldown <= 0 and (
            not self.obstacles or self.get_last_obstacle_distance() > self.min_gap
        )

        if can_spawn and random.random() < base_spawn_rate:
            self.spawn_obstacle()

    def trigger_difficulty_spike(self):
        """Trigger a temporary difficulty spike for excitement"""
        print(f"🔥 DIFFICULTY SPIKE! Training intensity increased!")
        # Spawn a challenging pattern immediately
        self.spawn_spike_pattern()

    def spawn_spike_pattern(self):
        """Spawn a challenging but fair pattern during spikes"""
        pattern_type = random.choice(["wave", "gap_test", "speed_burst"])

        if pattern_type == "wave":
            # Three obstacles with alternating heights
            for i in range(3):
                obstacle_type = random.choice(["OVERFITTING", "NOISY_DATA"])
                x_pos = SCREEN_WIDTH + 100 + (i * 180)
                obstacle = Obstacle(
                    obstacle_type, x_pos, self.current_speed_multiplier * 1.2
                )
                self.obstacles.add(obstacle)

        elif pattern_type == "gap_test":
            # Two obstacles with a tight but doable gap
            obstacle_types = ["VANISHING_GRADIENT", "DEAD_NEURON"]
            for i, obs_type in enumerate(obstacle_types):
                x_pos = SCREEN_WIDTH + 100 + (i * 160)
                obstacle = Obstacle(
                    obs_type, x_pos, self.current_speed_multiplier * 1.1
                )
                self.obstacles.add(obstacle)

        elif pattern_type == "speed_burst":
            # Single fast obstacle
            obstacle_type = random.choice(list(OBSTACLE_TYPES.keys()))
            obstacle = Obstacle(
                obstacle_type, SCREEN_WIDTH + 50, self.current_speed_multiplier * 1.5
            )
            self.obstacles.add(obstacle)

        self.spawn_cooldown = 120  # 2 second cooldown after spike

    def get_last_obstacle_distance(self):
        """Get distance from the rightmost obstacle"""
        if not self.obstacles:
            return float("inf")

        rightmost_x = max(obstacle.rect.right for obstacle in self.obstacles)
        return SCREEN_WIDTH - rightmost_x

    def spawn_obstacle(self):
        """Spawn a new obstacle with intelligent placement"""
        # Weight obstacle types based on difficulty
        if self.difficulty_multiplier < 1.3:
            # Early game - easier obstacles
            obstacle_types = ["OVERFITTING", "NOISY_DATA"]
            group_chance = 0.15  # 15% chance for groups
        elif self.difficulty_multiplier < 2.0:
            # Mid game - add vanishing gradient
            obstacle_types = ["OVERFITTING", "NOISY_DATA", "VANISHING_GRADIENT"]
            group_chance = 0.25  # 25% chance for groups
        else:
            # Late game - all obstacles
            obstacle_types = list(OBSTACLE_TYPES.keys())
            group_chance = 0.35  # 35% chance for groups

        # Calculate spawn position with proper spacing
        base_spawn_x = SCREEN_WIDTH + 50
        if self.obstacles:
            rightmost_x = max(obstacle.rect.right for obstacle in self.obstacles)
            base_spawn_x = max(base_spawn_x, rightmost_x + self.min_gap)

        # Add some randomness to spacing (less predictable)
        spawn_x = base_spawn_x + random.randint(0, self.max_gap - self.min_gap)

        # Sometimes spawn obstacles in manageable groups
        if random.random() < group_chance:
            # Spawn 2-3 obstacles with challenging but fair gaps
            group_size = random.randint(2, 3)
            obstacle_gap = random.randint(140, 200)  # Tighter gaps for more challenge

            for i in range(group_size):
                if i == 0:
                    # First obstacle - any type
                    obstacle_type = random.choice(obstacle_types)
                else:
                    # Subsequent obstacles - mixed difficulty
                    obstacle_type = random.choice(obstacle_types)

                obstacle_x = spawn_x + (i * obstacle_gap)
                obstacle = Obstacle(
                    obstacle_type, obstacle_x, self.current_speed_multiplier
                )
                self.obstacles.add(obstacle)
        else:
            # Single obstacle
            obstacle_type = random.choice(obstacle_types)
            obstacle = Obstacle(obstacle_type, spawn_x, self.current_speed_multiplier)
            self.obstacles.add(obstacle)

        # Set cooldown to prevent immediate next spawn (shorter for more action)
        self.spawn_cooldown = random.randint(20, 40)  # 0.33-0.66 second cooldown

    def check_collisions(self, player):
        """Check for collisions with player and close calls"""
        collision_detected = False

        for obstacle in self.obstacles:
            # Check for actual collision
            if obstacle.rect.colliderect(player.rect):
                damage_applied = player.take_damage(obstacle.get_damage_type())
                if damage_applied:
                    obstacle.kill()  # Remove obstacle after hit
                collision_detected = True

            # Check for close calls (exciting near misses)
            elif self.is_close_call(obstacle, player):
                player.register_close_call()

        return collision_detected

    def is_close_call(self, obstacle, player):
        """Check if player had a close call with obstacle"""
        # Close call if obstacle just passed player
        if (
            obstacle.rect.right >= player.rect.left - 10
            and obstacle.rect.right <= player.rect.left + 5
        ):
            # Check if they were close vertically
            vertical_distance = abs(obstacle.rect.centery - player.rect.centery)
            if vertical_distance < 80:  # Close vertical distance
                return True
        return False

    def draw(self, screen):
        """Draw all obstacles"""
        self.obstacles.draw(screen)

        # Draw obstacle labels for debugging/clarity
        for obstacle in self.obstacles:
            if obstacle.rect.x < SCREEN_WIDTH:  # Only for visible obstacles
                label = obstacle.type.replace("_", " ")
                font = pygame.font.Font(None, 16)
                label_surface = font.render(label, True, COLORS["WHITE"])
                label_rect = label_surface.get_rect()
                label_rect.centerx = obstacle.rect.centerx
                label_rect.bottom = obstacle.rect.top - 5
                screen.blit(label_surface, label_rect)

    def clear(self):
        """Clear all obstacles"""
        self.obstacles.empty()

    def get_obstacle_count(self):
        """Get current number of obstacles"""
        return len(self.obstacles)
