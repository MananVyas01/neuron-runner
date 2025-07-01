"""
✨ Particle Effects System
Advanced visual effects for enhanced gameplay experience.
"""

import pygame
import math
import random
from .config import *


class Particle:
    def __init__(self, x, y, velocity_x, velocity_y, color, size, lifetime):
        self.x = x
        self.y = y
        self.velocity_x = velocity_x
        self.velocity_y = velocity_y
        self.color = color
        self.size = size
        self.max_size = size
        self.lifetime = lifetime
        self.max_lifetime = lifetime
        self.alpha = 255

    def update(self):
        """Update particle position and properties"""
        self.x += self.velocity_x
        self.y += self.velocity_y

        # Apply gravity to particles
        self.velocity_y += 0.2

        # Reduce lifetime
        self.lifetime -= 1

        # Fade out over time
        life_ratio = self.lifetime / self.max_lifetime
        self.alpha = int(255 * life_ratio)
        self.size = int(self.max_size * life_ratio)

        return self.lifetime > 0

    def draw(self, surface):
        """Draw the particle"""
        if self.size > 0:
            color_with_alpha = (*self.color, self.alpha)
            particle_surface = pygame.Surface(
                (self.size * 2, self.size * 2), pygame.SRCALPHA
            )
            pygame.draw.circle(
                particle_surface, color_with_alpha, (self.size, self.size), self.size
            )
            surface.blit(particle_surface, (self.x - self.size, self.y - self.size))


class ParticleSystem:
    def __init__(self):
        self.particles = []

    def create_explosion(self, x, y, color, count=20):
        """Create an explosion effect"""
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(2, 8)
            velocity_x = math.cos(angle) * speed
            velocity_y = math.sin(angle) * speed - random.uniform(1, 3)  # Upward bias

            size = random.randint(2, 6)
            lifetime = random.randint(30, 60)

            particle = Particle(x, y, velocity_x, velocity_y, color, size, lifetime)
            self.particles.append(particle)

    def create_powerup_collection(self, x, y, color):
        """Create particle effect for powerup collection"""
        for _ in range(15):
            angle = random.uniform(-math.pi / 4, -3 * math.pi / 4)  # Upward arc
            speed = random.uniform(3, 6)
            velocity_x = math.cos(angle) * speed
            velocity_y = math.sin(angle) * speed

            size = random.randint(3, 5)
            lifetime = random.randint(40, 70)

            particle = Particle(x, y, velocity_x, velocity_y, color, size, lifetime)
            self.particles.append(particle)

    def create_trail(self, x, y, color, count=5):
        """Create a trail effect"""
        for _ in range(count):
            velocity_x = random.uniform(-2, 0)
            velocity_y = random.uniform(-1, 1)

            size = random.randint(1, 3)
            lifetime = random.randint(15, 30)

            particle = Particle(x, y, velocity_x, velocity_y, color, size, lifetime)
            self.particles.append(particle)

    def create_combo_burst(self, x, y):
        """Create special effect for combo achievements"""
        colors = [COLORS["DATASET"], COLORS["ACTIVATION"], COLORS["OPTIMIZER"]]

        for _ in range(30):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(4, 10)
            velocity_x = math.cos(angle) * speed
            velocity_y = math.sin(angle) * speed - 2  # Slight upward bias

            color = random.choice(colors)
            size = random.randint(3, 7)
            lifetime = random.randint(50, 80)

            particle = Particle(x, y, velocity_x, velocity_y, color, size, lifetime)
            self.particles.append(particle)

    def create_adrenaline_particles(self, x, y):
        """Create continuous particles for adrenaline mode"""
        for _ in range(3):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 3)
            velocity_x = math.cos(angle) * speed
            velocity_y = math.sin(angle) * speed - 1

            size = random.randint(2, 4)
            lifetime = random.randint(20, 40)

            particle = Particle(
                x, y, velocity_x, velocity_y, COLORS["LOSS"], size, lifetime
            )
            self.particles.append(particle)

    def update(self):
        """Update all particles"""
        self.particles = [particle for particle in self.particles if particle.update()]

    def draw(self, surface):
        """Draw all particles"""
        for particle in self.particles:
            particle.draw(surface)

    def clear(self):
        """Clear all particles"""
        self.particles.clear()


# Global particle system instance
particle_system = ParticleSystem()
