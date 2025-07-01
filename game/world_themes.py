"""
🌍 Dynamic World Themes System
Changes visual environment every 1000 points to represent different AI domains
"""

import pygame
import math
import random
from .config import *


class WorldTheme:
    """Represents a themed environment with visual and audio characteristics"""

    def __init__(
        self, name, description, colors, effects, particle_style, music_intensity=1.0
    ):
        self.name = name
        self.description = description
        self.colors = colors
        self.effects = effects
        self.particle_style = particle_style
        self.music_intensity = music_intensity
        self.active_time = 0

    def apply_background_effects(self, screen, scroll_offset, pulse_timer):
        """Apply theme-specific background effects"""
        if self.effects.get("grid_overlay"):
            self.draw_grid_overlay(screen, scroll_offset, pulse_timer)

        if self.effects.get("floating_particles"):
            self.draw_floating_particles(screen, pulse_timer)

        if self.effects.get("data_streams"):
            self.draw_data_streams(screen, scroll_offset, pulse_timer)

        if self.effects.get("neural_connections"):
            self.draw_neural_connections(screen, pulse_timer)

        if self.effects.get("matrix_rain"):
            self.draw_matrix_rain(screen, scroll_offset, pulse_timer)

    def draw_grid_overlay(self, screen, scroll_offset, pulse_timer):
        """Draw a grid overlay effect"""
        grid_size = 50
        alpha = int(30 + 20 * math.sin(pulse_timer * 0.5))
        grid_color = (*self.colors["accent"], alpha)

        # Vertical lines
        for x in range(-grid_size, SCREEN_WIDTH + grid_size, grid_size):
            adjusted_x = (x - scroll_offset) % (SCREEN_WIDTH + grid_size)
            pygame.draw.line(
                screen, grid_color, (adjusted_x, 0), (adjusted_x, SCREEN_HEIGHT)
            )

        # Horizontal lines
        for y in range(0, SCREEN_HEIGHT + grid_size, grid_size):
            pygame.draw.line(screen, grid_color, (0, y), (SCREEN_WIDTH, y))

    def draw_floating_particles(self, screen, pulse_timer):
        """Draw floating particles in the background"""
        num_particles = 20
        for i in range(num_particles):
            x = (i * 73 + pulse_timer * 50) % SCREEN_WIDTH
            y = (i * 47 + math.sin(pulse_timer + i) * 30) % SCREEN_HEIGHT
            size = 2 + int(math.sin(pulse_timer + i) * 2)
            alpha = int(100 + 50 * math.sin(pulse_timer + i * 0.7))

            particle_color = (*self.colors["particle"], alpha)
            particle_surface = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
            pygame.draw.circle(particle_surface, particle_color, (size, size), size)
            screen.blit(particle_surface, (x - size, y - size))

    def draw_data_streams(self, screen, scroll_offset, pulse_timer):
        """Draw flowing data streams"""
        stream_count = 8
        for i in range(stream_count):
            stream_y = (i * SCREEN_HEIGHT // stream_count) + 50
            stream_speed = 100 + i * 20

            for j in range(5):
                x = (
                    (scroll_offset * stream_speed + j * 150) % (SCREEN_WIDTH + 200)
                ) - 100
                if 0 <= x <= SCREEN_WIDTH:
                    alpha = int(150 - j * 25)
                    stream_color = (*self.colors["primary"], alpha)
                    pygame.draw.circle(screen, stream_color, (int(x), stream_y), 3 - j)

    def draw_neural_connections(self, screen, pulse_timer):
        """Draw neural network-style connections"""
        nodes = []
        node_count = 12

        # Generate node positions
        for i in range(node_count):
            x = (i % 4) * (SCREEN_WIDTH // 4) + SCREEN_WIDTH // 8
            y = (i // 4) * (SCREEN_HEIGHT // 3) + SCREEN_HEIGHT // 6
            # Add slight movement
            x += math.sin(pulse_timer + i) * 20
            y += math.cos(pulse_timer + i * 0.7) * 15
            nodes.append((x, y))

        # Draw connections
        connection_alpha = int(80 + 40 * math.sin(pulse_timer))
        connection_color = (*self.colors["accent"], connection_alpha)

        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes[i + 1 :], i + 1):
                distance = math.sqrt(
                    (node1[0] - node2[0]) ** 2 + (node1[1] - node2[1]) ** 2
                )
                if distance < 200:  # Only connect nearby nodes
                    pygame.draw.line(screen, connection_color, node1, node2, 1)

        # Draw nodes
        for i, (x, y) in enumerate(nodes):
            node_pulse = math.sin(pulse_timer + i * 0.3)
            node_size = 4 + int(node_pulse * 2)
            node_alpha = int(200 + 55 * node_pulse)
            node_color = (*self.colors["primary"], node_alpha)

            node_surface = pygame.Surface(
                (node_size * 2, node_size * 2), pygame.SRCALPHA
            )
            pygame.draw.circle(
                node_surface, node_color, (node_size, node_size), node_size
            )
            screen.blit(node_surface, (x - node_size, y - node_size))

    def draw_matrix_rain(self, screen, scroll_offset, pulse_timer):
        """Draw Matrix-style digital rain"""
        column_count = 40
        column_width = SCREEN_WIDTH // column_count

        for col in range(column_count):
            x = col * column_width
            # Different speeds for each column
            speed = 100 + (col % 5) * 50

            for i in range(8):
                y = ((pulse_timer * speed + i * 80) % (SCREEN_HEIGHT + 160)) - 80
                if 0 <= y <= SCREEN_HEIGHT:
                    alpha = int(255 - (i * 30))
                    char_color = (*self.colors["accent"], alpha)

                    # Draw a simple rectangle as "character"
                    char_rect = pygame.Rect(x, y, column_width - 2, 12)
                    char_surface = pygame.Surface(
                        (column_width - 2, 12), pygame.SRCALPHA
                    )
                    char_surface.fill(char_color)
                    screen.blit(char_surface, char_rect)


class WorldThemeManager:
    """Manages dynamic world theme transitions"""

    def __init__(self):
        self.current_theme_index = 0
        self.transition_progress = 0
        self.transitioning = False
        self.transition_duration = 2.0  # seconds
        self.last_transition_time = 0

        # Theme definitions
        self.themes = [
            WorldTheme(
                name="Neural Processing",
                description="🧠 Core neural network environment",
                colors={
                    "primary": COLORS["NEURON_BLUE"],
                    "accent": COLORS["ACTIVATION"],
                    "particle": COLORS["DATASET"],
                    "background": COLORS["BACKGROUND"],
                },
                effects={"neural_connections": True, "floating_particles": True},
                particle_style="neural",
                music_intensity=1.0,
            ),
            WorldTheme(
                name="Computer Vision",
                description="👁️ Image processing and pattern recognition",
                colors={
                    "primary": (64, 156, 255),  # CNN Blue
                    "accent": (0, 255, 127),  # Green
                    "particle": (100, 200, 255),  # Light blue
                    "background": (10, 15, 30),
                },
                effects={"grid_overlay": True, "data_streams": True},
                particle_style="geometric",
                music_intensity=1.2,
            ),
            WorldTheme(
                name="Natural Language",
                description="📝 Text processing and language understanding",
                colors={
                    "primary": (156, 64, 255),  # Purple
                    "accent": (255, 64, 156),  # Pink
                    "particle": (200, 100, 255),  # Light purple
                    "background": (15, 10, 25),
                },
                effects={"matrix_rain": True, "floating_particles": True},
                particle_style="flowing",
                music_intensity=0.9,
            ),
            WorldTheme(
                name="Transformer Attention",
                description="⚡ Self-attention and transformer networks",
                colors={
                    "primary": (255, 193, 7),  # Gold
                    "accent": (255, 87, 34),  # Orange
                    "particle": (255, 235, 59),  # Yellow
                    "background": (20, 15, 5),
                },
                effects={
                    "neural_connections": True,
                    "grid_overlay": True,
                    "floating_particles": True,
                },
                particle_style="attention",
                music_intensity=1.3,
            ),
            WorldTheme(
                name="Generative AI",
                description="🎨 Creative AI and content generation",
                colors={
                    "primary": (76, 175, 80),  # Green
                    "accent": (244, 67, 54),  # Red
                    "particle": (139, 195, 74),  # Light green
                    "background": (5, 20, 10),
                },
                effects={"data_streams": True, "matrix_rain": True},
                particle_style="creative",
                music_intensity=1.1,
            ),
            WorldTheme(
                name="Quantum ML",
                description="⚛️ Quantum computing meets machine learning",
                colors={
                    "primary": (138, 43, 226),  # Blue violet
                    "accent": (0, 191, 255),  # Deep sky blue
                    "particle": (147, 112, 219),  # Medium purple
                    "background": (10, 5, 20),
                },
                effects={
                    "neural_connections": True,
                    "floating_particles": True,
                    "data_streams": True,
                },
                particle_style="quantum",
                music_intensity=1.4,
            ),
        ]

        self.theme_transition_scores = [0, 1000, 2500, 4000, 6000, 8500]

        # Animation timers
        self.pulse_timer = 0
        self.theme_notification_timer = 0

    def initialize(self):
        """Initialize the world theme manager"""
        # Set up initial theme
        self.current_theme_index = 0
        self.transition_progress = 0
        self.transitioning = False
        print(f"🌍 World themes initialized with {len(self.themes)} themes")

    def update(self, score, dt):
        """Update theme manager"""
        self.pulse_timer += dt * 2

        # Check for theme transitions
        new_theme_index = self.get_theme_index_for_score(score)
        if new_theme_index != self.current_theme_index and not self.transitioning:
            self.start_transition(new_theme_index)

        # Update transition
        if self.transitioning:
            self.transition_progress += dt / self.transition_duration
            if self.transition_progress >= 1.0:
                self.complete_transition()

        # Update theme notification
        if self.theme_notification_timer > 0:
            self.theme_notification_timer -= dt

        # Update current theme
        self.themes[self.current_theme_index].active_time += dt

    def get_theme_index_for_score(self, score):
        """Get the appropriate theme index for the current score"""
        for i in range(len(self.theme_transition_scores) - 1, -1, -1):
            if score >= self.theme_transition_scores[i]:
                return min(i, len(self.themes) - 1)
        return 0

    def start_transition(self, new_theme_index):
        """Start transitioning to a new theme"""
        if new_theme_index < len(self.themes):
            self.transitioning = True
            self.transition_progress = 0
            self.next_theme_index = new_theme_index
            self.theme_notification_timer = 3.0  # Show notification for 3 seconds

            print(f"🌍 Transitioning to theme: {self.themes[new_theme_index].name}")

    def complete_transition(self):
        """Complete the theme transition"""
        self.current_theme_index = self.next_theme_index
        self.transitioning = False
        self.transition_progress = 0

        print(
            f"✅ Theme transition complete: {self.themes[self.current_theme_index].name}"
        )

    def get_current_theme(self):
        """Get the currently active theme"""
        return self.themes[self.current_theme_index]

    def get_background_color(self):
        """Get the current background color with transition blending"""
        current_theme = self.themes[self.current_theme_index]

        if not self.transitioning:
            return current_theme.colors["background"]

        # Blend colors during transition
        next_theme = self.themes[self.next_theme_index]
        current_bg = current_theme.colors["background"]
        next_bg = next_theme.colors["background"]

        t = self.transition_progress
        blended_color = tuple(
            int(current_bg[i] * (1 - t) + next_bg[i] * t) for i in range(3)
        )

        return blended_color

    def draw_background_effects(self, screen, scroll_offset):
        """Draw theme-specific background effects"""
        current_theme = self.themes[self.current_theme_index]

        # Apply current theme effects
        alpha_multiplier = (
            1.0 if not self.transitioning else (1.0 - self.transition_progress)
        )
        if alpha_multiplier > 0:
            current_theme.apply_background_effects(
                screen, scroll_offset, self.pulse_timer
            )

        # Apply next theme effects during transition
        if self.transitioning and self.transition_progress > 0.3:
            next_theme = self.themes[self.next_theme_index]
            alpha_multiplier = self.transition_progress - 0.3
            if alpha_multiplier > 0:
                next_theme.apply_background_effects(
                    screen, scroll_offset, self.pulse_timer
                )

    def draw_theme_notification(self, screen):
        """Draw theme change notification"""
        if self.theme_notification_timer <= 0:
            return

        # Fade in/out animation
        fade_duration = 0.5
        if self.theme_notification_timer > 3.0 - fade_duration:
            alpha = (3.0 - self.theme_notification_timer) / fade_duration
        elif self.theme_notification_timer < fade_duration:
            alpha = self.theme_notification_timer / fade_duration
        else:
            alpha = 1.0

        alpha = int(alpha * 255)

        # Notification background
        notification_width = 400
        notification_height = 80
        notification_x = (SCREEN_WIDTH - notification_width) // 2
        notification_y = 100

        notification_bg = pygame.Surface(
            (notification_width, notification_height), pygame.SRCALPHA
        )
        notification_bg.fill((*COLORS["DARK_BLUE"], int(alpha * 0.9)))
        screen.blit(notification_bg, (notification_x, notification_y))

        # Border with theme color
        current_theme = self.themes[self.current_theme_index]
        border_color = (*current_theme.colors["primary"], alpha)
        pygame.draw.rect(
            screen,
            border_color,
            (notification_x, notification_y, notification_width, notification_height),
            3,
        )

        # Notification text
        font_large = pygame.font.Font(None, 32)
        font_small = pygame.font.Font(None, 24)

        title_text = font_large.render(
            "🌍 ENTERING NEW DOMAIN", True, (*COLORS["WHITE"], alpha)
        )
        title_rect = title_text.get_rect(
            center=(notification_x + notification_width // 2, notification_y + 25)
        )
        screen.blit(title_text, title_rect)

        theme_text = font_small.render(
            current_theme.name.upper(), True, (*current_theme.colors["primary"], alpha)
        )
        theme_rect = theme_text.get_rect(
            center=(notification_x + notification_width // 2, notification_y + 50)
        )
        screen.blit(theme_text, theme_rect)

        desc_text = font_small.render(
            current_theme.description, True, (*COLORS["LIGHT_GRAY"], alpha)
        )
        desc_rect = desc_text.get_rect(
            center=(notification_x + notification_width // 2, notification_y + 70)
        )
        screen.blit(desc_text, desc_rect)


# Global world theme manager instance
world_themes = WorldThemeManager()
