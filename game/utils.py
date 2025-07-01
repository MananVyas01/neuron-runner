"""
🛠️ Game Utilities
Score tracking, loss graph simulation, and helper functions.
"""

import pygame
import math
import random
from .config import *


class GameStats:
    def __init__(self):
        self.score = 0
        self.high_score = 0
        self.current_epoch = 0
        self.epoch_timer = 0
        self.loss_history = [1.0]  # Start with high loss
        self.current_loss = 1.0
        self.training_accuracy = 0.0
        self.current_speed_multiplier = 1.0

        # Additional tracking for training journal
        self.distance_traveled = 0
        self.obstacles_dodged = 0
        self.powerups_collected = 0

    def update(self, player_status):
        """Update game statistics"""
        # Increment score based on survival time
        self.score += 1

        # Track distance traveled
        self.distance_traveled += self.current_speed_multiplier

        # Update epoch counter
        self.epoch_timer += 1
        if self.epoch_timer >= EPOCH_DURATION * FPS:
            self.current_epoch += 1
            self.epoch_timer = 0
            player_status["epochs"] = self.current_epoch

        # Simulate loss reduction (with some noise)
        base_loss_reduction = 0.001
        if player_status["activation"] > 50:
            # Good health = better learning
            loss_reduction = base_loss_reduction * (player_status["activation"] / 100)
        else:
            # Poor health = worse learning, loss might increase
            loss_reduction = -base_loss_reduction * 0.5

        # Add some randomness
        noise = random.uniform(-0.0005, 0.0005)
        self.current_loss = max(0.01, self.current_loss - loss_reduction + noise)

        # Update loss history for graph
        if len(self.loss_history) > 100:  # Keep last 100 points
            self.loss_history.pop(0)
        self.loss_history.append(self.current_loss)

        # Calculate training accuracy (inverse of loss)
        self.training_accuracy = max(0, min(100, (1 - self.current_loss) * 100))

        # Update speed multiplier
        self.current_speed_multiplier = 1.0 + (self.score * SPEED_INCREASE_RATE)

        # Update high score
        if self.score > self.high_score:
            self.high_score = self.score

    def reset(self):
        """Reset statistics for new game"""
        self.score = 0
        self.current_epoch = 0
        self.epoch_timer = 0
        self.loss_history = [1.0]
        self.current_loss = 1.0
        self.training_accuracy = 0.0
        self.current_speed_multiplier = 1.0
        # Reset additional tracking
        self.distance_traveled = 0
        self.obstacles_dodged = 0
        self.powerups_collected = 0


class LossGraph:
    def __init__(self, x, y, width, height):
        self.rect = pygame.Rect(x, y, width, height)
        self.surface = pygame.Surface((width, height), pygame.SRCALPHA)

    def draw(self, screen, loss_history):
        """Draw the loss curve graph"""
        # Clear surface
        self.surface.fill((0, 0, 0, 100))  # Semi-transparent background

        if len(loss_history) < 2:
            return

        # Draw axes
        pygame.draw.line(
            self.surface,
            COLORS["WHITE"],
            (10, self.rect.height - 10),
            (self.rect.width - 10, self.rect.height - 10),
            2,
        )  # X-axis
        pygame.draw.line(
            self.surface, COLORS["WHITE"], (10, 10), (10, self.rect.height - 10), 2
        )  # Y-axis

        # Draw loss curve
        if len(loss_history) > 1:
            points = []
            for i, loss in enumerate(loss_history):
                x = 10 + (i / max(1, len(loss_history) - 1)) * (self.rect.width - 20)
                y = self.rect.height - 10 - (loss * (self.rect.height - 20))
                points.append((x, y))

            if len(points) > 1:
                pygame.draw.lines(self.surface, COLORS["GRADIENT"], False, points, 3)

        # Draw to main screen
        screen.blit(self.surface, self.rect)


class UI:
    def __init__(self):
        pygame.font.init()
        self.font = pygame.font.Font(None, UI_FONT_SIZE)
        self.small_font = pygame.font.Font(None, SMALL_FONT_SIZE)
        self.loss_graph = LossGraph(GRAPH_X, GRAPH_Y, GRAPH_WIDTH, GRAPH_HEIGHT)

    def draw_hud(self, screen, stats, player_status):
        """Draw the heads-up display with enhanced visibility"""
        # Draw semi-transparent background for HUD
        hud_background = pygame.Surface((350, 280))
        hud_background.set_alpha(120)
        hud_background.fill(COLORS["BLACK"])
        screen.blit(hud_background, (10, 10))

        # Main stats with better formatting
        score_text = self.font.render(
            f"Training Progress: {stats.score:,}", True, COLORS["WHITE"]
        )
        epoch_text = self.font.render(
            f"Epoch: {stats.current_epoch}", True, COLORS["ACTIVATION"]
        )
        loss_text = self.font.render(
            f"Loss: {stats.current_loss:.4f}", True, COLORS["GRADIENT"]
        )
        accuracy_text = self.font.render(
            f"Accuracy: {stats.training_accuracy:.1f}%", True, COLORS["ACTIVATION"]
        )

        screen.blit(score_text, (20, 20))
        screen.blit(epoch_text, (20, 50))
        screen.blit(loss_text, (20, 80))
        screen.blit(accuracy_text, (20, 110))

        # Enhanced health/activation display
        self.draw_health_bar(screen, player_status["activation"], 20, 150)

        # Player neural network status
        layers_text = self.small_font.render(
            f"Network Layers: {player_status['layers']}", True, COLORS["NEURON_BLUE"]
        )
        speed_text = self.small_font.render(
            f"Optimizer Speed: {player_status['optimizer_speed']:.1f}x",
            True,
            COLORS["OPTIMIZER"],
        )
        lr_text = self.small_font.render(
            f"Learning Rate: {player_status['learning_rate']:.3f}",
            True,
            COLORS["WHITE"],
        )

        screen.blit(layers_text, (20, 185))
        screen.blit(speed_text, (20, 205))
        screen.blit(lr_text, (20, 225))

        # Status indicators with better visibility
        y_offset = 250
        if player_status["boosted"]:
            boost_text = self.small_font.render(
                "🔥 LAYER BOOST - ENHANCED PERFORMANCE", True, COLORS["ACTIVATION"]
            )
            boost_bg = pygame.Surface((boost_text.get_width() + 10, 20))
            boost_bg.set_alpha(100)
            boost_bg.fill(COLORS["ACTIVATION"])
            screen.blit(boost_bg, (15, y_offset - 2))
            screen.blit(boost_text, (20, y_offset))
            y_offset += 25

        if player_status["immune"]:
            immune_text = self.small_font.render(
                "🛡️ IMMUNITY ACTIVE - DAMAGE BLOCKED", True, COLORS["DATASET"]
            )
            immune_bg = pygame.Surface((immune_text.get_width() + 10, 20))
            immune_bg.set_alpha(100)
            immune_bg.fill(COLORS["DATASET"])
            screen.blit(immune_bg, (15, y_offset - 2))
            screen.blit(immune_text, (20, y_offset))
            y_offset += 25

        if player_status["frozen"]:
            frozen_text = self.small_font.render(
                "❄️ VANISHING GRADIENT - CONTROLS FROZEN", True, COLORS["LOSS"]
            )
            frozen_bg = pygame.Surface((frozen_text.get_width() + 10, 20))
            frozen_bg.set_alpha(100)
            frozen_bg.fill(COLORS["LOSS"])
            screen.blit(frozen_bg, (15, y_offset - 2))
            screen.blit(frozen_text, (20, y_offset))

        # Draw loss graph
        self.loss_graph.draw(screen, stats.loss_history)

        # Graph title with better styling
        graph_title = self.small_font.render(
            "Training Loss Curve", True, COLORS["WHITE"]
        )
        graph_bg = pygame.Surface((graph_title.get_width() + 10, 20))
        graph_bg.set_alpha(100)
        graph_bg.fill(COLORS["BLACK"])
        screen.blit(graph_bg, (GRAPH_X - 5, GRAPH_Y - 30))
        screen.blit(graph_title, (GRAPH_X, GRAPH_Y - 25))

        # Performance indicators
        self.draw_performance_indicators(screen, stats, player_status)

        # Excitement indicators
        self.draw_excitement_indicators(screen, player_status)

        # Store speed multiplier for display
        if hasattr(stats, "current_speed_multiplier"):
            self._last_speed_mult = stats.current_speed_multiplier
        else:
            self._last_speed_mult = 1.0

        # Enhanced controls display
        controls_bg = pygame.Surface((SCREEN_WIDTH - 40, 50))
        controls_bg.set_alpha(100)
        controls_bg.fill(COLORS["BLACK"])
        screen.blit(controls_bg, (20, SCREEN_HEIGHT - 70))

        controls_text1 = self.small_font.render(
            "Controls: SPACE=Jump | DOWN=Duck | L=Layer Boost | P=Pause | R=Restart",
            True,
            COLORS["WHITE"],
        )
        screen.blit(controls_text1, (25, SCREEN_HEIGHT - 65))

        # Show current speed
        speed_mult = getattr(self, "_last_speed_mult", 1.0)
        speed_text = self.small_font.render(
            f"🚀 Current Speed: {speed_mult:.1f}x | Score for next spike: {1000 - (stats.score % 1000)}",
            True,
            COLORS["ACTIVATION"],
        )
        screen.blit(speed_text, (25, SCREEN_HEIGHT - 45))

    def draw_excitement_indicators(self, screen, player_status):
        """Draw combo, adrenaline, and close call indicators"""
        y_start = 300

        # Combo indicator
        if player_status["combo"] > 0:
            combo_text = self.font.render(
                f"🔥 COMBO x{player_status['combo']}", True, COLORS["DATASET"]
            )
            combo_bg = pygame.Surface((combo_text.get_width() + 20, 30))
            combo_bg.set_alpha(150)
            combo_bg.fill(COLORS["DATASET"])
            screen.blit(combo_bg, (SCREEN_WIDTH - combo_text.get_width() - 30, y_start))
            screen.blit(
                combo_text, (SCREEN_WIDTH - combo_text.get_width() - 20, y_start + 5)
            )
            y_start += 40

        # Adrenaline mode indicator
        if player_status["adrenaline"]:
            adrenaline_text = self.font.render(
                "⚡ ADRENALINE MODE!", True, COLORS["LOSS"]
            )
            adrenaline_bg = pygame.Surface((adrenaline_text.get_width() + 20, 30))
            adrenaline_bg.set_alpha(150)
            adrenaline_bg.fill(COLORS["LOSS"])
            screen.blit(
                adrenaline_bg,
                (SCREEN_WIDTH - adrenaline_text.get_width() - 30, y_start),
            )
            screen.blit(
                adrenaline_text,
                (SCREEN_WIDTH - adrenaline_text.get_width() - 20, y_start + 5),
            )
            y_start += 40

        # Close call indicator
        if player_status["close_call"]:
            close_call_text = self.font.render(
                "💫 CLOSE CALL!", True, COLORS["ACTIVATION"]
            )
            close_call_bg = pygame.Surface((close_call_text.get_width() + 20, 30))
            close_call_bg.set_alpha(150)
            close_call_bg.fill(COLORS["ACTIVATION"])
            screen.blit(
                close_call_bg,
                (SCREEN_WIDTH - close_call_text.get_width() - 30, y_start),
            )
            screen.blit(
                close_call_text,
                (SCREEN_WIDTH - close_call_text.get_width() - 20, y_start + 5),
            )

    def draw_health_bar(self, screen, activation_level, x, y):
        """Draw a detailed health/activation bar"""
        bar_width = 300
        bar_height = 20

        # Background bar
        pygame.draw.rect(screen, COLORS["GRAY"], (x, y, bar_width, bar_height))
        pygame.draw.rect(screen, COLORS["WHITE"], (x, y, bar_width, bar_height), 2)

        # Health bar fill
        fill_width = int((activation_level / 100) * (bar_width - 4))

        # Color based on health level
        if activation_level > 70:
            health_color = COLORS["ACTIVATION"]
        elif activation_level > 40:
            health_color = COLORS["DATASET"]
        elif activation_level > 20:
            health_color = COLORS["GRADIENT"]
        else:
            health_color = COLORS["LOSS"]

        if fill_width > 0:
            pygame.draw.rect(
                screen, health_color, (x + 2, y + 2, fill_width, bar_height - 4)
            )

        # Health text
        health_text = self.small_font.render(
            f"Neural Activation: {activation_level:.0f}%", True, COLORS["WHITE"]
        )
        screen.blit(health_text, (x, y - 20))

        # Add pulsing effect for low health
        if activation_level < 30:
            pulse = int(abs(math.sin(pygame.time.get_ticks() * 0.01)) * 255)
            warning_color = (255, pulse, pulse)
            pygame.draw.rect(screen, warning_color, (x, y, bar_width, bar_height), 3)

    def draw_performance_indicators(self, screen, stats, player_status):
        """Draw additional performance indicators"""
        # Performance panel
        panel_x = SCREEN_WIDTH - 250
        panel_y = GRAPH_Y + GRAPH_HEIGHT + 40
        panel_width = 230
        panel_height = 120

        # Background
        panel_bg = pygame.Surface((panel_width, panel_height))
        panel_bg.set_alpha(120)
        panel_bg.fill(COLORS["BLACK"])
        screen.blit(panel_bg, (panel_x, panel_y))

        # Title
        title_text = self.small_font.render(
            "Performance Metrics", True, COLORS["WHITE"]
        )
        screen.blit(title_text, (panel_x + 10, panel_y + 10))

        # Metrics
        metrics = [
            f"Time Survived: {stats.score // 60:.1f}s",
            f"Perfect Dodges: {player_status.get('perfect_dodges', 0)}",
            f"Network Efficiency: {min(100, player_status['optimizer_speed'] * 50):.0f}%",
            f"Training Stability: {max(0, 100 - stats.current_loss * 100):.0f}%",
        ]

        for i, metric in enumerate(metrics):
            metric_text = self.small_font.render(metric, True, COLORS["WHITE"])
            screen.blit(metric_text, (panel_x + 10, panel_y + 35 + i * 20))

    def draw_game_over(self, screen, stats):
        """Draw game over screen"""
        # Semi-transparent overlay
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        overlay.set_alpha(180)
        overlay.fill(COLORS["BLACK"])
        screen.blit(overlay, (0, 0))

        # Game over text
        game_over_text = pygame.font.Font(None, 72).render(
            "TRAINING FAILED", True, COLORS["LOSS"]
        )
        game_over_rect = game_over_text.get_rect(
            center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 100)
        )
        screen.blit(game_over_text, game_over_rect)

        # Final stats
        final_score_text = self.font.render(
            f"Final Training Progress: {stats.score}", True, COLORS["WHITE"]
        )
        final_score_rect = final_score_text.get_rect(
            center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 50)
        )
        screen.blit(final_score_text, final_score_rect)

        epochs_text = self.font.render(
            f"Epochs Completed: {stats.current_epoch}", True, COLORS["WHITE"]
        )
        epochs_rect = epochs_text.get_rect(
            center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 20)
        )
        screen.blit(epochs_text, epochs_rect)

        high_score_text = self.font.render(
            f"Best Training Progress: {stats.high_score}", True, COLORS["DATASET"]
        )
        high_score_rect = high_score_text.get_rect(
            center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 10)
        )
        screen.blit(high_score_text, high_score_rect)

        # Restart instruction
        restart_text = self.font.render(
            "Press R to Restart Training", True, COLORS["ACTIVATION"]
        )
        restart_rect = restart_text.get_rect(
            center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 60)
        )
        screen.blit(restart_text, restart_rect)


def draw_background(screen, scroll_offset=0, player_status=None):
    """Draw the neural network themed background with dynamic effects"""
    screen.fill(COLORS["BACKGROUND"])

    # Dynamic background effects based on player state
    if player_status and player_status.get("adrenaline", False):
        # Pulsing red background during adrenaline mode
        pulse = abs(math.sin(pygame.time.get_ticks() * 0.01))
        red_tint = int(30 * pulse)
        adrenaline_bg = (
            COLORS["BACKGROUND"][0] + red_tint,
            COLORS["BACKGROUND"][1],
            COLORS["BACKGROUND"][2],
        )
        screen.fill(adrenaline_bg)

    # Draw ground
    pygame.draw.rect(
        screen, COLORS["GROUND"], (0, GROUND_Y, SCREEN_WIDTH, SCREEN_HEIGHT - GROUND_Y)
    )

    # Draw neural network grid pattern
    grid_size = 50
    grid_alpha = (
        60 if not (player_status and player_status.get("adrenaline", False)) else 120
    )

    for x in range(-grid_size, SCREEN_WIDTH + grid_size, grid_size):
        adjusted_x = (x - scroll_offset) % SCREEN_WIDTH
        grid_color = (
            (30, 40, 60)
            if not (player_status and player_status.get("adrenaline", False))
            else (60, 30, 30)
        )
        pygame.draw.line(
            screen, grid_color, (adjusted_x, 0), (adjusted_x, SCREEN_HEIGHT), 1
        )

    for y in range(0, SCREEN_HEIGHT, grid_size):
        grid_color = (
            (30, 40, 60)
            if not (player_status and player_status.get("adrenaline", False))
            else (60, 30, 30)
        )
        pygame.draw.line(screen, grid_color, (0, y), (SCREEN_WIDTH, y), 1)

    # Draw some decorative "neural connections" with dynamic effects
    connection_color = (60, 80, 120)
    if player_status and player_status.get("combo", 0) > 0:
        # Brighter connections during combos
        combo_intensity = min(player_status["combo"] / 10, 1.0)
        connection_color = (
            int(60 + 100 * combo_intensity),
            int(80 + 120 * combo_intensity),
            int(120 + 135 * combo_intensity),
        )

    for i in range(8):
        start_x = (i * 150 - scroll_offset) % SCREEN_WIDTH
        start_y = 100 + (i * 50) % (GROUND_Y - 200)
        end_x = start_x + 100
        end_y = start_y + random.randint(-30, 30)

        if 0 <= start_x <= SCREEN_WIDTH and 0 <= end_x <= SCREEN_WIDTH:
            pygame.draw.line(
                screen, connection_color, (start_x, start_y), (end_x, end_y), 2
            )


def check_collision(sprite1, sprite2):
    """Check collision between two sprites with some tolerance"""
    # Use smaller collision rect for more forgiving gameplay
    rect1 = sprite1.rect.inflate(-5, -5)
    rect2 = sprite2.rect.inflate(-5, -5)
    return rect1.colliderect(rect2)


def spawn_probability(base_rate, difficulty_multiplier=1.0):
    """Calculate spawn probability based on difficulty"""
    return min(0.1, base_rate * difficulty_multiplier)


def calculate_difficulty(score):
    """Calculate difficulty multiplier based on score"""
    return 1.0 + (score / 1000) * 0.5  # Gradually increase difficulty
