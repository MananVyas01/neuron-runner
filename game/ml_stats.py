"""
🧪 ML Stats System - INDUSTRIAL LEVEL Live ML Metrics
Real-time comprehensive ML concept visualization with professional-grade metrics dashboard
"""

import pygame
import math
import random
import numpy as np
from collections import deque
from .config import *


class MetricBuffer:
    """Professional circular buffer for metric history"""

    def __init__(self, maxlen=200):
        self.buffer = deque(maxlen=maxlen)
        self.maxlen = maxlen

    def add(self, value):
        self.buffer.append(value)

    def get_recent(self, n=50):
        return list(self.buffer)[-n:]

    def mean(self, n=None):
        data = list(self.buffer) if n is None else self.get_recent(n)
        return np.mean(data) if data else 0.0

    def std(self, n=None):
        data = list(self.buffer) if n is None else self.get_recent(n)
        return np.std(data) if len(data) > 1 else 0.0

    def trend(self, n=20):
        """Calculate trend direction (-1 to 1)"""
        if len(self.buffer) < n:
            return 0.0
        recent = self.get_recent(n)
        x = np.arange(len(recent))
        z = np.polyfit(x, recent, 1)
        return np.tanh(z[0] * 10)  # Normalize slope


class MLMetric:
    """Professional ML metric with statistics and visualization"""

    def __init__(
        self, name, initial_value, min_val=0, max_val=100, unit="", format_str="{:.2f}"
    ):
        self.name = name
        self.value = initial_value
        self.min_val = min_val
        self.max_val = max_val
        self.unit = unit
        self.format_str = format_str

        # Professional tracking
        self.history = MetricBuffer(200)
        self.target_value = initial_value
        self.momentum = 0.0
        self.volatility = 0.0

        # Visual states
        self.status = "stable"  # stable, improving, degrading, critical
        self.alert_level = 0  # 0=normal, 1=warning, 2=critical
        self.last_change = 0.0

        # Initialize history
        for _ in range(10):
            self.history.add(initial_value)

    def update(self, new_target, smoothing=0.1, momentum_factor=0.05):
        """Update metric with professional smoothing and momentum"""
        self.target_value = np.clip(new_target, self.min_val, self.max_val)

        # Apply momentum
        change = self.target_value - self.value
        self.momentum = self.momentum * 0.9 + change * momentum_factor

        # Smooth update with momentum
        self.value += (change * smoothing) + self.momentum
        self.value = np.clip(self.value, self.min_val, self.max_val)

        # Track change
        self.last_change = change

        # Update history and statistics
        self.history.add(self.value)
        self.volatility = self.history.std(20)

        # Update status
        self.update_status()

    def update_status(self):
        """Update metric status based on trends and thresholds"""
        trend = self.history.trend(15)

        # Status determination
        if self.name.lower() in ["accuracy", "f1_score", "precision", "recall"]:
            # Higher is better metrics
            if self.value > 85:
                self.status = "excellent"
                self.alert_level = 0
            elif self.value > 70:
                self.status = "good"
                self.alert_level = 0
            elif self.value > 50:
                self.status = "stable"
                self.alert_level = 1 if trend < -0.3 else 0
            else:
                self.status = "critical"
                self.alert_level = 2

        elif self.name.lower() in ["loss", "error_rate"]:
            # Lower is better metrics
            if self.value < 0.1:
                self.status = "excellent"
                self.alert_level = 0
            elif self.value < 0.3:
                self.status = "good"
                self.alert_level = 0
            elif self.value < 0.7:
                self.status = "stable"
                self.alert_level = 1 if trend > 0.3 else 0
            else:
                self.status = "critical"
                self.alert_level = 2

        else:
            # General metrics
            normalized = (self.value - self.min_val) / (self.max_val - self.min_val)
            if normalized > 0.8:
                self.status = "excellent"
            elif normalized > 0.6:
                self.status = "good"
            elif normalized > 0.4:
                self.status = "stable"
            else:
                self.status = "degrading"

            self.alert_level = 2 if normalized < 0.2 else (1 if normalized < 0.4 else 0)

    def get_display_value(self):
        """Get formatted display value"""
        return self.format_str.format(self.value) + self.unit


class MLStats:
    """Industrial-level ML statistics dashboard"""

    def __init__(self):
        # Core Performance Metrics
        self.metrics = {
            "accuracy": MLMetric("Accuracy", 50.0, 0, 100, "%"),
            "loss": MLMetric("Loss", 1.0, 0, 5.0, "", "{:.4f}"),
            "f1_score": MLMetric("F1 Score", 0.5, 0, 1.0, "", "{:.3f}"),
            "precision": MLMetric("Precision", 0.5, 0, 1.0, "", "{:.3f}"),
            "recall": MLMetric("Recall", 0.5, 0, 1.0, "", "{:.3f}"),
        }

        # Training Dynamics
        self.training_metrics = {
            "learning_rate": MLMetric(
                "Learning Rate", 0.001, 0.0001, 0.1, "", "{:.4f}"
            ),
            "gradient_norm": MLMetric("Gradient Norm", 1.0, 0, 10.0, "", "{:.3f}"),
            "weight_norm": MLMetric("Weight Norm", 5.0, 0, 20.0, "", "{:.2f}"),
            "batch_size": MLMetric("Batch Size", 32, 1, 512, "", "{:.0f}"),
        }

        # Model Health Indicators
        self.health_metrics = {
            "gradient_health": MLMetric("Gradient Health", 100.0, 0, 100, "%"),
            "training_stability": MLMetric("Training Stability", 100.0, 0, 100, "%"),
            "convergence_rate": MLMetric(
                "Convergence Rate", 1.0, 0, 3.0, "x", "{:.2f}"
            ),
            "overfitting_risk": MLMetric("Overfitting Risk", 0.0, 0, 100, "%"),
            "generalization_gap": MLMetric("Generalization Gap", 5.0, 0, 50, "%"),
        }

        # Advanced Metrics
        self.advanced_metrics = {
            "model_complexity": MLMetric(
                "Model Complexity", 50.0, 0, 100, "", "{:.1f}"
            ),
            "data_efficiency": MLMetric("Data Efficiency", 70.0, 0, 100, "%"),
            "computational_cost": MLMetric(
                "Computational Cost", 1.0, 0.1, 10.0, "x", "{:.2f}"
            ),
            "memory_usage": MLMetric("Memory Usage", 512, 128, 8192, "MB", "{:.0f}"),
            "inference_speed": MLMetric(
                "Inference Speed", 100, 1, 1000, "FPS", "{:.0f}"
            ),
        }

        # Hyperparameter Tracking
        self.hyperparams = {
            "optimizer": "Adam",
            "scheduler": "CosineAnnealing",
            "regularization": "L2 + Dropout",
            "architecture": "Custom",
            "epochs_completed": 0,
            "total_parameters": 0,
            "trainable_parameters": 0,
        }

        # Real-time Events
        self.active_events = []
        self.event_log = deque(maxlen=50)
        self.active_buffs = []
        self.active_debuffs = []

        # Professional Dashboard State
        self.dashboard_mode = "overview"  # overview, detailed, minimal
        self.selected_metric_group = "core"
        self.alert_system_active = True
        self.performance_baseline = {}

        # Visual elements
        self.pulse_timer = 0
        self.alert_flash_timer = 0
        self.chart_scroll_offset = 0
        self.warning_flash = 0

        print("📊 Industrial ML Stats Dashboard initialized!")
        self.setup_performance_baseline()

    def setup_performance_baseline(self):
        """Initialize performance baselines for comparison"""
        for category in [
            self.metrics,
            self.training_metrics,
            self.health_metrics,
            self.advanced_metrics,
        ]:
            for name, metric in category.items():
                self.performance_baseline[name] = metric.value

        # Update buffs and debuffs
        self.update_buffs_debuffs()

        # Flash warning effects
        if self.warning_flash > 0:
            self.warning_flash -= 1

    def update(self, game_stats, player_status):
        """Update ML stats based on game state"""
        # Update core metrics based on game performance
        self.metrics["accuracy"].update(player_status.get("accuracy_bonus", 0))
        self.metrics["loss"].update(max(0, 1.0 - (game_stats.score / 1000.0)))
        self.metrics["f1_score"].update(min(1.0, game_stats.obstacles_dodged / 100.0))
        self.metrics["precision"].update(min(1.0, (game_stats.score / 1000.0)))
        self.metrics["recall"].update(min(1.0, game_stats.powerups_collected / 50.0))

        # Update training metrics
        self.training_metrics["learning_rate"].update(
            0.001 + (game_stats.score / 100000.0)
        )
        self.training_metrics["gradient_norm"].update(
            player_status.get("gradient_norm", 1.0)
        )
        self.training_metrics["weight_norm"].update(5.0 + (game_stats.score / 500.0))
        self.training_metrics["batch_size"].update(
            32 + (game_stats.obstacles_dodged // 10)
        )

        # Update health metrics
        self.health_metrics["gradient_health"].update(
            max(0, 100 - (game_stats.score // 100))
        )
        self.health_metrics["training_stability"].update(
            min(100, 50 + (game_stats.powerups_collected * 5))
        )
        self.health_metrics["convergence_rate"].update(
            min(3.0, 1.0 + (game_stats.score / 2000.0))
        )
        self.health_metrics["overfitting_risk"].update(min(100, game_stats.score / 100))
        self.health_metrics["generalization_gap"].update(
            max(0, 50 - (game_stats.obstacles_dodged / 10))
        )

        # Update performance baseline
        self.update_buffs_debuffs()

        # Flash warning effects
        if self.warning_flash > 0:
            self.warning_flash -= 1

    # ...existing update methods...

    def update_training_stability(self, player_status):
        """Update training stability metric"""
        if player_status.get("adrenaline", False):
            # Adrenaline mode = less stable but faster learning
            self.training_stability = max(30, self.training_stability - 1.0)
            self.convergence_rate = min(3.0, self.convergence_rate + 0.1)
        else:
            # Stable training
            self.training_stability = min(100, self.training_stability + 0.3)
            self.convergence_rate = max(1.0, self.convergence_rate - 0.05)

    def update_overfitting_risk(self, game_performance):
        """Update overfitting risk based on gameplay patterns"""
        # High performance with low variety increases overfitting risk
        if hasattr(game_performance, "pattern_variety"):
            if game_performance.pattern_variety < 0.5:
                self.overfitting_risk = min(100, self.overfitting_risk + 0.5)
            else:
                self.overfitting_risk = max(0, self.overfitting_risk - 0.2)
        else:
            # Default behavior
            self.overfitting_risk = max(0, self.overfitting_risk - 0.1)

    def update_convergence_rate(self, player_status):
        """Update how fast the model is learning"""
        if player_status.get("immune", False):
            # Dropout/regularization improves convergence
            self.convergence_rate = min(2.5, self.convergence_rate + 0.1)

        if player_status.get("boosted", False):
            # Layer boost improves learning
            self.convergence_rate = min(3.0, self.convergence_rate + 0.2)

    def update_histories(self):
        """Update metric histories for graphs"""
        self.accuracy_history.pop(0)
        self.accuracy_history.append(self.accuracy)

        self.loss_history.pop(0)
        self.loss_history.append(self.loss)

        self.gradient_history.pop(0)
        self.gradient_history.append(self.gradient_health)

    def add_buff(self, buff_name, duration, effect):
        """Add a temporary buff"""
        self.active_buffs.append(
            {
                "name": buff_name,
                "effect": effect,
                "duration": duration,
                "timer": duration * FPS,
            }
        )
        print(f"✨ ML Buff: {buff_name}")

    def add_debuff(self, debuff_name, duration, effect):
        """Add a temporary debuff"""
        self.active_debuffs.append(
            {
                "name": debuff_name,
                "effect": effect,
                "duration": duration,
                "timer": duration * FPS,
            }
        )
        print(f"⚠️ ML Debuff: {debuff_name}")

    def update_buffs_debuffs(self):
        """Update buff and debuff timers"""
        # Update buffs
        for buff in self.active_buffs[:]:
            buff["timer"] -= 1
            if buff["timer"] <= 0:
                self.active_buffs.remove(buff)
                print(f"⏰ Buff expired: {buff['name']}")

        # Update debuffs
        for debuff in self.active_debuffs[:]:
            debuff["timer"] -= 1
            if debuff["timer"] <= 0:
                self.active_debuffs.remove(debuff)
                print(f"⏰ Debuff expired: {debuff['name']}")

    def apply_damage_effect(self, damage_type):
        """Apply ML-specific effects based on damage type"""
        if damage_type == "overfitting":
            self.add_debuff("Overfitting", 5, "reduced_generalization")
            self.overfitting_risk = min(100, self.overfitting_risk + 20)
            self.generalization_score = max(0, self.generalization_score - 15)

        elif damage_type == "vanishing_gradient":
            self.add_debuff("Vanishing Gradients", 3, "learning_blocked")
            self.gradient_health = max(0, self.gradient_health - 30)
            self.convergence_rate = max(0.1, self.convergence_rate - 0.5)

        elif damage_type == "noisy_data":
            self.add_debuff("Noisy Training", 4, "unstable_learning")
            self.training_stability = max(0, self.training_stability - 20)
            self.accuracy = max(0, self.accuracy - 10)

        elif damage_type == "dead_neuron":
            self.add_debuff("Dead Neurons", 6, "reduced_capacity")
            self.accuracy = max(0, self.accuracy - 15)
            self.convergence_rate = max(0.1, self.convergence_rate - 0.3)

    def apply_powerup_effect(self, powerup_type):
        """Apply ML-specific effects based on powerup type"""
        if powerup_type == "dataset":
            self.add_buff("Rich Dataset", 8, "improved_learning")
            self.accuracy = min(99, self.accuracy + 10)
            self.generalization_score = min(100, self.generalization_score + 15)

        elif powerup_type == "optimizer":
            self.add_buff("Advanced Optimizer", 6, "fast_convergence")
            self.convergence_rate = min(3.0, self.convergence_rate + 0.8)
            self.training_stability = min(100, self.training_stability + 20)

        elif powerup_type == "dropout":
            self.add_buff("Regularization", 10, "overfitting_protection")
            self.overfitting_risk = max(0, self.overfitting_risk - 30)
            self.generalization_score = min(100, self.generalization_score + 20)

    def get_performance_grade(self):
        """Get overall performance grade"""
        overall_score = (
            self.accuracy
            + (100 - self.overfitting_risk)
            + self.gradient_health
            + self.training_stability
        ) / 4

        if overall_score >= 90:
            return "A+", COLORS["ACTIVATION"]
        elif overall_score >= 80:
            return "A", COLORS["DATASET"]
        elif overall_score >= 70:
            return "B", COLORS["NEURON_BLUE"]
        elif overall_score >= 60:
            return "C", COLORS["OPTIMIZER"]
        elif overall_score >= 50:
            return "D", COLORS["GRADIENT"]
        else:
            return "F", COLORS["LOSS"]

    def draw_ml_stats_panel(self, screen):
        """Draw comprehensive ML stats panel"""
        panel_x = 20
        panel_y = 320
        panel_width = 350
        panel_height = 280

        # Panel background
        panel_bg = pygame.Surface((panel_width, panel_height))
        panel_bg.set_alpha(180)
        panel_bg.fill(COLORS["BLACK"])
        screen.blit(panel_bg, (panel_x, panel_y))

        # Panel border with warning flash
        border_color = COLORS["WHITE"]
        if self.warning_flash > 0:
            flash_intensity = self.warning_flash / 20
            border_color = (
                255,
                int(255 * (1 - flash_intensity)),
                int(255 * (1 - flash_intensity)),
            )

        pygame.draw.rect(
            screen, border_color, (panel_x, panel_y, panel_width, panel_height), 3
        )

        # Title
        font_title = pygame.font.Font(None, 28)
        title_text = font_title.render(
            "🧪 ML Performance Metrics", True, COLORS["WHITE"]
        )
        screen.blit(title_text, (panel_x + 10, panel_y + 10))

        # Core metrics
        font_metric = pygame.font.Font(None, 24)
        y_offset = panel_y + 45

        # Accuracy with bar
        accuracy_text = font_metric.render(
            f"Accuracy: {self.accuracy:.1f}%", True, COLORS["WHITE"]
        )
        screen.blit(accuracy_text, (panel_x + 10, y_offset))
        self.draw_metric_bar(
            screen,
            panel_x + 150,
            y_offset + 5,
            150,
            12,
            self.accuracy,
            100,
            COLORS["ACTIVATION"],
        )
        y_offset += 30

        # Loss with bar
        loss_text = font_metric.render(f"Loss: {self.loss:.3f}", True, COLORS["WHITE"])
        screen.blit(loss_text, (panel_x + 10, y_offset))
        loss_percent = max(0, min(100, (2.0 - self.loss) / 2.0 * 100))  # Invert for bar
        self.draw_metric_bar(
            screen,
            panel_x + 150,
            y_offset + 5,
            150,
            12,
            loss_percent,
            100,
            COLORS["GRADIENT"],
        )
        y_offset += 30

        # Gradient Health
        grad_color = (
            COLORS["ACTIVATION"] if self.gradient_health > 50 else COLORS["LOSS"]
        )
        grad_text = font_metric.render(
            f"Gradient Health: {self.gradient_health:.1f}%", True, COLORS["WHITE"]
        )
        screen.blit(grad_text, (panel_x + 10, y_offset))
        self.draw_metric_bar(
            screen,
            panel_x + 180,
            y_offset + 5,
            120,
            12,
            self.gradient_health,
            100,
            grad_color,
        )
        y_offset += 30

        # Training Stability
        stability_color = (
            COLORS["DATASET"] if self.training_stability > 70 else COLORS["OPTIMIZER"]
        )
        stability_text = font_metric.render(
            f"Training Stability: {self.training_stability:.1f}%", True, COLORS["WHITE"]
        )
        screen.blit(stability_text, (panel_x + 10, y_offset))
        self.draw_metric_bar(
            screen,
            panel_x + 180,
            y_offset + 5,
            120,
            12,
            self.training_stability,
            100,
            stability_color,
        )
        y_offset += 30

        # Overfitting Risk (red is bad)
        risk_color = (
            COLORS["LOSS"] if self.overfitting_risk > 50 else COLORS["ACTIVATION"]
        )
        risk_text = font_metric.render(
            f"Overfitting Risk: {self.overfitting_risk:.1f}%", True, COLORS["WHITE"]
        )
        screen.blit(risk_text, (panel_x + 10, y_offset))
        self.draw_metric_bar(
            screen,
            panel_x + 180,
            y_offset + 5,
            120,
            12,
            self.overfitting_risk,
            100,
            risk_color,
        )
        y_offset += 30

        # Convergence Rate
        conv_text = font_metric.render(
            f"Learning Rate: {self.convergence_rate:.2f}x", True, COLORS["WHITE"]
        )
        screen.blit(conv_text, (panel_x + 10, y_offset))
        y_offset += 25

        # Performance Grade
        grade, grade_color = self.get_performance_grade()
        grade_text = font_title.render(f"Grade: {grade}", True, grade_color)
        screen.blit(grade_text, (panel_x + 10, y_offset))

        # Draw active buffs/debuffs
        self.draw_buffs_debuffs(screen, panel_x + panel_width + 10, panel_y)

    def draw_metric_bar(self, screen, x, y, width, height, value, max_value, color):
        """Draw a metric bar with value"""
        # Background
        pygame.draw.rect(screen, COLORS["GRAY"], (x, y, width, height))

        # Fill
        fill_width = int((value / max_value) * width)
        if fill_width > 0:
            pygame.draw.rect(screen, color, (x, y, fill_width, height))

        # Border
        pygame.draw.rect(screen, COLORS["WHITE"], (x, y, width, height), 1)

        # Pulsing effect for critical values
        if value < 20 or (max_value == 100 and value > 80 and color == COLORS["LOSS"]):
            pulse = abs(math.sin(self.pulse_timer * 3)) * 50
            warning_color = (255, int(pulse), int(pulse))
            pygame.draw.rect(screen, warning_color, (x, y, width, height), 2)

    def draw_buffs_debuffs(self, screen, x, y):
        """Draw active buffs and debuffs"""
        if not self.active_buffs and not self.active_debuffs:
            return

        panel_width = 200
        total_effects = len(self.active_buffs) + len(self.active_debuffs)
        panel_height = 40 + total_effects * 25

        # Background
        panel_bg = pygame.Surface((panel_width, panel_height))
        panel_bg.set_alpha(150)
        panel_bg.fill(COLORS["BLACK"])
        screen.blit(panel_bg, (x, y))
        pygame.draw.rect(screen, COLORS["WHITE"], (x, y, panel_width, panel_height), 2)

        # Title
        font_small = pygame.font.Font(None, 20)
        title_text = font_small.render("Active Effects", True, COLORS["WHITE"])
        screen.blit(title_text, (x + 10, y + 10))

        y_offset = y + 35

        # Draw buffs
        for buff in self.active_buffs:
            remaining = buff["timer"] // FPS
            buff_text = font_small.render(
                f"✨ {buff['name']} ({remaining}s)", True, COLORS["ACTIVATION"]
            )
            screen.blit(buff_text, (x + 10, y_offset))
            y_offset += 25

        # Draw debuffs
        for debuff in self.active_debuffs:
            remaining = debuff["timer"] // FPS
            debuff_text = font_small.render(
                f"⚠️ {debuff['name']} ({remaining}s)", True, COLORS["LOSS"]
            )
            screen.blit(debuff_text, (x + 10, y_offset))
            y_offset += 25

    def draw_mini_graphs(self, screen):
        """Draw mini performance graphs"""
        graph_x = SCREEN_WIDTH - 220
        graph_y = 400
        graph_width = 200
        graph_height = 60

        # Accuracy graph
        self.draw_mini_graph(
            screen,
            graph_x,
            graph_y,
            graph_width,
            graph_height,
            self.accuracy_history,
            "Accuracy",
            COLORS["ACTIVATION"],
        )

        # Loss graph
        self.draw_mini_graph(
            screen,
            graph_x,
            graph_y + 80,
            graph_width,
            graph_height,
            self.loss_history,
            "Loss",
            COLORS["GRADIENT"],
            invert=True,
        )

        # Gradient health graph
        self.draw_mini_graph(
            screen,
            graph_x,
            graph_y + 160,
            graph_width,
            graph_height,
            self.gradient_history,
            "Gradient Health",
            COLORS["NEURON_BLUE"],
        )

    def draw_mini_graph(
        self, screen, x, y, width, height, data, title, color, invert=False
    ):
        """Draw a mini graph for a metric"""
        # Background
        graph_bg = pygame.Surface((width, height + 20))
        graph_bg.set_alpha(150)
        graph_bg.fill(COLORS["BLACK"])
        screen.blit(graph_bg, (x, y - 20))
        pygame.draw.rect(screen, color, (x, y - 20, width, height + 20), 2)

        # Title
        font_small = pygame.font.Font(None, 18)
        title_text = font_small.render(title, True, COLORS["WHITE"])
        screen.blit(title_text, (x + 5, y - 18))

        # Graph data
        if len(data) > 1:
            points = []
            for i, value in enumerate(data):
                px = x + (i / len(data)) * width
                if invert:
                    py = y + height - (value / 2.0) * height  # For loss (max 2.0)
                else:
                    py = y + height - (value / 100) * height  # For percentages
                points.append((px, py))

            if len(points) > 1:
                pygame.draw.lines(screen, color, False, points, 2)

    def reset(self):
        """Reset ML stats"""
        self.accuracy = 50.0
        self.loss = 1.0
        self.gradient_health = 100.0
        self.training_stability = 100.0
        self.convergence_rate = 1.0
        self.overfitting_risk = 0.0
        self.generalization_score = 50.0
        self.active_buffs.clear()
        self.active_debuffs.clear()
        self.warning_flash = 0

    def draw(self, screen):
        """Draw the ML stats dashboard"""
        if not hasattr(self, "font_large") or self.font_large is None:
            # Initialize fonts if needed
            try:
                self.font_large = pygame.font.Font(None, 36)
                self.font_medium = pygame.font.Font(None, 24)
                self.font_small = pygame.font.Font(None, 18)
            except:
                self.font_large = pygame.font.Font(None, 24)
                self.font_medium = pygame.font.Font(None, 18)
                self.font_small = pygame.font.Font(None, 16)

        # Draw a minimal stats overlay
        stats_rect = pygame.Rect(SCREEN_WIDTH - 200, 10, 190, 80)

        # Background
        bg_surface = pygame.Surface(
            (stats_rect.width, stats_rect.height), pygame.SRCALPHA
        )
        bg_surface.fill((*COLORS["BLACK"], 180))
        screen.blit(bg_surface, stats_rect)

        # Border
        pygame.draw.rect(screen, COLORS["ACTIVATION"], stats_rect, 2)

        # Title
        title_text = self.font_medium.render("ML Stats", True, COLORS["WHITE"])
        screen.blit(title_text, (stats_rect.x + 5, stats_rect.y + 5))

        # Key metrics
        y_offset = 30
        for name, metric in list(self.metrics.items())[:3]:  # Show first 3 metrics
            value_text = f"{name}: {metric.value:.1f}"
            text_surface = self.font_small.render(value_text, True, COLORS["GRAY"])
            screen.blit(text_surface, (stats_rect.x + 5, stats_rect.y + y_offset))
            y_offset += 15


# Global ML stats instance
ml_stats = MLStats()


def update(self, game_stats, player_status):
    """Update ML stats based on game state"""
    # Update core metrics based on game performance
    self.metrics["accuracy"].update(player_status.get("accuracy_bonus", 0))
    self.metrics["loss"].update(max(0, 1.0 - (game_stats.score / 1000.0)))
    self.metrics["learning_rate"].update(0.001 + (game_stats.score / 100000.0))
    self.metrics["batch_size"].update(32 + (game_stats.obstacles_dodged // 10))

    # Update training metrics
    self.training_metrics["epoch"].update(game_stats.score // 100)
    self.training_metrics["gradient_norm"].update(
        player_status.get("gradient_norm", 1.0)
    )
    self.training_metrics["weight_decay"].update(0.0001)
    self.training_metrics["momentum"].update(0.9)

    # Update health metrics
    self.health_metrics["gpu_temp"].update(60 + (game_stats.score / 100) % 20)
    self.health_metrics["memory_usage"].update(
        min(95, 40 + (game_stats.powerups_collected * 5))
    )
    self.health_metrics["cpu_usage"].update(
        min(90, 30 + (game_stats.obstacles_dodged // 5))
    )
    self.health_metrics["disk_io"].update(min(100, 20 + (game_stats.score // 50)))

    # Update advanced metrics
    self.advanced_metrics["perplexity"].update(max(1, 100 - (game_stats.score / 50)))
    self.advanced_metrics["bleu_score"].update(min(1.0, game_stats.score / 5000.0))
    self.advanced_metrics["rouge_score"].update(
        min(1.0, game_stats.powerups_collected / 50.0)
    )
    self.advanced_metrics["f1_score"].update(
        min(1.0, game_stats.obstacles_dodged / 100.0)
    )

    # Update performance baseline
    self.update_buffs_debuffs()

    # Flash warning effects
    if self.warning_flash > 0:
        self.warning_flash -= 1


# ...existing code...
