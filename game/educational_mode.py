"""
🎓 Educational Mode System
Interactive tutorials and educational content with step-by-step guidance
"""

import pygame
import time
from .config import *


class Tutorial:
    """Represents a single tutorial lesson"""

    def __init__(self, id, title, description, steps, prerequisites=None):
        self.id = id
        self.title = title
        self.description = description
        self.steps = steps
        self.prerequisites = prerequisites or []
        self.completed = False
        self.current_step = 0
        self.step_completed = False

    def reset(self):
        """Reset tutorial progress"""
        self.current_step = 0
        self.step_completed = False
        self.completed = False

    def next_step(self):
        """Advance to next step"""
        if self.current_step < len(self.steps) - 1:
            self.current_step += 1
            self.step_completed = False
            return True
        else:
            self.completed = True
            return False

    def get_current_step(self):
        """Get current tutorial step"""
        if self.current_step < len(self.steps):
            return self.steps[self.current_step]
        return None


class TutorialStep:
    """Represents a single step in a tutorial"""

    def __init__(
        self,
        title,
        instruction,
        condition_type,
        condition_data,
        hint=None,
        highlight_area=None,
    ):
        self.title = title
        self.instruction = instruction
        self.condition_type = (
            condition_type  # 'action', 'time', 'score', 'collision', 'powerup'
        )
        self.condition_data = condition_data  # Data needed to check condition
        self.hint = hint
        self.highlight_area = highlight_area  # Area to highlight on screen
        self.completion_time = 0

    def check_completion(self, game_state):
        """Check if step condition is met"""
        if self.condition_type == "action":
            # Check for specific key press
            keys = pygame.key.get_pressed()
            return keys[self.condition_data["key"]]

        elif self.condition_type == "time":
            # Wait for specific duration
            self.completion_time += 1 / 60  # Assuming 60 FPS
            return self.completion_time >= self.condition_data["duration"]

        elif self.condition_type == "score":
            # Reach specific score
            return game_state.get("score", 0) >= self.condition_data["target"]

        elif self.condition_type == "collision":
            # Experience collision with specific obstacle
            return (
                game_state.get("last_collision") == self.condition_data["obstacle_type"]
            )

        elif self.condition_type == "powerup":
            # Collect specific powerup
            return game_state.get("last_powerup") == self.condition_data["powerup_type"]

        elif self.condition_type == "custom":
            # Custom condition function
            return self.condition_data["function"](game_state)

        return False


class EducationalMode:
    """Main educational mode system with tutorials and explanations"""

    def __init__(self):
        self.active = False
        self.current_tutorial = None
        self.tutorial_index = 0
        self.paused_for_tutorial = False

        # UI elements
        self.tutorial_panel_rect = pygame.Rect(50, 50, SCREEN_WIDTH - 100, 200)
        self.tooltip_rect = pygame.Rect(0, 0, 300, 100)
        self.show_tooltips = True

        # Tutorial progress
        self.completed_tutorials = set()
        self.unlocked_tutorials = set()

        # Animation
        self.panel_slide_progress = 0
        self.highlight_pulse = 0

        # Initialize tutorials
        self.tutorials = self.create_tutorials()
        self.unlock_initial_tutorials()

        print("🎓 Educational mode initialized")

    def create_tutorials(self):
        """Create all tutorial lessons"""
        tutorials = []

        # Basic Controls Tutorial
        tutorials.append(
            Tutorial(
                id="basic_controls",
                title="🎮 Basic Controls",
                description="Learn the fundamental controls of Neuron Runner",
                steps=[
                    TutorialStep(
                        "Welcome to Neuron Runner!",
                        "You are a neural network learning to navigate the world of AI training. Let's start with basic movement.",
                        "time",
                        {"duration": 3.0},
                        "This game teaches you about AI while you play!",
                    ),
                    TutorialStep(
                        "Jump",
                        "Press SPACE to make your neuron jump over obstacles.",
                        "action",
                        {"key": pygame.K_SPACE},
                        "Jumping helps avoid overfitting obstacles!",
                        {"type": "key", "key": "SPACE"},
                    ),
                    TutorialStep(
                        "Duck",
                        "Press DOWN ARROW to duck under high obstacles.",
                        "action",
                        {"key": pygame.K_DOWN},
                        "Ducking represents neural network compression!",
                        {"type": "key", "key": "DOWN"},
                    ),
                    TutorialStep(
                        "Layer Boost",
                        "Press L to activate your layer boost ability for temporary immunity.",
                        "action",
                        {"key": pygame.K_l},
                        "Layer boost adds a dropout layer for protection!",
                        {"type": "key", "key": "L"},
                    ),
                ],
            )
        )

        # AI Concepts Tutorial
        tutorials.append(
            Tutorial(
                id="ai_concepts",
                title="🧠 AI Concepts",
                description="Learn about neural networks and machine learning",
                steps=[
                    TutorialStep(
                        "Neural Networks",
                        "Your character represents a neural network - a computer system inspired by biological brains.",
                        "time",
                        {"duration": 4.0},
                        "Neural networks learn by adjusting connections between 'neurons'",
                    ),
                    TutorialStep(
                        "Training Process",
                        "As you play, your network is 'training' - learning to perform better through experience.",
                        "score",
                        {"target": 100},
                        "Score represents how well your network is learning!",
                    ),
                    TutorialStep(
                        "Overfitting",
                        "Avoid the red 'Overfitting' obstacles - they represent learning the training data too well.",
                        "collision",
                        {"obstacle_type": "OVERFITTING"},
                        "Overfitting makes models perform poorly on new data!",
                    ),
                ],
                prerequisites=["basic_controls"],
            )
        )

        # Model Skins Tutorial
        tutorials.append(
            Tutorial(
                id="model_skins",
                title="🎨 Model Types",
                description="Explore different AI model architectures",
                steps=[
                    TutorialStep(
                        "Model Architectures",
                        "Different AI models excel at different tasks. You can choose your model type!",
                        "time",
                        {"duration": 3.0},
                        "Each model has unique strengths and abilities",
                    ),
                    TutorialStep(
                        "CNN (Convolutional Neural Network)",
                        "CNNs are great for image processing and pattern recognition. They're fast and efficient!",
                        "time",
                        {"duration": 4.0},
                        "Used in computer vision applications like photo recognition",
                    ),
                    TutorialStep(
                        "RNN (Recurrent Neural Network)",
                        "RNNs process sequences and have memory. They're perfect for text and time series data.",
                        "time",
                        {"duration": 4.0},
                        "The 'memory' allows them to understand context in sentences",
                    ),
                    TutorialStep(
                        "Transformers",
                        "Transformers use 'attention' to focus on important parts of input. Very powerful for language!",
                        "time",
                        {"duration": 4.0},
                        "GPT and BERT are famous transformer models",
                    ),
                ],
                prerequisites=["ai_concepts"],
            )
        )

        # Powerups Tutorial
        tutorials.append(
            Tutorial(
                id="powerups",
                title="⚡ Training Resources",
                description="Learn about datasets, optimizers, and other ML tools",
                steps=[
                    TutorialStep(
                        "Datasets",
                        "Green dataset powerups represent training data - the fuel of machine learning!",
                        "powerup",
                        {"powerup_type": "DATASET"},
                        "More and better data usually means better AI models",
                    ),
                    TutorialStep(
                        "Optimizers",
                        "Blue optimizer powerups represent algorithms that help networks learn faster and better.",
                        "powerup",
                        {"powerup_type": "OPTIMIZER"},
                        "Adam, SGD, and RMSprop are popular optimizers",
                    ),
                    TutorialStep(
                        "Regularization",
                        "Purple dropout powerups represent techniques that prevent overfitting.",
                        "powerup",
                        {"powerup_type": "DROPOUT"},
                        "Dropout randomly 'turns off' some neurons during training",
                    ),
                ],
                prerequisites=["ai_concepts"],
            )
        )

        # Advanced Features Tutorial
        tutorials.append(
            Tutorial(
                id="advanced_features",
                title="🔧 Advanced Features",
                description="Explore the model builder and other advanced tools",
                steps=[
                    TutorialStep(
                        "Model Builder",
                        "Press B to open the model builder - design your own neural network architecture!",
                        "action",
                        {"key": pygame.K_b},
                        "Drag and drop layers to create custom models",
                        {"type": "key", "key": "B"},
                    ),
                    TutorialStep(
                        "Training Journal",
                        "Press J to view your training journal - track progress and unlock achievements!",
                        "action",
                        {"key": pygame.K_j},
                        "Your journal remembers all your training sessions",
                        {"type": "key", "key": "J"},
                    ),
                    TutorialStep(
                        "Special Abilities",
                        "Press X to use your model's special ability - each AI type has unique powers!",
                        "action",
                        {"key": pygame.K_x},
                        "CNNs get speed burst, Transformers can teleport!",
                        {"type": "key", "key": "X"},
                    ),
                ],
                prerequisites=["model_skins", "powerups"],
            )
        )

        return tutorials

    def unlock_initial_tutorials(self):
        """Unlock the first set of tutorials"""
        self.unlocked_tutorials.add("basic_controls")

    def activate(self):
        """Activate educational mode"""
        self.active = True
        self.start_next_tutorial()
        print("🎓 Educational mode activated")

    def deactivate(self):
        """Deactivate educational mode"""
        self.active = False
        self.current_tutorial = None
        self.paused_for_tutorial = False
        print("🎓 Educational mode deactivated")

    def start_tutorial(self, tutorial_id):
        """Start a specific tutorial"""
        tutorial = next((t for t in self.tutorials if t.id == tutorial_id), None)
        if tutorial and tutorial_id in self.unlocked_tutorials:
            self.current_tutorial = tutorial
            tutorial.reset()
            self.paused_for_tutorial = True
            self.panel_slide_progress = 0
            print(f"🎓 Started tutorial: {tutorial.title}")
            return True
        return False

    def start_next_tutorial(self):
        """Start the next available tutorial"""
        for tutorial in self.tutorials:
            if (
                tutorial.id in self.unlocked_tutorials
                and tutorial.id not in self.completed_tutorials
            ):
                return self.start_tutorial(tutorial.id)
        return False

    def complete_current_tutorial(self):
        """Complete the current tutorial"""
        if self.current_tutorial:
            self.completed_tutorials.add(self.current_tutorial.id)
            self.unlock_dependent_tutorials(self.current_tutorial.id)

            print(f"✅ Tutorial completed: {self.current_tutorial.title}")

            self.current_tutorial = None
            self.paused_for_tutorial = False

            # Auto-start next tutorial
            self.start_next_tutorial()

    def unlock_dependent_tutorials(self, completed_tutorial_id):
        """Unlock tutorials that depend on the completed one"""
        for tutorial in self.tutorials:
            if completed_tutorial_id in tutorial.prerequisites:
                all_prerequisites_met = all(
                    prereq in self.completed_tutorials
                    for prereq in tutorial.prerequisites
                )
                if all_prerequisites_met:
                    self.unlocked_tutorials.add(tutorial.id)
                    print(f"🔓 Unlocked tutorial: {tutorial.title}")

    def handle_event(self, event):
        """Handle educational mode events"""
        if not self.active:
            return False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.deactivate()
                return True
            elif event.key == pygame.K_n and self.current_tutorial:
                # Next step
                if not self.current_tutorial.next_step():
                    self.complete_current_tutorial()
                return True
            elif event.key == pygame.K_s and self.current_tutorial:
                # Skip current tutorial
                self.complete_current_tutorial()
                return True

        return False

    def update(self, game_state):
        """Update educational mode"""
        if not self.active or not self.current_tutorial:
            return

        # Update animations
        self.panel_slide_progress = min(1.0, self.panel_slide_progress + 0.05)
        self.highlight_pulse += 0.1

        # Check step completion
        current_step = self.current_tutorial.get_current_step()
        if current_step and not self.current_tutorial.step_completed:
            if current_step.check_completion(game_state):
                self.current_tutorial.step_completed = True
                # Auto-advance after a brief delay for some step types
                if current_step.condition_type in ["action", "powerup", "collision"]:
                    pygame.time.set_timer(
                        pygame.USEREVENT + 1, 1500
                    )  # 1.5 second delay

    def draw(self, screen):
        """Draw educational mode interface"""
        if not self.active or not self.current_tutorial:
            return

        self.draw_tutorial_panel(screen)
        self.draw_highlights(screen)
        self.draw_progress_indicator(screen)

    def draw_tutorial_panel(self, screen):
        """Draw the main tutorial panel"""
        current_step = self.current_tutorial.get_current_step()
        if not current_step:
            return

        # Slide animation
        slide_offset = (1 - self.panel_slide_progress) * -300
        panel_rect = pygame.Rect(
            self.tutorial_panel_rect.x,
            self.tutorial_panel_rect.y + slide_offset,
            self.tutorial_panel_rect.width,
            self.tutorial_panel_rect.height,
        )

        # Panel background
        panel_bg = pygame.Surface(
            (panel_rect.width, panel_rect.height), pygame.SRCALPHA
        )
        panel_bg.fill((*COLORS["DARK_BLUE"], 220))
        screen.blit(panel_bg, panel_rect)

        # Panel border
        border_color = (
            COLORS["ACTIVATION"]
            if self.current_tutorial.step_completed
            else COLORS["DATASET"]
        )
        pygame.draw.rect(screen, border_color, panel_rect, 3)

        # Tutorial title
        title_font = pygame.font.Font(None, 36)
        title_text = title_font.render(
            self.current_tutorial.title, True, COLORS["WHITE"]
        )
        screen.blit(title_text, (panel_rect.x + 20, panel_rect.y + 15))

        # Step title
        step_font = pygame.font.Font(None, 28)
        step_text = step_font.render(current_step.title, True, COLORS["ACTIVATION"])
        screen.blit(step_text, (panel_rect.x + 20, panel_rect.y + 55))

        # Instruction text
        instruction_font = pygame.font.Font(None, 24)
        instruction_lines = self.wrap_text(
            current_step.instruction, instruction_font, panel_rect.width - 40
        )

        for i, line in enumerate(instruction_lines):
            instruction_surface = instruction_font.render(line, True, COLORS["WHITE"])
            screen.blit(
                instruction_surface, (panel_rect.x + 20, panel_rect.y + 90 + i * 25)
            )

        # Hint text
        if current_step.hint:
            hint_font = pygame.font.Font(None, 20)
            hint_text = hint_font.render(
                f"💡 {current_step.hint}", True, COLORS["YELLOW"]
            )
            screen.blit(
                hint_text, (panel_rect.x + 20, panel_rect.y + panel_rect.height - 50)
            )

        # Controls help
        controls_font = pygame.font.Font(None, 18)
        if self.current_tutorial.step_completed:
            controls_text = controls_font.render(
                "Press N for next step | ESC to exit", True, COLORS["LIGHT_GRAY"]
            )
        else:
            controls_text = controls_font.render(
                "Complete the instruction above | S to skip | ESC to exit",
                True,
                COLORS["LIGHT_GRAY"],
            )

        screen.blit(
            controls_text, (panel_rect.x + 20, panel_rect.y + panel_rect.height - 25)
        )

    def draw_highlights(self, screen):
        """Draw highlight areas for tutorial steps"""
        current_step = self.current_tutorial.get_current_step()
        if not current_step or not current_step.highlight_area:
            return

        highlight = current_step.highlight_area
        pulse_alpha = int(100 + 50 * math.sin(self.highlight_pulse))

        if highlight["type"] == "key":
            # Highlight key instruction
            key_text = f"Press {highlight['key']}"
            key_font = pygame.font.Font(None, 48)
            key_surface = key_font.render(
                key_text, True, (*COLORS["YELLOW"], pulse_alpha)
            )

            key_rect = key_surface.get_rect(
                center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 100)
            )

            # Background for key highlight
            bg_rect = key_rect.inflate(40, 20)
            bg_surface = pygame.Surface(
                (bg_rect.width, bg_rect.height), pygame.SRCALPHA
            )
            bg_surface.fill((*COLORS["BLACK"], pulse_alpha))
            screen.blit(bg_surface, bg_rect)

            screen.blit(key_surface, key_rect)

    def draw_progress_indicator(self, screen):
        """Draw tutorial progress indicator"""
        if not self.current_tutorial:
            return

        progress_width = 300
        progress_height = 8
        progress_x = (SCREEN_WIDTH - progress_width) // 2
        progress_y = self.tutorial_panel_rect.bottom + 20

        # Progress background
        progress_bg = pygame.Rect(
            progress_x, progress_y, progress_width, progress_height
        )
        pygame.draw.rect(screen, COLORS["DARK_GRAY"], progress_bg)

        # Progress fill
        progress_ratio = (self.current_tutorial.current_step + 1) / len(
            self.current_tutorial.steps
        )
        progress_fill_width = int(progress_width * progress_ratio)
        progress_fill = pygame.Rect(
            progress_x, progress_y, progress_fill_width, progress_height
        )
        pygame.draw.rect(screen, COLORS["ACTIVATION"], progress_fill)

        # Step indicator
        step_text = f"Step {self.current_tutorial.current_step + 1} of {len(self.current_tutorial.steps)}"
        step_font = pygame.font.Font(None, 20)
        step_surface = step_font.render(step_text, True, COLORS["WHITE"])
        step_rect = step_surface.get_rect(
            center=(progress_x + progress_width // 2, progress_y + 25)
        )
        screen.blit(step_surface, step_rect)

    def wrap_text(self, text, font, max_width):
        """Wrap text to fit within max_width"""
        words = text.split(" ")
        lines = []
        current_line = []

        for word in words:
            test_line = " ".join(current_line + [word])
            if font.size(test_line)[0] <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(" ".join(current_line))
                    current_line = [word]
                else:
                    lines.append(word)

        if current_line:
            lines.append(" ".join(current_line))

        return lines

    def is_tutorial_mode_active(self):
        """Check if tutorial mode is currently active"""
        return self.active and self.current_tutorial is not None

    def should_pause_game(self):
        """Check if game should be paused for tutorial"""
        return self.paused_for_tutorial


# Global educational mode instance
educational_mode = EducationalMode()
