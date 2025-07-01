"""
📚 AI Fact Cards System
Educational flashcards and trivia that appear during gameplay
"""

import pygame
import random
import time
import math
from .config import *


class FactCard:
    """Individual AI fact card with question and answer"""

    def __init__(self, category, question, answer, difficulty="medium", fun_fact=None):
        self.category = category
        self.question = question
        self.answer = answer
        self.difficulty = difficulty
        self.fun_fact = fun_fact
        self.shown_count = 0
        self.last_shown = 0

    def to_dict(self):
        """Convert to dictionary for serialization"""
        return {
            "category": self.category,
            "question": self.question,
            "answer": self.answer,
            "difficulty": self.difficulty,
            "fun_fact": self.fun_fact,
            "shown_count": self.shown_count,
            "last_shown": self.last_shown,
        }


class FactCardSystem:
    """Manages AI fact cards and trivia system"""

    def __init__(self):
        self.facts = []
        self.current_card = None
        self.showing_card = False
        self.card_start_time = 0
        self.card_duration = 5.0  # seconds
        self.auto_advance = True

        # Trigger conditions
        self.last_trigger_score = 0
        self.trigger_interval = 1000  # Show card every 1000 points
        self.end_run_trigger = True

        # Animation
        self.slide_progress = 0
        self.slide_speed = 3.0
        self.pulse_timer = 0

        # Card dimensions and position
        self.card_width = 500
        self.card_height = 300
        self.card_x = (SCREEN_WIDTH - self.card_width) // 2
        self.card_y = (SCREEN_HEIGHT - self.card_height) // 2

        # Interaction
        self.user_dismissed = False

        # Initialize fact database
        self.initialize_facts()

        # Statistics
        self.cards_shown = 0
        self.categories_learned = set()

    def initialize_facts(self):
        """Initialize the database of AI facts"""
        self.facts = [
            # Machine Learning Basics
            FactCard(
                "ML Basics",
                "What does 'overfitting' mean in machine learning?",
                "When a model learns the training data too well and performs poorly on new data",
                "easy",
                "It's like memorizing answers without understanding - you ace the practice test but fail the real exam!",
            ),
            FactCard(
                "ML Basics",
                "What is the difference between supervised and unsupervised learning?",
                "Supervised learning uses labeled data, unsupervised learning finds patterns in unlabeled data",
                "easy",
                "Supervised is like learning with a teacher, unsupervised is like figuring things out on your own!",
            ),
            FactCard(
                "ML Basics",
                "What is a neural network activation function?",
                "A mathematical function that determines the output of a neural network node",
                "medium",
                "Popular ones include ReLU, Sigmoid, and Tanh - each with different strengths!",
            ),
            # Deep Learning
            FactCard(
                "Deep Learning",
                "What does CNN stand for and what is it used for?",
                "Convolutional Neural Network - primarily used for image processing and computer vision",
                "medium",
                "CNNs are inspired by how the visual cortex processes images in animal brains!",
            ),
            FactCard(
                "Deep Learning",
                "What is backpropagation?",
                "The algorithm used to train neural networks by calculating gradients and updating weights",
                "hard",
                "It's like learning from your mistakes - the network adjusts based on how wrong it was!",
            ),
            FactCard(
                "Deep Learning",
                "What is the vanishing gradient problem?",
                "When gradients become too small during backpropagation, preventing effective training of deep networks",
                "hard",
                "It's why ReLU activation and skip connections were game-changers for deep learning!",
            ),
            # AI History
            FactCard(
                "AI History",
                "Who coined the term 'Artificial Intelligence'?",
                "John McCarthy at the Dartmouth Conference in 1956",
                "medium",
                "The same conference that launched AI as a field of study!",
            ),
            FactCard(
                "AI History",
                "What was the first AI winter?",
                "A period from mid-1970s to early 1980s when AI research funding was drastically reduced",
                "medium",
                "It happened because AI promises were overhyped and computing power wasn't ready yet!",
            ),
            FactCard(
                "AI History",
                "What chess computer defeated world champion Garry Kasparov?",
                "IBM's Deep Blue in 1997",
                "easy",
                "It was a historic moment showing AI could beat humans at complex strategic games!",
            ),
            # Modern AI
            FactCard(
                "Modern AI",
                "What does GPT stand for?",
                "Generative Pre-trained Transformer",
                "easy",
                "The 'transformer' architecture revolutionized natural language processing!",
            ),
            FactCard(
                "Modern AI",
                "What is attention mechanism in neural networks?",
                "A technique that allows models to focus on relevant parts of input when making predictions",
                "hard",
                "It's like having a spotlight that can dynamically focus on important information!",
            ),
            FactCard(
                "Modern AI",
                "What is transfer learning?",
                "Using a pre-trained model as the starting point for a new, related task",
                "medium",
                "It's like learning to drive a truck when you already know how to drive a car!",
            ),
            # Fun Facts
            FactCard(
                "Fun Facts",
                "How many parameters does GPT-3 have?",
                "175 billion parameters",
                "medium",
                "That's more connections than there are stars in the Milky Way galaxy!",
            ),
            FactCard(
                "Fun Facts",
                "What AI technique is used in recommendation systems?",
                "Collaborative filtering and matrix factorization",
                "medium",
                "Netflix saves $1 billion per year thanks to their recommendation algorithm!",
            ),
            FactCard(
                "Fun Facts",
                "What is the Turing Test?",
                "A test to determine if a machine can exhibit intelligent behavior indistinguishable from a human",
                "easy",
                "Proposed by Alan Turing in 1950 - still a gold standard for AI evaluation!",
            ),
            # Technical Concepts
            FactCard(
                "Technical",
                "What is regularization in machine learning?",
                "Techniques to prevent overfitting by adding penalties or constraints to the model",
                "medium",
                "L1 and L2 regularization are like adding rules to keep your model honest!",
            ),
            FactCard(
                "Technical",
                "What is gradient descent?",
                "An optimization algorithm that finds the minimum of a function by iteratively moving in the direction of steepest descent",
                "medium",
                "Imagine rolling a ball down a hill to find the lowest point - that's gradient descent!",
            ),
            FactCard(
                "Technical",
                "What is batch normalization?",
                "A technique that normalizes the inputs to each layer in a neural network",
                "hard",
                "It helps training converge faster and makes networks more stable!",
            ),
            # AI Ethics & Future
            FactCard(
                "AI Ethics",
                "What is algorithmic bias?",
                "When AI systems produce prejudiced results due to biased training data or design",
                "medium",
                "It's a critical issue - AI systems can perpetuate and amplify human biases!",
            ),
            FactCard(
                "AI Ethics",
                "What is the AI alignment problem?",
                "The challenge of ensuring AI systems pursue intended goals and remain beneficial to humans",
                "hard",
                "As AI becomes more powerful, making sure it does what we want becomes crucial!",
            ),
            FactCard(
                "Future AI",
                "What is Artificial General Intelligence (AGI)?",
                "AI that can understand, learn, and apply knowledge across a wide range of tasks like humans",
                "medium",
                "Current AI is narrow (specialized), but AGI would be as versatile as human intelligence!",
            ),
        ]

        # Shuffle facts for variety
        random.shuffle(self.facts)

    def should_show_card(self, score, game_ended=False):
        """Check if we should show a fact card"""
        if self.showing_card:
            return False

        # End of run trigger
        if game_ended and self.end_run_trigger:
            return True

        # Score interval trigger
        if score >= self.last_trigger_score + self.trigger_interval:
            self.last_trigger_score = score
            return True

        return False

    def show_random_card(self):
        """Show a random fact card"""
        if not self.facts:
            return

        # Choose card based on difficulty and frequency
        available_cards = [f for f in self.facts if f.shown_count < 3]
        if not available_cards:
            available_cards = self.facts  # Reset if all seen

        # Prefer cards not shown recently or less frequently
        weights = []
        current_time = time.time()
        for fact in available_cards:
            time_weight = max(
                1, (current_time - fact.last_shown) / 3600
            )  # Hours since last shown
            frequency_weight = max(
                1, 5 - fact.shown_count
            )  # Less shown = higher weight
            weights.append(time_weight * frequency_weight)

        # Weighted random selection
        self.current_card = random.choices(available_cards, weights=weights)[0]
        self.current_card.shown_count += 1
        self.current_card.last_shown = current_time

        # Start showing animation
        self.showing_card = True
        self.card_start_time = time.time()
        self.slide_progress = 0
        self.user_dismissed = False

        # Update statistics
        self.cards_shown += 1
        self.categories_learned.add(self.current_card.category)

        print(
            f"📚 Showing fact card: {self.current_card.category} - {self.current_card.question[:50]}..."
        )

    def handle_event(self, event):
        """Handle pygame events"""
        if not self.showing_card:
            return False

        if event.type == pygame.KEYDOWN:
            if event.key in [pygame.K_SPACE, pygame.K_RETURN, pygame.K_ESCAPE]:
                self.dismiss_card()
                return True

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left click
                self.dismiss_card()
                return True

        return False

    def dismiss_card(self):
        """Dismiss the current fact card"""
        self.showing_card = False
        self.user_dismissed = True
        self.current_card = None

    def update(self):
        """Update the fact card system"""
        if not self.showing_card:
            return

        # Update slide animation
        if self.slide_progress < 1.0:
            self.slide_progress += self.slide_speed * (1 / 60)  # Assuming 60 FPS
            self.slide_progress = min(1.0, self.slide_progress)

        # Update pulse animation
        self.pulse_timer += 0.1

        # Auto-advance if enabled
        if self.auto_advance:
            elapsed = time.time() - self.card_start_time
            if elapsed >= self.card_duration and not self.user_dismissed:
                self.dismiss_card()

    def draw(self, screen):
        """Draw the fact card interface"""
        if not self.showing_card or not self.current_card:
            return

        # Calculate slide position
        slide_offset = (1 - self.slide_progress) * 100
        current_y = self.card_y - slide_offset

        # Draw semi-transparent background
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        overlay.set_alpha(int(180 * self.slide_progress))
        overlay.fill(COLORS["DARK_BLUE"])
        screen.blit(overlay, (0, 0))

        # Draw card background
        card_rect = pygame.Rect(
            self.card_x, current_y, self.card_width, self.card_height
        )

        # Card shadow
        shadow_rect = card_rect.copy()
        shadow_rect.x += 5
        shadow_rect.y += 5
        pygame.draw.rect(screen, (*COLORS["BLACK"], 100), shadow_rect, border_radius=15)

        # Main card
        pygame.draw.rect(screen, COLORS["WHITE"], card_rect, border_radius=15)

        # Category header
        category_rect = pygame.Rect(card_rect.x, card_rect.y, card_rect.width, 50)
        category_colors = {
            "ML Basics": COLORS["NEURON_BLUE"],
            "Deep Learning": COLORS["ACTIVATION"],
            "AI History": COLORS["DATASET"],
            "Modern AI": COLORS["PURPLE"],
            "Fun Facts": COLORS["ORANGE"],
            "Technical": COLORS["GRADIENT"],
            "AI Ethics": COLORS["LOSS"],
            "Future AI": COLORS["YELLOW"],
        }
        category_color = category_colors.get(self.current_card.category, COLORS["GRAY"])
        pygame.draw.rect(
            screen,
            category_color,
            category_rect,
            border_top_left_radius=15,
            border_top_right_radius=15,
        )

        # Category text
        category_font = pygame.font.Font(None, 32)
        category_text = category_font.render(
            f"📚 {self.current_card.category}", True, COLORS["WHITE"]
        )
        category_text_rect = category_text.get_rect(center=category_rect.center)
        screen.blit(category_text, category_text_rect)

        # Difficulty indicator
        difficulty_colors = {
            "easy": COLORS["GREEN"],
            "medium": COLORS["YELLOW"],
            "hard": COLORS["LOSS"],
        }
        diff_color = difficulty_colors.get(self.current_card.difficulty, COLORS["GRAY"])
        diff_circle = pygame.Rect(card_rect.right - 40, card_rect.y + 10, 20, 20)
        pygame.draw.circle(screen, diff_color, diff_circle.center, 10)

        # Question
        question_y = card_rect.y + 70
        question_font = pygame.font.Font(None, 28)
        question_lines = self.wrap_text(
            self.current_card.question, question_font, card_rect.width - 40
        )

        for i, line in enumerate(question_lines):
            question_surface = question_font.render(line, True, COLORS["DARK_BLUE"])
            screen.blit(question_surface, (card_rect.x + 20, question_y + i * 30))

        # Answer
        answer_y = question_y + len(question_lines) * 30 + 20
        answer_font = pygame.font.Font(None, 24)
        answer_lines = self.wrap_text(
            self.current_card.answer, answer_font, card_rect.width - 40
        )

        for i, line in enumerate(answer_lines):
            answer_surface = answer_font.render(line, True, COLORS["DARK_GRAY"])
            screen.blit(answer_surface, (card_rect.x + 20, answer_y + i * 25))

        # Fun fact (if available)
        if self.current_card.fun_fact:
            fun_fact_y = answer_y + len(answer_lines) * 25 + 15
            fun_fact_font = pygame.font.Font(None, 20)
            fun_fact_lines = self.wrap_text(
                f"💡 {self.current_card.fun_fact}", fun_fact_font, card_rect.width - 40
            )

            for i, line in enumerate(fun_fact_lines):
                fun_fact_surface = fun_fact_font.render(line, True, COLORS["PURPLE"])
                screen.blit(fun_fact_surface, (card_rect.x + 20, fun_fact_y + i * 22))

        # Progress bar (if auto-advance)
        if self.auto_advance:
            elapsed = time.time() - self.card_start_time
            progress = min(1.0, elapsed / self.card_duration)

            progress_rect = pygame.Rect(
                card_rect.x + 20, card_rect.bottom - 30, card_rect.width - 40, 6
            )
            pygame.draw.rect(
                screen, COLORS["LIGHT_GRAY"], progress_rect, border_radius=3
            )

            filled_width = int(progress_rect.width * progress)
            filled_rect = pygame.Rect(
                progress_rect.x, progress_rect.y, filled_width, progress_rect.height
            )
            pygame.draw.rect(screen, category_color, filled_rect, border_radius=3)

        # Dismiss instruction
        instruction_font = pygame.font.Font(None, 20)
        instruction_text = instruction_font.render(
            "Press SPACE, ENTER, or click to continue", True, COLORS["GRAY"]
        )
        instruction_rect = instruction_text.get_rect(
            center=(card_rect.centerx, card_rect.bottom - 10)
        )
        screen.blit(instruction_text, instruction_rect)

        # Pulse effect on border
        pulse_alpha = int(100 + 50 * math.sin(self.pulse_timer))
        pulse_color = (*category_color, pulse_alpha)
        pygame.draw.rect(screen, pulse_color, card_rect, 4, border_radius=15)

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
                    # Word is too long, add it anyway
                    lines.append(word)

        if current_line:
            lines.append(" ".join(current_line))

        return lines

    def get_learning_stats(self):
        """Get learning statistics"""
        return {
            "cards_shown": self.cards_shown,
            "categories_learned": len(self.categories_learned),
            "total_categories": len(set(f.category for f in self.facts)),
            "completion_percentage": (
                (
                    len(self.categories_learned)
                    / len(set(f.category for f in self.facts))
                )
                * 100
                if self.facts
                else 0
            ),
        }

    def reset_progress(self):
        """Reset learning progress"""
        for fact in self.facts:
            fact.shown_count = 0
            fact.last_shown = 0

        self.cards_shown = 0
        self.categories_learned.clear()
        self.last_trigger_score = 0


# Global fact card system instance
fact_cards = FactCardSystem()
