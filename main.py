"""
🧠 Neuron Runner - Main Game
Entry point for the neural network endless runner game.
"""

import pygame
import sys
import math
from game.config import *
from game.player import NeuronPlayer
from game.obstacles import ObstacleManager
from game.powerups import PowerupManager
from game.utils import GameStats, UI, draw_background, calculate_difficulty
from game.particles import particle_system
from game.achievements import achievement_system
from game.model_skins import model_selector
from game.challenge_modes import challenge_manager
from game.ml_stats import ml_stats
from game.model_builder import model_builder
from game.fact_cards import fact_cards
from game.training_journal import training_journal
from game.world_themes import world_themes
from game.music_system import dynamic_music_system
from game.replay_system import replay_system
from game.ai_competition import ai_competition, ai_interface
from game.educational_mode import educational_mode

# Optional sound system
try:
    from game.sound import sound_system
except ImportError:
    # Fallback if sound dependencies are missing
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

    sound_system = DummySoundSystem()


class NeuronRunnerGame:
    def __init__(self):
        pygame.init()
        pygame.mixer.init()

        # Setup display
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption(TITLE)
        self.clock = pygame.time.Clock()

        # Game state
        self.game_state = GAME_STATES["PLAYING"]  # Start playing directly
        self.running = True
        self.game_time = 0  # Track total game time

        # Initialize game components
        self.player = NeuronPlayer()
        self.obstacle_manager = ObstacleManager()
        self.powerup_manager = PowerupManager()
        self.stats = GameStats()
        self.ui = UI()

        # Model selection is handled by default model
        self.show_model_selection = False

        # Visual effects
        self.background_scroll = 0
        self.screen_shake = 0
        self.damage_flash = 0  # Screen flash when taking damage
        self.warning_system = []  # Warning indicators for obstacles
        self.difficulty_spike_notification = 0  # Spike notification timer
        self.pulse_timer = 0  # For theme animations

        # Initialize world themes and music
        world_themes.initialize()
        dynamic_music_system.initialize()

        # Initialize replay system
        replay_system.start_recording()

        # AI competition state
        self.ai_mode = False
        self.ai_bot_name = None
        self.ai_last_state = None

        print("🧠 Neuron Runner initialized!")
        print(
            "🎯 Controls: SPACE=Jump, DOWN=Duck, L=Layer Boost, B=Model Builder, J=Journal"
        )
        print("📹 R=Toggle Recording, A=AI Mode, G=Generate GIF")
        print("🎮 Starting training session...")

        # Start initial training session
        current_model = model_selector.get_selected_model()
        current_challenge = challenge_manager.get_current_challenge()
        training_journal.start_session(current_model, current_challenge)

    def handle_events(self):
        """Handle all pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            # Handle model builder events
            if model_builder.handle_event(event):
                continue

            # Handle fact card events
            if fact_cards.handle_event(event):
                continue

            # Handle training journal events
            if training_journal.handle_event(event):
                continue

            elif event.type == pygame.KEYDOWN:
                if event.key == KEYS["PAUSE"]:
                    self.toggle_pause()
                elif (
                    event.key == KEYS["RESTART"]
                    and self.game_state == GAME_STATES["GAME_OVER"]
                ):
                    self.restart_game()
                elif event.key == pygame.K_b:  # B key to open model builder
                    model_builder.show()
                elif event.key == pygame.K_j:  # J key to open training journal
                    training_journal.show_journal()
                elif event.key == pygame.K_r:  # R key to toggle recording
                    if replay_system.toggle_recording():
                        print("🔴 Recording started")
                    else:
                        print("⏹️ Recording stopped")
                elif event.key == pygame.K_g:  # G key to generate GIF
                    if self.game_state == GAME_STATES["GAME_OVER"]:
                        current_model = model_selector.get_selected_model()
                        current_challenge = challenge_manager.get_current_challenge()
                        challenge_name = (
                            current_challenge.name if current_challenge else None
                        )
                        replay_system.generate_replay_gif(
                            self.stats,
                            current_model.name if current_model else "Default",
                            challenge_name,
                        )
                elif event.key == pygame.K_a:  # A key to toggle AI mode
                    self.toggle_ai_mode()
                elif event.key == pygame.K_t:  # T key for AI tournament
                    if ai_competition:
                        available_bots = list(ai_competition.bots.keys())
                        if len(available_bots) >= 2:
                            tournament = ai_competition.create_bot_tournament(
                                f"Tournament_{len(ai_competition.tournament_manager.tournaments) + 1}",
                                available_bots[:4],  # Max 4 bots
                            )
                            print("🏆 Tournament created!")
                        else:
                            print("❌ Need at least 2 AI bots for tournament")
                elif event.key == pygame.K_c:  # C key for human vs AI challenge
                    if ai_competition and ai_competition.bots:
                        bot_name = list(ai_competition.bots.keys())[0]
                        ai_competition.human_vs_ai_challenge(bot_name)
                elif event.key == pygame.K_e:  # E key to toggle educational mode
                    educational_mode.toggle()
                    print(
                        f"🎓 Educational mode: {'ON' if educational_mode.is_active else 'OFF'}"
                    )
                elif event.key == pygame.K_m:  # M key to toggle music visualization
                    dynamic_music_system.toggle_visualization()
                elif event.key == pygame.K_v:  # V key to cycle ML stats view
                    ml_stats.cycle_dashboard_mode()
                elif event.key == pygame.K_F1:  # F1 for help
                    self.show_help_overlay = not getattr(
                        self, "show_help_overlay", False
                    )

        # Handle continuous key presses
        keys = pygame.key.get_pressed()
        if self.game_state == GAME_STATES["PLAYING"]:
            if not self.ai_mode:
                self.player.handle_input(keys)
            else:
                self.handle_ai_input()

    def update_game(self):
        """Update all game logic"""
        if self.game_state != GAME_STATES["PLAYING"]:
            return

        # Update game time
        self.game_time += 1 / FPS  # Increment by frame time

        # Update player
        self.player.update()

        # Calculate difficulty based on score
        difficulty = calculate_difficulty(self.stats.score)

        # Check for difficulty spike notifications
        if self.stats.score > 0 and self.stats.score % DIFFICULTY_SPIKE_INTERVAL == 0:
            if self.difficulty_spike_notification <= 0:
                self.difficulty_spike_notification = 3 * FPS  # 3 second notification

        if self.difficulty_spike_notification > 0:
            self.difficulty_spike_notification -= 1

        # Update managers with speed information
        self.obstacle_manager.update(difficulty, self.stats.score)
        self.powerup_manager.update(self.obstacle_manager.current_speed_multiplier)

        # Update particle system
        particle_system.update()

        # Check collisions
        if self.obstacle_manager.check_collisions(self.player):
            self.screen_shake = 15  # Increased screen shake
            self.damage_flash = 20  # Add damage flash effect
            # Create damage particles
            particle_system.create_explosion(
                self.player.rect.centerx, self.player.rect.centery, COLORS["LOSS"], 15
            )
            # Play damage sound
            sound_system.play_damage()
        else:
            # Track obstacles dodged (approximately)
            for obstacle in self.obstacle_manager.obstacles:
                if obstacle.rect.right < self.player.rect.left and not getattr(
                    obstacle, "counted", False
                ):
                    self.stats.obstacles_dodged += 1
                    obstacle.counted = True  # Mark as counted

        # Check powerup collections
        collected_powerup = self.powerup_manager.check_collections(self.player)
        if collected_powerup:
            # Track powerup collection
            self.stats.powerups_collected += 1
            # Create collection particles
            particle_system.create_powerup_collection(
                collected_powerup["x"],
                collected_powerup["y"],
                collected_powerup["color"],
            )
            # Play powerup sound
            sound_system.play_powerup()

        # Update warning system
        self.update_warning_system()

        # Update statistics
        player_status = self.player.get_status()
        self.stats.update(player_status)

        # Update achievement system
        achievement_system.update(self.stats, player_status)

        # Update challenge manager
        challenge_manager.update(self.stats.score, self.game_time)

        # Update ML stats
        ml_stats.update(self.stats, player_status)

        # Update model builder
        model_builder.update()

        # Update fact cards
        fact_cards.update()

        # Update training journal
        training_journal.update()

        # Update world themes based on score
        dt = self.clock.get_time() / 1000.0  # Convert to seconds
        world_themes.update(self.stats.score, dt)

        # Update music system with game state
        current_theme = world_themes.get_current_theme()
        game_intensity = self.calculate_game_intensity()
        dynamic_music_system.update(game_intensity, current_theme)

        # Update pulse timer for theme animations
        self.pulse_timer += 1

        # Capture frame for replay system
        replay_system.capture_frame(self.screen, self.stats, self.player.rect.center)

        # Update training journal session with current stats
        if training_journal.current_session:
            training_journal.update_session(
                self.stats.score,
                self.stats.distance_traveled,
                self.stats.obstacles_dodged,
                self.stats.powerups_collected,
                self.player.special_abilities_used,
            )

        # Check if we should show a fact card
        if fact_cards.should_show_card(self.stats.score):
            fact_cards.show_random_card()

        # Check game over condition
        if self.player.activation_level <= 0:
            self.game_state = GAME_STATES["GAME_OVER"]
            # End training journal session
            training_journal.end_session("collision")
            # Show fact card at end of run
            if fact_cards.should_show_card(self.stats.score, game_ended=True):
                fact_cards.show_random_card()
            print(f"💀 Training failed! Final progress: {self.stats.score}")

        # Update visual effects
        self.background_scroll += 2
        if self.screen_shake > 0:
            self.screen_shake -= 1
        if self.damage_flash > 0:
            self.damage_flash -= 1

    def update_warning_system(self):
        """Update obstacle warning indicators"""
        self.warning_system.clear()

        # Check for obstacles that are close to the player
        for obstacle in self.obstacle_manager.obstacles:
            distance_to_player = obstacle.rect.x - self.player.rect.right

            # Show warning if obstacle is approaching
            if 200 <= distance_to_player <= 400:
                warning_intensity = 1.0 - (distance_to_player - 200) / 200
                self.warning_system.append(
                    {
                        "y": obstacle.rect.centery,
                        "type": obstacle.type,
                        "intensity": warning_intensity,
                    }
                )

    def calculate_game_intensity(self):
        """Calculate current game intensity for music system"""
        base_intensity = min(self.stats.score / 10000, 1.0)  # 0-1 based on score
        speed_intensity = min(self.obstacle_manager.current_speed_multiplier / 3.0, 1.0)
        challenge_intensity = 0.2 if challenge_manager.get_current_challenge() else 0.0
        player_intensity = 0.3 if self.player.layer_boost_timer > 0 else 0.0

        return min(
            base_intensity + speed_intensity + challenge_intensity + player_intensity,
            1.0,
        )

    def toggle_ai_mode(self):
        """Toggle AI mode on/off"""
        if not ai_competition:
            print("❌ AI competition system not available")
            return

        self.ai_mode = not self.ai_mode
        if self.ai_mode:
            # Start AI training if we have a bot
            if ai_competition.bots:
                bot_name = list(ai_competition.bots.keys())[
                    0
                ]  # Use first available bot
                self.ai_bot_name = bot_name
                ai_competition.start_training(bot_name)
                print(f"🤖 AI Mode activated - Bot: {bot_name}")
            else:
                # Create a new bot
                ai_competition.create_new_bot("AutoRunner")
                self.ai_bot_name = "AutoRunner"
                ai_competition.start_training("AutoRunner")
                print("🤖 AI Mode activated - New bot created: AutoRunner")
        else:
            if self.ai_bot_name:
                ai_competition.stop_training(self.ai_bot_name)
            print("👤 Human mode activated")

    def handle_ai_input(self):
        """Handle AI decision making"""
        if not ai_competition or not ai_interface or not self.ai_bot_name:
            return

        # Get current game state
        current_state = ai_interface.get_current_state(
            self.player, self.obstacle_manager, self.powerup_manager, self.stats
        )

        # Get AI action
        action = ai_competition.get_bot_action(current_state)

        # Execute action
        ai_interface.execute_action(action, self.player)

        # Store state for training
        if self.ai_last_state is not None:
            # Calculate reward
            collision_occurred = self.player.activation_level <= 0
            powerup_collected = False  # This would be tracked from powerup collection
            reward = ai_interface.reward_calculator.calculate_reward(
                self.stats, self.player, collision_occurred, powerup_collected
            )

            # Update bot training
            ai_competition.update_bot_training(
                self.ai_last_state, action, reward, current_state, collision_occurred
            )

        self.ai_last_state = current_state

    def render(self):
        """Render all game graphics"""
        # Apply screen shake
        shake_offset_x = 0
        shake_offset_y = 0
        if self.screen_shake > 0:
            import random

            shake_offset_x = random.randint(-self.screen_shake, self.screen_shake)
            shake_offset_y = random.randint(-self.screen_shake, self.screen_shake)

        # Draw background with world theme
        player_status = self.player.get_status()
        draw_background(self.screen, self.background_scroll, player_status)

        # Apply world theme effects
        current_theme = world_themes.get_current_theme()
        if current_theme:
            current_theme.apply_background_effects(
                self.screen, self.background_scroll, self.pulse_timer
            )

        # Create a surface for shaking effect
        if self.screen_shake > 0:
            game_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
            player_status = self.player.get_status()
            draw_background(game_surface, self.background_scroll, player_status)

            # Apply theme effects to shake surface too
            if current_theme:
                current_theme.apply_background_effects(
                    game_surface, self.background_scroll, self.pulse_timer
                )

            draw_surface = game_surface
        else:
            draw_surface = self.screen

        # Draw game objects
        if self.game_state == GAME_STATES["PLAYING"]:
            # Draw warning indicators first
            self.draw_warnings(draw_surface)

            # Draw player with proper positioning
            draw_surface.blit(self.player.image, self.player.rect)

            # Draw obstacles and powerups
            self.obstacle_manager.draw(draw_surface)
            self.powerup_manager.draw(draw_surface)

            # Draw particle effects
            particle_system.draw(draw_surface)

            # Draw achievements
            achievement_system.draw(draw_surface)

            # Draw UI
            player_status = self.player.get_status()
            self.ui.draw_hud(draw_surface, self.stats, player_status)

            # Draw excitement indicators (combos, adrenaline, close calls)
            self.ui.draw_excitement_indicators(draw_surface, player_status)

            # Draw difficulty spike notification
            if self.difficulty_spike_notification > 0:
                self.draw_difficulty_spike_notification(draw_surface)

            # Draw damage flash overlay
            if self.damage_flash > 0:
                flash_alpha = int((self.damage_flash / 20) * 100)
                flash_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
                flash_surface.set_alpha(flash_alpha)
                flash_surface.fill(COLORS["LOSS"])
                draw_surface.blit(flash_surface, (0, 0))

        elif self.game_state == GAME_STATES["GAME_OVER"]:
            # Still draw game elements but with overlay
            draw_surface.blit(self.player.image, self.player.rect)
            self.obstacle_manager.draw(draw_surface)
            self.powerup_manager.draw(draw_surface)

            # Draw game over screen
            self.ui.draw_game_over(draw_surface, self.stats)

        elif self.game_state == GAME_STATES["PAUSED"]:
            # Draw everything but add pause overlay
            draw_surface.blit(self.player.image, self.player.rect)
            self.obstacle_manager.draw(draw_surface)
            self.powerup_manager.draw(draw_surface)

            player_status = self.player.get_status()
            self.ui.draw_hud(draw_surface, self.stats, player_status)

            # Pause overlay
            pause_overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
            pause_overlay.set_alpha(128)
            pause_overlay.fill(COLORS["BLACK"])
            draw_surface.blit(pause_overlay, (0, 0))

            # Pause text
            font = pygame.font.Font(None, 72)
            pause_text = font.render("TRAINING PAUSED", True, COLORS["WHITE"])
            pause_rect = pause_text.get_rect(
                center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
            )
            draw_surface.blit(pause_text, pause_rect)

            resume_text = pygame.font.Font(None, 36).render(
                "Press P to Resume", True, COLORS["ACTIVATION"]
            )
            resume_rect = resume_text.get_rect(
                center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 60)
            )
            draw_surface.blit(resume_text, resume_rect)

        # Apply screen shake
        if self.screen_shake > 0:
            self.screen.fill(COLORS["BACKGROUND"])
            self.screen.blit(game_surface, (shake_offset_x, shake_offset_y))

        # Draw overlay systems (always on top)
        ml_stats.draw(self.screen)
        model_builder.draw(self.screen)
        fact_cards.draw(self.screen)
        training_journal.draw(self.screen)
        training_journal.draw_notifications(self.screen)

        # Update display
        pygame.display.flip()

    def draw_warnings(self, surface):
        """Draw warning indicators for approaching obstacles"""
        for warning in self.warning_system:
            # Draw warning indicator on the right side of screen
            warning_x = SCREEN_WIDTH - 30
            warning_y = warning["y"]
            warning_alpha = int(warning["intensity"] * 255)

            # Choose warning color based on obstacle type
            if warning["type"] == "DEAD_NEURON":
                warning_color = (*COLORS["LOSS"], warning_alpha)
            elif warning["type"] == "VANISHING_GRADIENT":
                warning_color = (*COLORS["GRADIENT"], warning_alpha)
            else:
                warning_color = (*COLORS["DATASET"], warning_alpha)

            # Draw warning indicator
            warning_surface = pygame.Surface((20, 40), pygame.SRCALPHA)
            pygame.draw.polygon(
                warning_surface, warning_color, [(0, 20), (15, 10), (15, 30)]
            )
            pygame.draw.polygon(
                warning_surface, COLORS["WHITE"], [(0, 20), (15, 10), (15, 30)], 2
            )

            surface.blit(warning_surface, (warning_x, warning_y - 20))

    def draw_difficulty_spike_notification(self, surface):
        """Draw difficulty spike notification"""
        # Pulsing notification
        pulse = math.sin(pygame.time.get_ticks() * 0.01) * 0.3 + 0.7
        alpha = int(255 * pulse)

        # Large notification text
        font = pygame.font.Font(None, 64)
        spike_text = font.render("🔥 DIFFICULTY SPIKE! 🔥", True, COLORS["LOSS"])

        # Background
        bg_width = spike_text.get_width() + 40
        bg_height = spike_text.get_height() + 20
        bg_x = (SCREEN_WIDTH - bg_width) // 2
        bg_y = 100

        spike_bg = pygame.Surface((bg_width, bg_height))
        spike_bg.set_alpha(alpha)
        spike_bg.fill(COLORS["LOSS"])
        surface.blit(spike_bg, (bg_x, bg_y))

        # Border
        pygame.draw.rect(surface, COLORS["WHITE"], (bg_x, bg_y, bg_width, bg_height), 3)

        # Text
        text_x = bg_x + 20
        text_y = bg_y + 10
        surface.blit(spike_text, (text_x, text_y))

        # Sub-text
        sub_font = pygame.font.Font(None, 32)
        sub_text = sub_font.render("INCOMING CHALLENGE!", True, COLORS["WHITE"])
        sub_x = (SCREEN_WIDTH - sub_text.get_width()) // 2
        sub_y = bg_y + bg_height + 10
        surface.blit(sub_text, (sub_x, sub_y))

    def toggle_pause(self):
        """Toggle game pause state"""
        if self.game_state == GAME_STATES["PLAYING"]:
            self.game_state = GAME_STATES["PAUSED"]
        elif self.game_state == GAME_STATES["PAUSED"]:
            self.game_state = GAME_STATES["PLAYING"]

    def restart_game(self):
        """Restart the game"""
        self.player.reset()
        self.obstacle_manager.clear()
        self.powerup_manager.clear()
        self.stats.reset()
        self.game_state = GAME_STATES["PLAYING"]
        self.background_scroll = 0
        self.screen_shake = 0
        self.damage_flash = 0
        self.warning_system.clear()
        self.difficulty_spike_notification = 0
        particle_system.clear()

        # Start new training journal session
        current_model = model_selector.get_selected_model()
        current_challenge = challenge_manager.get_current_challenge()
        training_journal.start_session(current_model, current_challenge)

        print("🔄 Training session restarted!")

    def run(self):
        """Main game loop"""
        print("🚀 Starting Neuron Runner...")

        while self.running:
            # Handle events
            self.handle_events()

            # Update game logic
            self.update_game()

            # Render graphics
            self.render()

            # Control frame rate
            self.clock.tick(FPS)

        # Cleanup
        pygame.quit()
        print("👋 Thanks for training with Neuron Runner!")


def main():
    """Entry point for the game"""
    try:
        game = NeuronRunnerGame()
        game.run()
    except Exception as e:
        print(f"❌ Error running Neuron Runner: {e}")
        pygame.quit()
        sys.exit(1)


if __name__ == "__main__":
    main()
