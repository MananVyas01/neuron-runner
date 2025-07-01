"""
🤖 AI Agent Competition Mode - INDUSTRIAL LEVEL Train Your Bot
Advanced Q-learning bot training with professional tournament system and comprehensive analytics
"""

import pygame
import numpy as np
import json
import os
import random
import math
import time
from collections import deque, defaultdict
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from .config import *

@dataclass
class BotPerformanceStats:
    """Comprehensive bot performance statistics"""
    games_played: int = 0
    total_score: int = 0
    best_score: int = 0
    average_score: float = 0.0
    total_distance: float = 0.0
    obstacles_dodged: int = 0
    powerups_collected: int = 0
    successful_jumps: int = 0
    failed_jumps: int = 0
    special_abilities_used: int = 0
    survival_time: float = 0.0
    consistency_rating: float = 0.0
    learning_progress: float = 0.0
    win_rate: float = 0.0
    
    def update_from_game(self, game_result: Dict):
        """Update stats from a completed game"""
        self.games_played += 1
        score = game_result.get('score', 0)
        self.total_score += score
        self.best_score = max(self.best_score, score)
        self.average_score = self.total_score / self.games_played
        
        self.total_distance += game_result.get('distance', 0)
        self.obstacles_dodged += game_result.get('obstacles_dodged', 0)
        self.powerups_collected += game_result.get('powerups_collected', 0)
        self.survival_time += game_result.get('survival_time', 0)
        
        # Calculate consistency (lower std deviation = higher consistency)
        recent_scores = game_result.get('recent_scores', [score])
        if len(recent_scores) > 1:
            self.consistency_rating = max(0, 100 - np.std(recent_scores))

class GameState:
    """Enhanced game state representation for AI"""
    
    def __init__(self, player_pos, obstacles, powerups, speed, score, additional_context=None):
        self.player_y = player_pos[1] / SCREEN_HEIGHT  # Normalized
        self.player_velocity_y = player_pos[2] if len(player_pos) > 2 else 0
        self.player_grounded = player_pos[3] if len(player_pos) > 3 else True
        self.player_health = player_pos[4] if len(player_pos) > 4 else 1.0
        
        # Enhanced obstacle processing
        self.obstacles = self._process_obstacles(obstacles)
        self.powerups = self._process_powerups(powerups)
        
        # Game context
        self.speed = min(speed / 5.0, 2.0)  # Normalized speed
        self.score = min(score / 10000.0, 1.0)  # Normalized score
        
        # Additional context
        self.time_since_last_powerup = additional_context.get('time_since_last_powerup', 0) if additional_context else 0
        self.combo_multiplier = additional_context.get('combo_multiplier', 1.0) if additional_context else 1.0
        self.difficulty_level = additional_context.get('difficulty_level', 1.0) if additional_context else 1.0
        
    def _process_obstacles(self, obstacles):
        """Process and normalize obstacle data"""
        processed = []
        for i, obs in enumerate(obstacles[:8]):  # Consider up to 8 obstacles
            if i < len(obstacles):
                processed.extend([
                    obs['x'] / SCREEN_WIDTH,  # Normalized X distance
                    obs['y'] / SCREEN_HEIGHT,  # Normalized Y position
                    obs.get('width', 50) / SCREEN_WIDTH,  # Normalized width
                    obs.get('height', 50) / SCREEN_HEIGHT,  # Normalized height
                    obs.get('type', 0) / 3.0,  # Normalized obstacle type
                    obs.get('speed', 1.0),  # Relative speed
                ])
            else:
                processed.extend([1.0, 0.5, 0.05, 0.05, 0.0, 1.0])  # Default values
        return processed
    
    def _process_powerups(self, powerups):
        """Process and normalize powerup data"""
        processed = []
        for i, pup in enumerate(powerups[:4]):  # Consider up to 4 powerups
            if i < len(powerups):
                processed.extend([
                    pup['x'] / SCREEN_WIDTH,  # Normalized X distance
                    pup['y'] / SCREEN_HEIGHT,  # Normalized Y position
                    pup.get('type', 0) / 3.0,  # Normalized powerup type
                    pup.get('value', 1.0),  # Powerup value
                ])
            else:
                processed.extend([1.0, 0.5, 0.0, 0.0])  # Default values
        return processed
    
    def to_vector(self):
        """Convert state to comprehensive feature vector"""
        features = []
        
        # Player state (5 features)
        features.extend([
            self.player_y,
            self.player_velocity_y / 20.0,  # Normalized velocity
            1.0 if self.player_grounded else 0.0,
            self.player_health,
            self.combo_multiplier / 5.0  # Normalized combo
        ])
        
        # Obstacle features (48 features: 8 obstacles × 6 features each)
        features.extend(self.obstacles)
        
        # Powerup features (16 features: 4 powerups × 4 features each)
        features.extend(self.powerups)
        
        # Game context features (6 features)
        features.extend([
            self.speed,
            self.score,
            self.time_since_last_powerup / 1000.0,  # Normalized time
            self.difficulty_level / 3.0,  # Normalized difficulty
            math.sin(time.time()),  # Temporal pattern
            math.cos(time.time())   # Temporal pattern
        ])
        
        return np.array(features, dtype=np.float32)

class AdvancedQLearningAgent:
    """Professional Q-learning agent with advanced features"""
    
    def __init__(self, state_size=75, action_size=6, learning_rate=0.001, architecture='dqn'):
        self.state_size = state_size
        self.action_size = action_size  # 0=nothing, 1=jump, 2=duck, 3=boost, 4=special, 5=combo
        self.learning_rate = learning_rate
        self.architecture = architecture
        
        # Advanced hyperparameters
        self.discount_factor = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.tau = 0.001  # For soft target updates
        
        # Neural network approximation (using linear for simplicity)
        self.q_network = self._build_network()
        self.target_network = self._build_network()
        self.update_target_network()
        
        # Experience replay with prioritization
        self.memory = deque(maxlen=50000)
        self.priority_memory = deque(maxlen=10000)
        self.batch_size = 64
        self.min_memory_size = 1000
        
        # Advanced training features
        self.double_dqn = True
        self.dueling_dqn = True
        self.noisy_nets = False
        
        # Performance tracking
        self.stats = BotPerformanceStats()
        self.training_history = {
            'episode_rewards': deque(maxlen=1000),
            'episode_lengths': deque(maxlen=1000),
            'loss_history': deque(maxlen=1000),
            'epsilon_history': deque(maxlen=1000),
        }
        
        # Meta-learning features
        self.adaptation_rate = 0.1
        self.strategy_weights = np.ones(action_size) / action_size
        self.action_success_rates = defaultdict(list)
        
        print(f"🤖 Advanced Q-Learning Agent initialized with {architecture} architecture")
    
    def _build_network(self):
        """Build neural network for Q-value approximation"""
        if self.architecture == 'linear':
            # Simple linear model
            weights = np.random.normal(0, 0.1, (self.state_size, self.action_size))
            bias = np.zeros(self.action_size)
            return {'weights': weights, 'bias': bias, 'type': 'linear'}
        
        elif self.architecture == 'dqn':
            # Deep network simulation (simplified for this implementation)
            hidden1_size = 256
            hidden2_size = 128
            
            weights1 = np.random.normal(0, 0.1, (self.state_size, hidden1_size))
            bias1 = np.zeros(hidden1_size)
            weights2 = np.random.normal(0, 0.1, (hidden1_size, hidden2_size))
            bias2 = np.zeros(hidden2_size)
            weights3 = np.random.normal(0, 0.1, (hidden2_size, self.action_size))
            bias3 = np.zeros(self.action_size)
            
            return {
                'weights1': weights1, 'bias1': bias1,
                'weights2': weights2, 'bias2': bias2,
                'weights3': weights3, 'bias3': bias3,
                'type': 'dqn'
            }
    
    def get_q_values(self, state, network=None):
        """Get Q-values from network"""
        if network is None:
            network = self.q_network
        
        if network['type'] == 'linear':
            return np.dot(state, network['weights']) + network['bias']
        
        elif network['type'] == 'dqn':
            # Forward pass through network
            h1 = np.maximum(0, np.dot(state, network['weights1']) + network['bias1'])  # ReLU
            h2 = np.maximum(0, np.dot(h1, network['weights2']) + network['bias2'])  # ReLU
            output = np.dot(h2, network['weights3']) + network['bias3']
            return output
    
    def act(self, state, training=True):
        """Advanced action selection with multiple strategies"""
        if training and np.random.random() <= self.epsilon:
            # Epsilon-greedy with strategy bias
            if np.random.random() < 0.3:  # 30% truly random
                return np.random.randint(0, self.action_size)
            else:  # 70% biased by success rates
                weights = np.array([np.mean(self.action_success_rates.get(i, [0.5])) 
                                  for i in range(self.action_size)])
                weights = weights / np.sum(weights)
                return np.random.choice(self.action_size, p=weights)
        
        # Q-value based selection
        q_values = self.get_q_values(state)
        
        # Add noise for exploration during training
        if training and np.random.random() < 0.1:
            noise = np.random.normal(0, 0.1, q_values.shape)
            q_values += noise
        
        return np.argmax(q_values)
    
    def remember(self, state, action, reward, next_state, done, priority=1.0):
        """Store experience with priority"""
        experience = (state, action, reward, next_state, done, priority)
        
        if priority > 1.5:  # High priority experiences
            self.priority_memory.append(experience)
        
        self.memory.append(experience)
        
        # Update action success rates
        success_rate = max(0, reward) / 100.0  # Normalize reward to success rate
        self.action_success_rates[action].append(success_rate)
        if len(self.action_success_rates[action]) > 50:
            self.action_success_rates[action].pop(0)
    
    def replay(self):
        """Advanced experience replay with prioritization"""
        if len(self.memory) < self.min_memory_size:
            return 0.0
        
        # Sample experiences (mix of random and prioritized)
        batch_size = min(self.batch_size, len(self.memory))
        
        # 70% from regular memory, 30% from priority memory
        regular_size = int(batch_size * 0.7)
        priority_size = batch_size - regular_size
        
        regular_batch = random.sample(list(self.memory), regular_size)
        priority_batch = random.sample(list(self.priority_memory), 
                                     min(priority_size, len(self.priority_memory)))
        
        batch = regular_batch + priority_batch
        
        # Process batch
        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])
        
        # Calculate targets
        if self.double_dqn:
            # Double DQN
            next_actions = np.argmax([self.get_q_values(s) for s in next_states], axis=1)
            next_q_values = np.array([self.get_q_values(s, self.target_network)[a] 
                                    for s, a in zip(next_states, next_actions)])
        else:
            # Standard DQN
            next_q_values = np.max([self.get_q_values(s, self.target_network) 
                                  for s in next_states], axis=1)
        
        targets = rewards + (self.discount_factor * next_q_values * ~dones)
        
        # Update network (simplified gradient descent)
        total_loss = 0.0
        for i, (state, action, target) in enumerate(zip(states, actions, targets)):
            current_q = self.get_q_values(state)
            td_error = target - current_q[action]
            
            # Update weights (simplified backpropagation)
            if self.q_network['type'] == 'linear':
                grad_w = np.outer(state, np.zeros_like(current_q))
                grad_w[:, action] = state * td_error
                grad_b = np.zeros_like(current_q)
                grad_b[action] = td_error
                
                self.q_network['weights'] += self.learning_rate * grad_w
                self.q_network['bias'] += self.learning_rate * grad_b
            
            total_loss += td_error ** 2
        
        # Soft update target network
        self.update_target_network()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        avg_loss = total_loss / len(batch)
        self.training_history['loss_history'].append(avg_loss)
        self.training_history['epsilon_history'].append(self.epsilon)
        
        return avg_loss
    
    def update_target_network(self):
        """Soft update of target network"""
        if self.q_network['type'] == 'linear':
            self.target_network['weights'] = (
                self.tau * self.q_network['weights'] + 
                (1 - self.tau) * self.target_network['weights']
            )
            self.target_network['bias'] = (
                self.tau * self.q_network['bias'] + 
                (1 - self.tau) * self.target_network['bias']
            )
    
    def save_model(self, filepath):
        """Save comprehensive model data"""
        model_data = {
            'q_network': {k: v.tolist() if isinstance(v, np.ndarray) else v 
                         for k, v in self.q_network.items()},
            'target_network': {k: v.tolist() if isinstance(v, np.ndarray) else v 
                              for k, v in self.target_network.items()},
            'hyperparameters': {
                'learning_rate': self.learning_rate,
                'epsilon': self.epsilon,
                'discount_factor': self.discount_factor,
                'architecture': self.architecture
            },
            'stats': asdict(self.stats),
            'training_history': {k: list(v) for k, v in self.training_history.items()},
            'action_success_rates': {str(k): list(v) for k, v in self.action_success_rates.items()},
            'meta_info': {
                'created_time': time.time(),
                'version': '2.0',
                'total_training_steps': sum(self.training_history['episode_lengths'])
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2)
    
    def load_model(self, filepath):
        """Load comprehensive model data"""
        try:
            with open(filepath, 'r') as f:
                model_data = json.load(f)
            
            # Load networks
            self.q_network = {k: np.array(v) if isinstance(v, list) else v 
                             for k, v in model_data['q_network'].items()}
            self.target_network = {k: np.array(v) if isinstance(v, list) else v 
                                  for k, v in model_data['target_network'].items()}
            
            # Load hyperparameters
            hyperparams = model_data.get('hyperparameters', {})
            self.learning_rate = hyperparams.get('learning_rate', self.learning_rate)
            self.epsilon = hyperparams.get('epsilon', self.epsilon_min)
            self.discount_factor = hyperparams.get('discount_factor', self.discount_factor)
            
            # Load stats
            if 'stats' in model_data:
                self.stats = BotPerformanceStats(**model_data['stats'])
            
            # Load training history
            if 'training_history' in model_data:
                for key, values in model_data['training_history'].items():
                    self.training_history[key] = deque(values, maxlen=1000)
            
            # Load action success rates
            if 'action_success_rates' in model_data:
                self.action_success_rates = {
                    int(k): deque(v, maxlen=50) for k, v in model_data['action_success_rates'].items()
                }
            
            print(f"✅ Advanced model loaded successfully from {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False

try:
    import pickle
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    print("⚠️ AI competition system dependencies not fully available.")

class TournamentManager:
    """Advanced tournament management system"""
    
    def __init__(self):
        self.tournaments = {}
        self.active_tournament = None
        self.tournament_history = []
        
    def create_tournament(self, name, participants, tournament_type='round_robin'):
        """Create a new tournament"""
        tournament = {
            'name': name,
            'type': tournament_type,
            'participants': participants,
            'matches': [],
            'results': {},
            'status': 'created',
            'created_time': time.time()
        }
        
        self.tournaments[name] = tournament
        return tournament
    
    def run_tournament(self, tournament_name):
        """Run a complete tournament"""
        if tournament_name not in self.tournaments:
            return None
        
        tournament = self.tournaments[tournament_name]
        print(f"🏆 Starting tournament: {tournament['name']}")
        
        # Generate matches based on tournament type
        if tournament['type'] == 'round_robin':
            self.generate_round_robin_matches(tournament)
        elif tournament['type'] == 'elimination':
            self.generate_elimination_matches(tournament)
        
        return tournament

class AICompetition:
    """INDUSTRIAL LEVEL AI bot competitions and leaderboards management"""
    
    def __init__(self):
        self.bots = {}  # Dictionary of trained bots
        self.leaderboard = []
        self.current_bot = None
        self.training_mode = False
        self.competition_results = []
        self.tournament_manager = TournamentManager()
        
        # Advanced analytics
        self.performance_analytics = {
            'training_curves': {},
            'convergence_analysis': {},
            'strategy_analysis': {},
            'meta_learning_data': {}
        }
        
        # Create AI models directory
        self.models_dir = "ai_models"
        self.analytics_dir = "ai_analytics"
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.analytics_dir, exist_ok=True)
        
        # Load existing models
        self.load_all_models()
        
        print("🤖 INDUSTRIAL AI Competition System initialized!")
        print(f"   📁 Models directory: {self.models_dir}")
        print(f"   📊 Analytics directory: {self.analytics_dir}")
    
    def create_new_bot(self, name, architecture='dqn', hyperparams=None):
        """Create a new AI bot with specified architecture"""
        if name in self.bots:
            print(f"Bot '{name}' already exists!")
            return False
        
        # Use the advanced Q-learning agent
        self.bots[name] = AdvancedQLearningAgent(
            architecture=architecture,
            **(hyperparams or {})
        )
        print(f"🤖 Created new {architecture.upper()} bot: {name}")
        return True
    
    def start_training(self, bot_name):
        """Start training a specific bot"""
        if bot_name not in self.bots:
            print(f"Bot '{bot_name}' not found!")
            return False
        
        self.current_bot = self.bots[bot_name]
        self.training_mode = True
        print(f"🎓 Started training bot: {bot_name}")
        print(f"   🧠 Architecture: {self.current_bot.architecture}")
        print(f"   📈 Epsilon: {self.current_bot.epsilon:.3f}")
        print(f"   🎯 Games played: {self.current_bot.stats.games_played}")
        return True
    
    def stop_training(self, bot_name):
        """Stop training and save the bot"""
        if bot_name not in self.bots:
            return False
        
        self.training_mode = False
        model_path = os.path.join(self.models_dir, f"{bot_name}.json")
        self.bots[bot_name].save_model(model_path)
        
        # Save analytics
        self.save_bot_analytics(bot_name)
        
        print(f"💾 Saved bot: {bot_name}")
        print(f"   📊 Performance: {self.bots[bot_name].stats.average_score:.1f} avg score")
        print(f"   🎯 Best score: {self.bots[bot_name].stats.best_score}")
        return True
    
    def save_bot_analytics(self, bot_name):
        """Save comprehensive bot analytics"""
        if bot_name not in self.bots:
            return
        
        bot = self.bots[bot_name]
        analytics = {
            'training_curve': list(bot.training_history['episode_rewards']),
            'loss_curve': list(bot.training_history['loss_history']),
            'epsilon_decay': list(bot.training_history['epsilon_history']),
            'episode_lengths': list(bot.training_history['episode_lengths']),
            'action_preferences': {str(k): np.mean(v) for k, v in bot.action_success_rates.items()},
            'performance_stats': asdict(bot.stats),
            'hyperparameters': {
                'learning_rate': bot.learning_rate,
                'architecture': bot.architecture,
                'epsilon_min': bot.epsilon_min,
                'discount_factor': bot.discount_factor
            }
        }
        
        analytics_path = os.path.join(self.analytics_dir, f"{bot_name}_analytics.json")
        with open(analytics_path, 'w') as f:
            json.dump(analytics, f, indent=2)
    
    def get_bot_action(self, game_state):
        """Get action from current bot"""
        if not self.current_bot or not self.training_mode:
            return 0  # No action
        
        state_vector = game_state.to_vector()
        return self.current_bot.act(state_vector, training=True)
    
    def update_bot_training(self, old_state, action, reward, new_state, done):
        """Update bot training with experience"""
        if not self.current_bot or not self.training_mode:
            return
        
        old_vector = old_state.to_vector()
        new_vector = new_state.to_vector()
        
        # Calculate priority based on reward magnitude
        priority = 1.0 + abs(reward) / 100.0
        
        self.current_bot.remember(old_vector, action, reward, new_vector, done, priority)
        loss = self.current_bot.replay()
        
        if done:
            # Update episode statistics
            self.current_bot.stats.games_played += 1
            # Additional episode completion logic would go here
    
    def load_all_models(self):
        """Load all saved AI models"""
        if not os.path.exists(self.models_dir):
            return
        
        loaded_count = 0
        for filename in os.listdir(self.models_dir):
            if filename.endswith('.json'):
                bot_name = filename[:-5]  # Remove .json extension
                self.bots[bot_name] = AdvancedQLearningAgent()
                
                model_path = os.path.join(self.models_dir, filename)
                if self.bots[bot_name].load_model(model_path):
                    loaded_count += 1
                    print(f"📚 Loaded bot: {bot_name}")
                else:
                    del self.bots[bot_name]
        
        print(f"✅ Loaded {loaded_count} AI bots")
    
    def get_comprehensive_bot_stats(self, bot_name):
        """Get comprehensive statistics for a specific bot"""
        if bot_name not in self.bots:
            return None
        
        bot = self.bots[bot_name]
        
        # Calculate advanced metrics
        recent_performance = list(bot.training_history['episode_rewards'])[-20:] if bot.training_history['episode_rewards'] else [0]
        performance_trend = np.polyfit(range(len(recent_performance)), recent_performance, 1)[0] if len(recent_performance) > 1 else 0
        
        return {
            'basic_stats': asdict(bot.stats),
            'training_stats': {
                'episodes_trained': len(bot.training_history['episode_rewards']),
                'current_epsilon': bot.epsilon,
                'learning_rate': bot.learning_rate,
                'architecture': bot.architecture
            },
            'performance_analysis': {
                'recent_average': np.mean(recent_performance),
                'performance_trend': performance_trend,
                'consistency_score': bot.stats.consistency_rating,
                'learning_efficiency': self.calculate_learning_efficiency(bot)
            },
            'action_analysis': {
                'action_preferences': {str(k): np.mean(v) for k, v in bot.action_success_rates.items()},
                'most_successful_action': max(bot.action_success_rates.items(), 
                                            key=lambda x: np.mean(x[1]))[0] if bot.action_success_rates else 0
            }
        }
    
    def calculate_learning_efficiency(self, bot):
        """Calculate how efficiently the bot is learning"""
        if len(bot.training_history['episode_rewards']) < 10:
            return 0.0
        
        # Compare early vs recent performance
        early_performance = np.mean(list(bot.training_history['episode_rewards'])[:10])
        recent_performance = np.mean(list(bot.training_history['episode_rewards'])[-10:])
        
        improvement = recent_performance - early_performance
        episodes_taken = len(bot.training_history['episode_rewards'])
        
        # Normalize efficiency score (0-100)
        efficiency = min(100, max(0, (improvement / max(1, episodes_taken)) * 1000))
        return efficiency
    
    def create_bot_tournament(self, tournament_name, bot_names, tournament_type='round_robin'):
        """Create a tournament between bots"""
        available_bots = [name for name in bot_names if name in self.bots]
        
        if len(available_bots) < 2:
            print("❌ Need at least 2 bots for a tournament")
            return None
        
        tournament = self.tournament_manager.create_tournament(
            tournament_name, available_bots, tournament_type
        )
        
        print(f"🏆 Created tournament: {tournament_name}")
        print(f"   🤖 Participants: {', '.join(available_bots)}")
        print(f"   📋 Type: {tournament_type}")
        
        return tournament
    
    def human_vs_ai_challenge(self, bot_name, challenge_type='score_battle'):
        """Set up various types of human vs AI challenges"""
        if bot_name not in self.bots:
            print(f"Bot '{bot_name}' not found!")
            return False
        
        bot_stats = self.get_comprehensive_bot_stats(bot_name)
        
        print(f"🥊 HUMAN vs AI CHALLENGE: You vs {bot_name}")
        print(f"   🤖 Bot Level: {bot_stats['basic_stats']['average_score']:.0f} avg score")
        print(f"   🏆 Bot Best: {bot_stats['basic_stats']['best_score']}")
        print(f"   📊 Bot Consistency: {bot_stats['performance_analysis']['consistency_score']:.1f}%")
        print(f"   🎯 Challenge Type: {challenge_type}")
        print()
        print("   💡 Tips for beating the AI:")
        print("      • Focus on consistency over high scores")
        print("      • Collect powerups strategically") 
        print("      • Use model-specific abilities effectively")
        
        return True

class AIGameInterface:
    """Enhanced interface for AI to interact with the game"""
    
    def __init__(self):
        self.last_state = None
        self.last_action = 0
        self.reward_calculator = AdvancedRewardCalculator()
        self.action_history = deque(maxlen=100)
        self.state_history = deque(maxlen=50)
    
    def get_current_state(self, player, obstacle_manager, powerup_manager, stats, additional_context=None):
        """Extract comprehensive game state for AI"""
        # Enhanced player position data
        player_pos = (
            player.rect.x, 
            player.rect.y, 
            getattr(player, 'velocity_y', 0),
            getattr(player, 'grounded', True),
            player.activation_level / 100.0  # Normalized health
        )
        
        # Get comprehensive obstacle data
        obstacles = []
        for obs in obstacle_manager.obstacles:
            if obs.rect.x > player.rect.x - 200:  # Extended range
                obstacles.append({
                    'x': obs.rect.x - player.rect.x,  # Relative position
                    'y': obs.rect.y,
                    'width': obs.rect.width,
                    'height': obs.rect.height,
                    'type': getattr(obs, 'obstacle_type', 0),
                    'speed': getattr(obs, 'speed', 1.0)
                })
        
        # Get comprehensive powerup data
        powerups = []
        for pup in powerup_manager.powerups:
            if pup.rect.x > player.rect.x - 200:  # Extended range
                powerups.append({
                    'x': pup.rect.x - player.rect.x,  # Relative position
                    'y': pup.rect.y,
                    'type': getattr(pup, 'powerup_type', 0),
                    'value': getattr(pup, 'value', 1.0)
                })
        
        # Enhanced additional context
        enhanced_context = {
            'time_since_last_powerup': getattr(stats, 'time_since_last_powerup', 0),
            'combo_multiplier': getattr(stats, 'combo_multiplier', 1.0),
            'difficulty_level': obstacle_manager.current_speed_multiplier,
            'recent_performance': np.mean([s.get('score', 0) for s in list(self.state_history)[-10:]]) if self.state_history else 0
        }
        
        if additional_context:
            enhanced_context.update(additional_context)
        
        current_state = GameState(
            player_pos=player_pos,
            obstacles=obstacles,
            powerups=powerups,
            speed=obstacle_manager.current_speed_multiplier,
            score=stats.score,
            additional_context=enhanced_context
        )
        
        # Store state history
        self.state_history.append({
            'score': stats.score,
            'obstacles_dodged': getattr(stats, 'obstacles_dodged', 0),
            'timestamp': time.time()
        })
        
        return current_state
    
    def execute_action(self, action, player):
        """Execute AI action on the player with enhanced action mapping"""
        self.action_history.append({'action': action, 'timestamp': time.time()})
        
        # Enhanced action mapping
        if action == 1:  # Jump
            keys_pressed = {pygame.K_SPACE: True}
            player.handle_input(keys_pressed)
        elif action == 2:  # Duck
            keys_pressed = {pygame.K_DOWN: True}
            player.handle_input(keys_pressed)
        elif action == 3:  # Layer Boost
            keys_pressed = {pygame.K_l: True}
            player.handle_input(keys_pressed)
        elif action == 4:  # Special Ability
            keys_pressed = {pygame.K_x: True}
            player.handle_input(keys_pressed)
        elif action == 5:  # Combo action (multiple keys)
            keys_pressed = {pygame.K_SPACE: True, pygame.K_l: True}
            player.handle_input(keys_pressed)
        # action == 0 means do nothing

class AdvancedRewardCalculator:
    """Advanced reward calculation system"""
    
    def __init__(self):
        self.last_score = 0
        self.last_distance = 0
        self.last_obstacles_dodged = 0
        self.last_powerups_collected = 0
        self.last_health = 100
        self.performance_baseline = 0
        self.adaptive_scaling = 1.0
    
    def calculate_reward(self, stats, player, collision_occurred, powerup_collected, near_miss=False):
        """Calculate sophisticated reward based on multiple factors"""
        reward = 0
        
        # Base survival reward (scales with game difficulty)
        base_survival = 1.0 * (1 + stats.score / 10000)
        reward += base_survival
        
        # Score-based rewards
        score_increase = stats.score - self.last_score
        if score_increase > 0:
            reward += score_increase * 0.1
        
        # Distance-based rewards
        distance_increase = getattr(stats, 'distance_traveled', 0) - self.last_distance
        reward += distance_increase * 0.01
        
        # Obstacle dodging rewards (higher for consecutive dodges)
        obstacles_dodged_increase = getattr(stats, 'obstacles_dodged', 0) - self.last_obstacles_dodged
        if obstacles_dodged_increase > 0:
            consecutive_bonus = min(obstacles_dodged_increase * 2, 20)  # Cap bonus
            reward += 10 + consecutive_bonus
        
        # Near miss bonus (risk-taking reward)
        if near_miss:
            reward += 5
        
        # Powerup collection rewards
        powerups_collected_increase = getattr(stats, 'powerups_collected', 0) - self.last_powerups_collected
        if powerups_collected_increase > 0:
            reward += powerups_collected_increase * 8
        
        # Health management rewards
        current_health = getattr(player, 'activation_level', 100)
        if current_health > self.last_health:
            reward += (current_health - self.last_health) * 0.5  # Health recovery bonus
        
        # Penalties
        if collision_occurred:
            # Adaptive penalty based on performance
            penalty = -50 * (1 + self.adaptive_scaling)
            reward += penalty
        
        # Performance consistency bonus
        if stats.score > self.performance_baseline:
            consistency_bonus = min((stats.score - self.performance_baseline) * 0.01, 10)
            reward += consistency_bonus
        
        # Update tracking variables
        self.last_score = stats.score
        self.last_distance = getattr(stats, 'distance_traveled', 0)
        self.last_obstacles_dodged = getattr(stats, 'obstacles_dodged', 0)
        self.last_powerups_collected = getattr(stats, 'powerups_collected', 0)
        self.last_health = current_health
        
        # Update performance baseline (moving average)
        self.performance_baseline = self.performance_baseline * 0.99 + stats.score * 0.01
        
        # Adaptive scaling adjustment
        if reward > 20:  # High reward
            self.adaptive_scaling *= 1.01
        elif reward < -20:  # High penalty
            self.adaptive_scaling *= 0.99
        
        return reward

# Global AI competition system (with error handling)
try:
    ai_competition = AICompetition() if AI_AVAILABLE else None
    ai_interface = AIGameInterface() if AI_AVAILABLE else None
    if not AI_AVAILABLE:
        print("⚠️ AI Competition features disabled - install numpy for full functionality")
except Exception as e:
    print(f"❌ Error initializing AI Competition system: {e}")
    ai_competition = None
    ai_interface = None
