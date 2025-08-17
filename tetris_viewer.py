# tetris_viewer.py
# Watch your trained Tetris agent play with real-time visualization

import os
import numpy as np
import pygame
import time
import argparse
from typing import List, Tuple, Optional

import torch
import torch.nn as nn

# Import your existing classes (make sure these are available)
from tetris_dqn_placement import (
    TetrisEnv, PlacementDQNAgent, ALL_PLACEMENTS, PIECES, PIECE_NAMES, 
    PIECE_INDEX, BOARD_W, BOARD_H, analyze_board
)

class TetrisViewer:
    """Enhanced Tetris visualization with game stats and controls"""
    
    COLORS = {
        "I": (0, 255, 255),    # Cyan
        "O": (255, 255, 0),    # Yellow
        "T": (128, 0, 128),    # Purple
        "S": (0, 255, 0),      # Green
        "Z": (255, 0, 0),      # Red
        "J": (0, 0, 255),      # Blue
        "L": (255, 165, 0),    # Orange
        "PLACED": (200, 200, 200),  # Light gray for placed pieces
        "GHOST": (100, 100, 100),   # Dark gray for ghost piece
    }
    
    def __init__(self, cell_size=30, fps=60):
        pygame.init()
        
        self.cell_size = cell_size
        self.fps = fps
        
        # Calculate dimensions
        self.board_width = BOARD_W * cell_size
        self.board_height = BOARD_H * cell_size
        self.sidebar_width = 300
        self.total_width = self.board_width + self.sidebar_width
        
        # Create display
        self.screen = pygame.display.set_mode((self.total_width, self.board_height))
        pygame.display.set_caption("Tetris AI Agent Viewer")
        
        # Fonts
        self.font_large = pygame.font.SysFont("Arial", 24, bold=True)
        self.font_medium = pygame.font.SysFont("Arial", 18)
        self.font_small = pygame.font.SysFont("Arial", 14)
        
        self.clock = pygame.time.Clock()
        
        # Game state tracking
        self.game_stats = {
            'games_played': 0,
            'total_lines': 0,
            'total_pieces': 0,
            'best_lines': 0,
            'current_lines': 0,
            'current_pieces': 0,
            'current_score': 0
        }
        
        self.paused = False
        self.speed_multiplier = 1.0
        
    def draw_cell(self, x, y, color, border=True):
        """Draw a single cell"""
        rect = pygame.Rect(x * self.cell_size, y * self.cell_size, 
                          self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, color, rect)
        
        if border:
            pygame.draw.rect(self.screen, (50, 50, 50), rect, 1)
    
    def draw_board(self, env):
        """Draw the game board with placed pieces"""
        # Fill background
        board_rect = pygame.Rect(0, 0, self.board_width, self.board_height)
        pygame.draw.rect(self.screen, (20, 20, 30), board_rect)
        
        # Draw placed pieces
        for y in range(BOARD_H):
            for x in range(BOARD_W):
                if env.board[y, x] == 1:
                    self.draw_cell(x, y, self.COLORS["PLACED"])
        
        # Draw grid lines
        for x in range(BOARD_W + 1):
            pygame.draw.line(self.screen, (40, 40, 40), 
                           (x * self.cell_size, 0), 
                           (x * self.cell_size, self.board_height))
        
        for y in range(BOARD_H + 1):
            pygame.draw.line(self.screen, (40, 40, 40), 
                           (0, y * self.cell_size), 
                           (self.board_width, y * self.cell_size))
    
    def draw_piece_preview(self, piece_name, x_offset, y_offset, scale=0.8):
        """Draw a piece preview in the sidebar"""
        if not piece_name or piece_name not in PIECES:
            return
            
        shape = PIECES[piece_name][0]  # Use first rotation
        color = self.COLORS[piece_name]
        
        # Calculate centered position
        piece_width = shape.shape[1] * self.cell_size * scale
        piece_height = shape.shape[0] * self.cell_size * scale
        
        start_x = x_offset + (100 - piece_width) // 2
        start_y = y_offset + (60 - piece_height) // 2
        
        for y in range(shape.shape[0]):
            for x in range(shape.shape[1]):
                if shape[y, x]:
                    rect = pygame.Rect(
                        start_x + x * self.cell_size * scale,
                        start_y + y * self.cell_size * scale,
                        self.cell_size * scale,
                        self.cell_size * scale
                    )
                    pygame.draw.rect(self.screen, color, rect)
                    pygame.draw.rect(self.screen, (100, 100, 100), rect, 1)
    
    def draw_sidebar(self, env, agent, action_info=None):
        """Draw the information sidebar"""
        sidebar_x = self.board_width + 10
        y_pos = 10
        
        # Background
        sidebar_rect = pygame.Rect(self.board_width, 0, self.sidebar_width, self.board_height)
        pygame.draw.rect(self.screen, (30, 30, 40), sidebar_rect)
        
        # Title
        title = self.font_large.render("Tetris AI", True, (255, 255, 255))
        self.screen.blit(title, (sidebar_x, y_pos))
        y_pos += 40
        
        # Current piece
        if env.cur_piece:
            label = self.font_medium.render("Current:", True, (200, 200, 200))
            self.screen.blit(label, (sidebar_x, y_pos))
            y_pos += 25
            
            # Current piece preview
            preview_rect = pygame.Rect(sidebar_x, y_pos, 100, 60)
            pygame.draw.rect(self.screen, (50, 50, 50), preview_rect, 2)
            self.draw_piece_preview(env.cur_piece, sidebar_x, y_pos)
            y_pos += 70
        
        # Next piece
        if env.next_piece:
            label = self.font_medium.render("Next:", True, (200, 200, 200))
            self.screen.blit(label, (sidebar_x, y_pos))
            y_pos += 25
            
            # Next piece preview
            preview_rect = pygame.Rect(sidebar_x, y_pos, 100, 60)
            pygame.draw.rect(self.screen, (50, 50, 50), preview_rect, 2)
            self.draw_piece_preview(env.next_piece, sidebar_x, y_pos)
            y_pos += 80
        
        # Game statistics
        stats_title = self.font_medium.render("Current Game:", True, (255, 255, 100))
        self.screen.blit(stats_title, (sidebar_x, y_pos))
        y_pos += 30
        
        current_stats = [
            f"Lines: {self.game_stats['current_lines']}",
            f"Pieces: {self.game_stats['current_pieces']}",
            f"Score: {self.game_stats['current_score']:.0f}",
        ]
        
        for stat in current_stats:
            text = self.font_small.render(stat, True, (200, 200, 200))
            self.screen.blit(text, (sidebar_x, y_pos))
            y_pos += 20
        
        y_pos += 20
        
        # Overall statistics
        overall_title = self.font_medium.render("Session Stats:", True, (255, 255, 100))
        self.screen.blit(overall_title, (sidebar_x, y_pos))
        y_pos += 30
        
        overall_stats = [
            f"Games: {self.game_stats['games_played']}",
            f"Total Lines: {self.game_stats['total_lines']}",
            f"Best Game: {self.game_stats['best_lines']} lines",
            f"Avg Lines/Game: {self.game_stats['total_lines']/(max(1, self.game_stats['games_played'])):.1f}",
        ]
        
        for stat in overall_stats:
            text = self.font_small.render(stat, True, (200, 200, 200))
            self.screen.blit(text, (sidebar_x, y_pos))
            y_pos += 20
        
        y_pos += 30
        
        # Agent info
        if agent:
            agent_title = self.font_medium.render("Agent Info:", True, (100, 255, 100))
            self.screen.blit(agent_title, (sidebar_x, y_pos))
            y_pos += 30
            
            epsilon = agent.get_epsilon()
            agent_info = [
                f"Epsilon: {epsilon:.3f}",
                f"Training Steps: {agent.steps:,}",
                f"Buffer Size: {len(agent.buffer):,}",
            ]
            
            if action_info:
                agent_info.append(f"Action: {action_info}")
            
            for info in agent_info:
                text = self.font_small.render(info, True, (200, 200, 200))
                self.screen.blit(text, (sidebar_x, y_pos))
                y_pos += 18
        
        # Controls
        y_pos = self.board_height - 150
        controls_title = self.font_medium.render("Controls:", True, (255, 100, 100))
        self.screen.blit(controls_title, (sidebar_x, y_pos))
        y_pos += 25
        
        controls = [
            "SPACE: Pause/Resume",
            "R: Reset Game",
            "Q: Quit",
            "1-5: Speed Control",
            f"Speed: {self.speed_multiplier:.1f}x",
        ]
        
        if self.paused:
            controls.insert(0, "⏸ PAUSED")
        
        for control in controls:
            color = (255, 255, 100) if control.startswith("⏸") else (180, 180, 180)
            text = self.font_small.render(control, True, color)
            self.screen.blit(text, (sidebar_x, y_pos))
            y_pos += 18
    
    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    return False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_r:
                    return "reset"
                elif event.key == pygame.K_1:
                    self.speed_multiplier = 0.5
                elif event.key == pygame.K_2:
                    self.speed_multiplier = 1.0
                elif event.key == pygame.K_3:
                    self.speed_multiplier = 2.0
                elif event.key == pygame.K_4:
                    self.speed_multiplier = 5.0
                elif event.key == pygame.K_5:
                    self.speed_multiplier = 10.0
        
        return True
    
    def update_stats(self, env, reward):
        """Update game statistics"""
        self.game_stats['current_lines'] = env.total_lines
        self.game_stats['current_pieces'] = env.total_pieces
        self.game_stats['current_score'] += reward
        
        if env.done:
            self.game_stats['games_played'] += 1
            self.game_stats['total_lines'] += env.total_lines
            self.game_stats['total_pieces'] += env.total_pieces
            
            if env.total_lines > self.game_stats['best_lines']:
                self.game_stats['best_lines'] = env.total_lines
            
            # Reset current game stats
            self.game_stats['current_lines'] = 0
            self.game_stats['current_pieces'] = 0
            self.game_stats['current_score'] = 0
    
    def render(self, env, agent, action_info=None):
        """Render the complete game state"""
        self.screen.fill((0, 0, 0))
        
        # Draw main game
        self.draw_board(env)
        
        # Draw sidebar
        self.draw_sidebar(env, agent, action_info)
        
        # Update display
        pygame.display.flip()
        
        # Control frame rate
        self.clock.tick(int(self.fps * self.speed_multiplier))

def load_trained_agent(checkpoint_path, device="cpu"):
    """Load a trained agent from checkpoint"""
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return None
    
    print(f"Loading agent from: {checkpoint_path}")
    
    # Create agent
    agent = PlacementDQNAgent(
        n_actions=len(ALL_PLACEMENTS),
        device=device,
        lr=1e-4,  # Not used for inference
        eps_start=0.0,  # No exploration for viewing
        eps_end=0.0,
        eps_decay=1
    )
    
    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        agent.q_net.load_state_dict(checkpoint['q_net'])
        agent.target_net.load_state_dict(checkpoint['target_net'])
        agent.steps = checkpoint.get('steps', 0)
        
        print(f"✅ Agent loaded successfully!")
        print(f"   Training steps: {agent.steps:,}")
        
        if 'episode' in checkpoint:
            print(f"   From episode: {checkpoint['episode']}")
        
        return agent
        
    except Exception as e:
        print(f"❌ Failed to load agent: {e}")
        return None

def get_action_description(action_idx):
    """Get human-readable description of the action"""
    if action_idx >= len(ALL_PLACEMENTS):
        return "Invalid Action"
    
    action = ALL_PLACEMENTS[action_idx]
    return f"{action.piece_name} rot:{action.rotation} x:{action.x}"

def watch_agent_play(checkpoint_path, games=None, fps=4, cell_size=25):
    """Watch the trained agent play Tetris"""
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load agent
    agent = load_trained_agent(checkpoint_path, device)
    if agent is None:
        return
    
    # Create environment and viewer
    env = TetrisEnv(seed=None)  # Random seed for variety
    viewer = TetrisViewer(cell_size=cell_size, fps=fps)
    
    print("\n🎮 Controls:")
    print("   SPACE: Pause/Resume")
    print("   R: Reset Game") 
    print("   Q: Quit")
    print("   1-5: Speed Control (0.5x - 10x)")
    print("\n▶️ Starting game viewer...")
    
    games_played = 0
    
    try:
        while games is None or games_played < games:
            # Reset game
            state = env.reset()
            viewer.update_stats(env, 0)
            
            print(f"\n🎯 Game {games_played + 1} started")
            
            step_count = 0
            while not env.done:
                # Handle events
                event_result = viewer.handle_events()
                if event_result is False:
                    print("\n👋 Viewer closed by user")
                    return
                elif event_result == "reset":
                    print("🔄 Game reset by user")
                    break
                
                if not viewer.paused:
                    # Get agent action
                    mask = env.get_valid_actions()
                    action_idx = agent._get_best_action(state, mask)
                    action_desc = get_action_description(action_idx)
                    
                    # Take step
                    next_state, reward, done, info = env.step(action_idx)
                    
                    # Update stats
                    viewer.update_stats(env, reward)
                    
                    # Print interesting events
                    if info.get('lines', 0) > 0:
                        lines = info['lines']
                        total = info.get('total_lines', 0)
                        line_names = {1: "Single", 2: "Double", 3: "Triple", 4: "TETRIS!"}
                        line_name = line_names.get(lines, f"{lines}-line")
                        print(f"   🔥 {line_name} clear! Total lines: {total}")
                    
                    state = next_state
                    step_count += 1
                    
                    # Render with action info
                    viewer.render(env, agent, action_desc)
                else:
                    # Just render when paused
                    viewer.render(env, agent, "PAUSED")
            
            if env.done:
                games_played += 1
                total_lines = env.total_lines
                total_pieces = env.total_pieces
                
                print(f"   📊 Game {games_played} finished:")
                print(f"      Lines cleared: {total_lines}")
                print(f"      Pieces placed: {total_pieces}")
                print(f"      Steps taken: {step_count}")
                
                # Show final state for a moment
                viewer.render(env, agent, "GAME OVER")
                time.sleep(2.0 / viewer.speed_multiplier)
    
    except KeyboardInterrupt:
        print("\n⏹️ Stopped by user")
    
    finally:
        pygame.quit()
        print(f"\n📈 Session Summary:")
        print(f"   Games played: {viewer.game_stats['games_played']}")
        print(f"   Total lines: {viewer.game_stats['total_lines']}")
        print(f"   Best game: {viewer.game_stats['best_lines']} lines")
        if viewer.game_stats['games_played'] > 0:
            avg_lines = viewer.game_stats['total_lines'] / viewer.game_stats['games_played']
            print(f"   Average lines per game: {avg_lines:.1f}")

def find_latest_checkpoint(checkpoint_dir="checkpoints"):
    """Find the most recent checkpoint"""
    if not os.path.exists(checkpoint_dir):
        return None
    
    files = [f for f in os.listdir(checkpoint_dir) 
             if f.startswith("placement_tetris_ep") and f.endswith(".pth")]
    
    if not files:
        return None
    
    # Extract episode numbers and find the latest
    episodes = []
    for f in files:
        try:
            ep_str = f.replace("placement_tetris_ep", "").replace(".pth", "")
            episodes.append((int(ep_str), f))
        except ValueError:
            continue
    
    if episodes:
        latest_ep, latest_file = max(episodes)
        return os.path.join(checkpoint_dir, latest_file)
    
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Watch trained Tetris agent play")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint file")
    parser.add_argument("--games", type=int, help="Number of games to play (default: infinite)")
    parser.add_argument("--fps", type=int, default=4, help="Frames per second (default: 4)")
    parser.add_argument("--size", type=int, default=25, help="Cell size in pixels (default: 25)")
    
    args = parser.parse_args()
    
    # Find checkpoint
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        print("Looking for latest checkpoint...")
        checkpoint_path = find_latest_checkpoint()
        if checkpoint_path is None:
            print("❌ No checkpoints found! Train an agent first.")
            exit(1)
        print(f"Found: {checkpoint_path}")
    
    # Watch the agent play
    watch_agent_play(
        checkpoint_path=checkpoint_path,
        games=args.games,
        fps=args.fps,
        cell_size=args.size
    )