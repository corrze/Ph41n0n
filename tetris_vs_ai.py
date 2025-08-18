# tetris_vs_ai.py
# Play Tetris against your trained AI in real-time!

import os
import pygame
import numpy as np
import random
import time
import threading
from typing import Optional, Tuple, List
from enum import Enum

import torch

# Import your existing classes
from tetris_dqn_placement import (
    TetrisEnv, PlacementDQNAgent, ALL_PLACEMENTS, PIECES, PIECE_NAMES, 
    PIECE_INDEX, BOARD_W, BOARD_H, fits, place_and_clear
)

class GameState(Enum):
    MENU = "menu"
    PLAYING = "playing"
    GAME_OVER = "game_over"

class HumanTetrisEnv:
    """Human-controllable Tetris environment"""
    
    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        self.board = np.zeros((BOARD_H, BOARD_W), dtype=np.uint8)
        self.cur_piece = None
        self.next_piece = None
        self.piece_x = 0
        self.piece_y = 0
        self.piece_rotation = 0
        self.done = False
        self.bag = []
        self.total_lines = 0
        self.total_pieces = 0
        self.score = 0
        self.drop_timer = 0
        self.drop_interval = 60  # Frames between automatic drops
        self.last_drop_time = 0

    def _refill_bag(self):
        self.bag = PIECE_NAMES[:]
        self.rng.shuffle(self.bag)

    def _draw_piece(self):
        if not self.bag:
            self._refill_bag()
        return self.bag.pop()

    def _spawn_piece(self):
        """Spawn new piece at top center"""
        self.cur_piece = self.next_piece if self.next_piece else self._draw_piece()
        self.next_piece = self._draw_piece()
        self.total_pieces += 1
        
        self.piece_rotation = 0
        self.piece_x = BOARD_W // 2 - 1
        self.piece_y = 0
        self.drop_timer = 0
        
        # Check if spawn position is valid
        if not self._piece_fits(self.piece_x, self.piece_y, self.piece_rotation):
            self.done = True

    def _piece_fits(self, x: int, y: int, rotation: int) -> bool:
        if self.cur_piece is None:
            return False
        shape = PIECES[self.cur_piece][rotation]
        return fits(self.board, shape, x, y)

    def _place_piece(self):
        """Place current piece and clear lines"""
        shape = PIECES[self.cur_piece][self.piece_rotation]
        self.board, lines = place_and_clear(self.board, shape, self.piece_x, self.piece_y)
        
        # Score based on lines cleared (standard Tetris scoring)
        line_scores = {0: 0, 1: 100, 2: 300, 3: 500, 4: 800}
        self.score += line_scores.get(lines, 0)
        self.total_lines += lines
        
        self._spawn_piece()
        return lines

    def reset(self):
        self.board[:] = 0
        self.done = False
        self.bag = []
        self.next_piece = None
        self.total_lines = 0
        self.total_pieces = 0
        self.score = 0
        self.drop_timer = 0
        self._spawn_piece()

    def update(self):
        """Update game state (gravity)"""
        if self.done:
            return
        
        self.drop_timer += 1
        if self.drop_timer >= self.drop_interval:
            self.drop_timer = 0
            if self._piece_fits(self.piece_x, self.piece_y + 1, self.piece_rotation):
                self.piece_y += 1
            else:
                self._place_piece()

    def move_left(self):
        if not self.done and self._piece_fits(self.piece_x - 1, self.piece_y, self.piece_rotation):
            self.piece_x -= 1

    def move_right(self):
        if not self.done and self._piece_fits(self.piece_x + 1, self.piece_y, self.piece_rotation):
            self.piece_x += 1

    def rotate(self):
        if self.done:
            return
        new_rotation = (self.piece_rotation + 1) % len(PIECES[self.cur_piece])
        
        # Try rotation at current position
        if self._piece_fits(self.piece_x, self.piece_y, new_rotation):
            self.piece_rotation = new_rotation
        else:
            # Try wall kicks
            for dx in [-1, 1, -2, 2]:
                if self._piece_fits(self.piece_x + dx, self.piece_y, new_rotation):
                    self.piece_x += dx
                    self.piece_rotation = new_rotation
                    break

    def soft_drop(self):
        if not self.done and self._piece_fits(self.piece_x, self.piece_y + 1, self.piece_rotation):
            self.piece_y += 1
            self.score += 1  # Bonus for soft drop

    def hard_drop(self):
        if self.done:
            return
        drop_distance = 0
        while self._piece_fits(self.piece_x, self.piece_y + 1, self.piece_rotation):
            self.piece_y += 1
            drop_distance += 1
        self.score += drop_distance * 2  # Bonus for hard drop
        self._place_piece()

    def get_board_with_piece(self):
        """Get board with current piece rendered"""
        if self.done or self.cur_piece is None:
            return self.board.copy()
            
        board_with_piece = self.board.copy()
        shape = PIECES[self.cur_piece][self.piece_rotation]
        h, w = shape.shape
        
        for r in range(h):
            for c in range(w):
                if (shape[r, c] and 
                    0 <= self.piece_y + r < BOARD_H and 
                    0 <= self.piece_x + c < BOARD_W):
                    board_with_piece[self.piece_y + r, self.piece_x + c] = 2
        
        return board_with_piece

    def get_ghost_piece_pos(self):
        """Get position where piece would land"""
        if self.done or self.cur_piece is None:
            return None
        
        ghost_y = self.piece_y
        while self._piece_fits(self.piece_x, ghost_y + 1, self.piece_rotation):
            ghost_y += 1
        return (self.piece_x, ghost_y, self.piece_rotation)

class TetrisVsAI:
    """Main game class for human vs AI Tetris"""
    
    COLORS = {
        "I": (0, 255, 255),    # Cyan
        "O": (255, 255, 0),    # Yellow
        "T": (128, 0, 128),    # Purple
        "S": (0, 255, 0),      # Green
        "Z": (255, 0, 0),      # Red
        "J": (0, 0, 255),      # Blue
        "L": (255, 165, 0),    # Orange
        "PLACED": (200, 200, 200),     # Light gray
        "FALLING": (255, 255, 255),    # White
        "GHOST": (100, 100, 100),      # Dark gray
        "AI_PLACED": (150, 150, 255),  # Light blue for AI
        "AI_FALLING": (100, 150, 255), # Blue for AI falling piece
    }
    
    def __init__(self, checkpoint_path: str):
        pygame.init()
        
        # Display settings
        self.cell_size = 25
        self.board_width = BOARD_W * self.cell_size
        self.board_height = BOARD_H * self.cell_size
        self.sidebar_width = 200
        self.gap = 50
        
        self.screen_width = self.board_width * 2 + self.sidebar_width + self.gap * 3
        self.screen_height = self.board_height + 100
        
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Tetris vs AI")
        
        # Fonts
        self.font_large = pygame.font.SysFont("Arial", 32, bold=True)
        self.font_medium = pygame.font.SysFont("Arial", 20, bold=True)
        self.font_small = pygame.font.SysFont("Arial", 16)
        
        self.clock = pygame.time.Clock()
        
        # Game state
        self.state = GameState.MENU
        self.human_env = HumanTetrisEnv(seed=None)
        self.ai_env = TetrisEnv(seed=None)
        
        # Load AI agent
        self.ai_agent = self.load_ai_agent(checkpoint_path)
        self.ai_thread = None
        self.ai_action_queue = []
        self.ai_active = False
        
        # Game timing
        self.fps = 60
        self.ai_move_timer = 0
        self.ai_move_interval = 30  # AI moves every 30 frames (0.5 seconds at 60 FPS)
        
        # UI positions
        self.human_board_x = self.gap
        self.ai_board_x = self.human_board_x + self.board_width + self.gap
        self.sidebar_x = self.ai_board_x + self.board_width + self.gap
        self.board_y = 50

    def load_ai_agent(self, checkpoint_path: str):
        """Load the trained AI agent"""
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}")
            return None
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        agent = PlacementDQNAgent(
            n_actions=len(ALL_PLACEMENTS),
            device=device,
            eps_start=0.0,  # No exploration
            eps_end=0.0,
            eps_decay=1
        )
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
        except:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        agent.q_net.load_state_dict(checkpoint['q_net'])
        agent.target_net.load_state_dict(checkpoint['target_net'])
        
        print(f"AI agent loaded from {checkpoint_path}")
        return agent

    def ai_player_thread(self):
        """AI player runs in separate thread"""
        while self.ai_active and not self.ai_env.done:
            if self.state == GameState.PLAYING:
                # Get AI action
                state = self.ai_env.get_state()
                mask = self.ai_env.get_valid_actions()
                
                if self.ai_agent and not self.ai_env.done:
                    action_idx = self.ai_agent._get_best_action(state, mask)
                    self.ai_action_queue.append(action_idx)
                
                time.sleep(0.5)  # AI thinks for 0.5 seconds
            else:
                time.sleep(0.1)

    def draw_cell(self, x, y, color, size=None):
        """Draw a cell at board coordinates"""
        if size is None:
            size = self.cell_size
        rect = pygame.Rect(x * self.cell_size, y * self.cell_size, size, size)
        pygame.draw.rect(self.screen, color, rect)
        pygame.draw.rect(self.screen, (50, 50, 50), rect, 1)

    def draw_board(self, board, offset_x, offset_y, is_ai=False):
        """Draw a Tetris board"""
        # Background
        board_rect = pygame.Rect(offset_x, offset_y, self.board_width, self.board_height)
        pygame.draw.rect(self.screen, (20, 20, 30), board_rect)
        
        # Draw placed pieces
        placed_color = self.COLORS["AI_PLACED"] if is_ai else self.COLORS["PLACED"]
        
        for y in range(BOARD_H):
            for x in range(BOARD_W):
                if board[y, x] == 1:
                    rect = pygame.Rect(
                        offset_x + x * self.cell_size,
                        offset_y + y * self.cell_size,
                        self.cell_size, self.cell_size
                    )
                    pygame.draw.rect(self.screen, placed_color, rect)
                    pygame.draw.rect(self.screen, (50, 50, 50), rect, 1)
                elif board[y, x] == 2:  # Falling piece
                    falling_color = self.COLORS["AI_FALLING"] if is_ai else self.COLORS["FALLING"]
                    rect = pygame.Rect(
                        offset_x + x * self.cell_size,
                        offset_y + y * self.cell_size,
                        self.cell_size, self.cell_size
                    )
                    pygame.draw.rect(self.screen, falling_color, rect)
                    pygame.draw.rect(self.screen, (100, 100, 100), rect, 1)
        
        # Draw grid
        for x in range(BOARD_W + 1):
            pygame.draw.line(self.screen, (40, 40, 40),
                           (offset_x + x * self.cell_size, offset_y),
                           (offset_x + x * self.cell_size, offset_y + self.board_height))
        
        for y in range(BOARD_H + 1):
            pygame.draw.line(self.screen, (40, 40, 40),
                           (offset_x, offset_y + y * self.cell_size),
                           (offset_x + self.board_width, offset_y + y * self.cell_size))

    def draw_ghost_piece(self, env, offset_x, offset_y):
        """Draw ghost piece (where piece will land)"""
        ghost_pos = env.get_ghost_piece_pos()
        if ghost_pos and env.cur_piece:
            ghost_x, ghost_y, ghost_rotation = ghost_pos
            shape = PIECES[env.cur_piece][ghost_rotation]
            h, w = shape.shape
            
            for r in range(h):
                for c in range(w):
                    if shape[r, c]:
                        rect = pygame.Rect(
                            offset_x + (ghost_x + c) * self.cell_size,
                            offset_y + (ghost_y + r) * self.cell_size,
                            self.cell_size, self.cell_size
                        )
                        pygame.draw.rect(self.screen, self.COLORS["GHOST"], rect, 2)

    def draw_next_piece(self, piece_name, x, y):
        """Draw next piece preview"""
        if not piece_name or piece_name not in PIECES:
            return
        
        shape = PIECES[piece_name][0]
        color = self.COLORS[piece_name]
        
        # Center the piece in preview area
        piece_width = shape.shape[1] * 20
        piece_height = shape.shape[0] * 20
        start_x = x + (80 - piece_width) // 2
        start_y = y + (60 - piece_height) // 2
        
        for r in range(shape.shape[0]):
            for c in range(shape.shape[1]):
                if shape[r, c]:
                    rect = pygame.Rect(start_x + c * 20, start_y + r * 20, 20, 20)
                    pygame.draw.rect(self.screen, color, rect)
                    pygame.draw.rect(self.screen, (100, 100, 100), rect, 1)

    def draw_menu(self):
        """Draw start menu"""
        self.screen.fill((30, 30, 50))
        
        # Title
        title = self.font_large.render("TETRIS vs AI", True, (255, 255, 255))
        title_rect = title.get_rect(center=(self.screen_width // 2, 150))
        self.screen.blit(title, title_rect)
        
        # Instructions
        instructions = [
            "Human Controls:",
            "A/D - Move Left/Right",
            "S - Soft Drop", 
            "W - Rotate",
            "SPACE - Hard Drop",
            "",
            "Press ENTER to Start!",
            "Press ESC to Quit"
        ]
        
        y = 250
        for instruction in instructions:
            color = (255, 255, 100) if "Press" in instruction else (200, 200, 200)
            text = self.font_small.render(instruction, True, color)
            text_rect = text.get_rect(center=(self.screen_width // 2, y))
            self.screen.blit(text, text_rect)
            y += 30

    def draw_game_over(self):
        """Draw game over screen"""
        # Semi-transparent overlay
        overlay = pygame.Surface((self.screen_width, self.screen_height))
        overlay.set_alpha(128)
        overlay.fill((0, 0, 0))
        self.screen.blit(overlay, (0, 0))
        
        # Game Over text
        game_over_text = self.font_large.render("GAME OVER", True, (255, 100, 100))
        game_over_rect = game_over_text.get_rect(center=(self.screen_width // 2, 200))
        self.screen.blit(game_over_text, game_over_rect)
        
        # Results
        human_won = not self.human_env.done or (self.human_env.done and self.ai_env.done and self.human_env.score > self.ai_env.total_lines * 100)
        ai_won = not self.ai_env.done or (self.human_env.done and self.ai_env.done and self.ai_env.total_lines * 100 > self.human_env.score)
        
        if human_won and not ai_won:
            winner_text = self.font_medium.render("HUMAN WINS!", True, (100, 255, 100))
        elif ai_won and not human_won:
            winner_text = self.font_medium.render("AI WINS!", True, (255, 100, 100))
        else:
            winner_text = self.font_medium.render("TIE GAME!", True, (255, 255, 100))
        
        winner_rect = winner_text.get_rect(center=(self.screen_width // 2, 250))
        self.screen.blit(winner_text, winner_rect)
        
        # Final scores
        human_score_text = self.font_small.render(f"Human Score: {self.human_env.score}", True, (200, 200, 200))
        ai_score_text = self.font_small.render(f"AI Score: {self.ai_env.total_lines * 100}", True, (200, 200, 200))
        
        human_score_rect = human_score_text.get_rect(center=(self.screen_width // 2, 300))
        ai_score_rect = ai_score_text.get_rect(center=(self.screen_width // 2, 330))
        
        self.screen.blit(human_score_text, human_score_rect)
        self.screen.blit(ai_score_text, ai_score_rect)
        
        # Restart instruction
        restart_text = self.font_small.render("Press R to Restart or ESC to Quit", True, (255, 255, 100))
        restart_rect = restart_text.get_rect(center=(self.screen_width // 2, 400))
        self.screen.blit(restart_text, restart_rect)

    def draw_game(self):
        """Draw the main game"""
        self.screen.fill((40, 40, 60))
        
        # Draw both boards
        human_board = self.human_env.get_board_with_piece()
        ai_board = self.ai_env.get_full_state() if hasattr(self.ai_env, 'get_full_state') else self.ai_env.board
        
        self.draw_board(human_board, self.human_board_x, self.board_y, is_ai=False)
        self.draw_board(ai_board, self.ai_board_x, self.board_y, is_ai=True)
        
        # Draw ghost pieces
        self.draw_ghost_piece(self.human_env, self.human_board_x, self.board_y)
        
        # Labels
        human_label = self.font_medium.render("HUMAN", True, (255, 255, 255))
        ai_label = self.font_medium.render("AI", True, (255, 255, 255))
        
        human_label_rect = human_label.get_rect(center=(self.human_board_x + self.board_width // 2, 20))
        ai_label_rect = ai_label.get_rect(center=(self.ai_board_x + self.board_width // 2, 20))
        
        self.screen.blit(human_label, human_label_rect)
        self.screen.blit(ai_label, ai_label_rect)
        
        # Sidebar with stats
        y = self.board_y
        
        # Human stats
        stats = [
            "HUMAN STATS:",
            f"Score: {self.human_env.score}",
            f"Lines: {self.human_env.total_lines}",
            f"Pieces: {self.human_env.total_pieces}",
            "",
            "AI STATS:",
            f"Lines: {self.ai_env.total_lines}",
            f"Pieces: {self.ai_env.total_pieces}",
            f"Score: {self.ai_env.total_lines * 100}",
            "",
            "NEXT PIECES:"
        ]
        
        for stat in stats:
            color = (255, 255, 100) if "STATS:" in stat else (200, 200, 200)
            text = self.font_small.render(stat, True, color)
            self.screen.blit(text, (self.sidebar_x, y))
            y += 25
        
        # Next piece previews
        if self.human_env.next_piece:
            text = self.font_small.render("Human:", True, (200, 200, 200))
            self.screen.blit(text, (self.sidebar_x, y))
            self.draw_next_piece(self.human_env.next_piece, self.sidebar_x, y + 20)
            y += 100
        
        if self.ai_env.next_piece:
            text = self.font_small.render("AI:", True, (200, 200, 200))
            self.screen.blit(text, (self.sidebar_x, y))
            self.draw_next_piece(self.ai_env.next_piece, self.sidebar_x, y + 20)

    def start_game(self):
        """Start a new game"""
        self.human_env.reset()
        self.ai_env.reset()
        self.state = GameState.PLAYING
        
        # Start AI thread
        self.ai_active = True
        self.ai_action_queue = []
        self.ai_thread = threading.Thread(target=self.ai_player_thread, daemon=True)
        self.ai_thread.start()

    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                
                if self.state == GameState.MENU:
                    if event.key == pygame.K_RETURN:
                        self.start_game()
                
                elif self.state == GameState.PLAYING:
                    # Human controls
                    if event.key == pygame.K_a:
                        self.human_env.move_left()
                    elif event.key == pygame.K_d:
                        self.human_env.move_right()
                    elif event.key == pygame.K_w:
                        self.human_env.rotate()
                    elif event.key == pygame.K_s:
                        self.human_env.soft_drop()
                    elif event.key == pygame.K_SPACE:
                        self.human_env.hard_drop()
                
                elif self.state == GameState.GAME_OVER:
                    if event.key == pygame.K_r:
                        self.start_game()
        
        return True

    def update_game(self):
        """Update game state"""
        if self.state != GameState.PLAYING:
            return
        
        # Update human game
        self.human_env.update()
        
        # Update AI game
        self.ai_move_timer += 1
        if self.ai_move_timer >= self.ai_move_interval and self.ai_action_queue:
            action_idx = self.ai_action_queue.pop(0)
            if not self.ai_env.done:
                self.ai_env.step(action_idx)
            self.ai_move_timer = 0
        
        # Check for game over
        if self.human_env.done and self.ai_env.done:
            self.ai_active = False
            self.state = GameState.GAME_OVER
        elif self.human_env.done or self.ai_env.done:
            # Wait a bit for dramatic effect, then end game
            if not hasattr(self, 'game_over_timer'):
                self.game_over_timer = 0
            self.game_over_timer += 1
            if self.game_over_timer > 120:  # 2 seconds at 60 FPS
                self.ai_active = False
                self.state = GameState.GAME_OVER

    def run(self):
        """Main game loop"""
        running = True
        
        while running:
            running = self.handle_events()
            
            if self.state == GameState.PLAYING:
                self.update_game()
            
            # Draw everything
            if self.state == GameState.MENU:
                self.draw_menu()
            elif self.state == GameState.PLAYING:
                self.draw_game()
            elif self.state == GameState.GAME_OVER:
                self.draw_game()  # Draw game state
                self.draw_game_over()  # Overlay game over screen
            
            pygame.display.flip()
            self.clock.tick(self.fps)
        
        # Cleanup
        self.ai_active = False
        if self.ai_thread and self.ai_thread.is_alive():
            self.ai_thread.join(timeout=1.0)
        
        pygame.quit()

def find_latest_checkpoint(checkpoint_dir="checkpoints"):
    """Find the most recent checkpoint"""
    if not os.path.exists(checkpoint_dir):
        return None
    
    files = [f for f in os.listdir(checkpoint_dir) 
             if f.startswith("placement_tetris_ep") and f.endswith(".pth")]
    
    if not files:
        return None
    
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
    # Find the latest checkpoint
    checkpoint_path = find_latest_checkpoint()
    
    if checkpoint_path is None:
        print("No checkpoint found! Please train an agent first.")
        exit(1)
    
    print(f"Using checkpoint: {checkpoint_path}")
    
    # Start the game
    game = TetrisVsAI(checkpoint_path)
    game.run()