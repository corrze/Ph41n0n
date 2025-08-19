# Official-style Tetris PvP with FIXED piece rotations

# Human vs AI
# python tetris_pvp.py --ai

# Human vs Human
# python tetris_pvp.py

# Specific checkpoint
# python tetris_pvp.py --checkpoint checkpoints_fixed/placement_tetris_fixed_ep10000.pth --ai

import os
import pygame
import numpy as np
import random
import time
import threading
from typing import Optional, Tuple, List
from enum import Enum
import argparse

import torch

# Import FIXED classes - try multiple sources
try:
    # Try to import from FIXED training file
    from tetris_dqn_placement import (
        TetrisEnv, PlacementDQNAgent, ALL_PLACEMENTS, PIECES, PIECE_NAMES, 
        PIECE_INDEX, BOARD_W, BOARD_H, fits, place_and_clear
    )
    print("✅ Using FIXED training classes for PvP")
except ImportError:
    # Fallback: Import base and override pieces
    print("⚠️  FIXED training file not found, using embedded fixed pieces for PvP")
    
    from tetris_dqn_placement import (
        TetrisEnv, PlacementDQNAgent, PIECE_NAMES, 
        PIECE_INDEX, BOARD_W, BOARD_H, fits, place_and_clear
    )

    # FIXED PIECE DEFINITIONS - EMBEDDED
    PIECES = {
        "I": [
            np.array([[1, 1, 1, 1]], dtype=np.uint8),  # Horizontal
            np.array([[1], [1], [1], [1]], dtype=np.uint8),  # Vertical
        ],
        "O": [
            np.array([[1, 1], [1, 1]], dtype=np.uint8),  # Only one rotation
        ],
        "T": [
            np.array([[0,1,0], [1,1,1]], dtype=np.uint8),  # T pointing up
            np.array([[1,0], [1,1], [1,0]], dtype=np.uint8),  # T pointing right  
            np.array([[1,1,1], [0,1,0]], dtype=np.uint8),  # T pointing down
            np.array([[0,1], [1,1], [0,1]], dtype=np.uint8),  # T pointing left
        ],
        "S": [
            np.array([[0,1,1], [1,1,0]], dtype=np.uint8),  # S horizontal
            np.array([[1,0], [1,1], [0,1]], dtype=np.uint8),  # S vertical
        ],
        "Z": [
            np.array([[1,1,0], [0,1,1]], dtype=np.uint8),  # Z horizontal
            np.array([[0,1], [1,1], [1,0]], dtype=np.uint8),  # Z vertical
        ],
        "J": [
            np.array([[1,0,0], [1,1,1]], dtype=np.uint8),  # J pointing up
            np.array([[1,1], [1,0], [1,0]], dtype=np.uint8),  # J pointing right
            np.array([[1,1,1], [0,0,1]], dtype=np.uint8),  # J pointing down
            np.array([[0,1], [0,1], [1,1]], dtype=np.uint8),  # J pointing left
        ],
        "L": [
            np.array([[0,0,1], [1,1,1]], dtype=np.uint8),  # L pointing up
            np.array([[1,0], [1,0], [1,1]], dtype=np.uint8),  # L pointing right
            np.array([[1,1,1], [1,0,0]], dtype=np.uint8),  # L pointing down
            np.array([[1,1], [0,1], [0,1]], dtype=np.uint8),  # L pointing left
        ],
    }
    
    # Regenerate ALL_PLACEMENTS with fixed pieces
    class PlacementAction:
        def __init__(self, piece_name: str, rotation: int, x: int):
            self.piece_name = piece_name
            self.rotation = rotation
            self.x = x
        
        def __repr__(self):
            return f"Place({self.piece_name}, rot={self.rotation}, x={self.x})"

    def generate_all_placements():
        all_actions = []
        for piece_name, rotations in PIECES.items():
            for rot_idx, shape in enumerate(rotations):
                shape_width = shape.shape[1]
                for x in range(BOARD_W - shape_width + 1):
                    all_actions.append(PlacementAction(piece_name, rot_idx, x))
        return all_actions

    ALL_PLACEMENTS = generate_all_placements()
    print(f"🔧 FIXED PvP: Using {len(ALL_PLACEMENTS)} placement actions")

# Verify piece rotations
print("🔍 VERIFYING FIXED PIECE ROTATIONS FOR PVP:")
expected_rotations = {"I": 2, "O": 1, "T": 4, "S": 2, "Z": 2, "J": 4, "L": 4}
for piece_name in PIECE_NAMES:
    actual = len(PIECES[piece_name])
    expected = expected_rotations[piece_name]
    status = "✅" if actual == expected else "❌"
    print(f"  {status} {piece_name}: {actual} rotations (expected {expected})")

class PvPToAIAdapter:
    """Adapter to make PvPTetrisEnv compatible with AI agent"""
    
    def __init__(self, pvp_env):
        self.pvp_env = pvp_env
        self.board = pvp_env.board
        self.cur_piece = pvp_env.cur_piece
        self.next_piece = pvp_env.next_piece
        self.done = pvp_env.done
        
    def get_state(self):
        """Convert PvP state to AI-expected format"""
        # One-hot encode pieces
        cur_piece_vec = np.zeros(7, dtype=np.float32)
        next_piece_vec = np.zeros(7, dtype=np.float32)
        
        if self.pvp_env.cur_piece:
            cur_piece_vec[PIECE_INDEX[self.pvp_env.cur_piece]] = 1.0
        if self.pvp_env.next_piece:
            next_piece_vec[PIECE_INDEX[self.pvp_env.next_piece]] = 1.0
            
        return {
            "board": self.pvp_env.board.astype(np.float32),
            "cur_piece": cur_piece_vec,
            "next_piece": next_piece_vec,
        }
    
    def get_valid_actions(self):
        """Get valid placement actions for current piece"""
        if self.pvp_env.done or self.pvp_env.cur_piece is None:
            return np.zeros(len(ALL_PLACEMENTS), dtype=np.float32)
        
        # Create action mask
        mask = np.zeros(len(ALL_PLACEMENTS), dtype=np.float32)
        for i, action in enumerate(ALL_PLACEMENTS):
            if action.piece_name == self.pvp_env.cur_piece:
                shape = PIECES[self.pvp_env.cur_piece][action.rotation]
                # Check if this placement is valid
                y = self._hard_drop_test(self.pvp_env.board, shape, action.x)
                if y is not None and y >= 0:
                    mask[i] = 1.0
        return mask
    
    def _hard_drop_test(self, board, shape, x):
        """Test where piece would land"""
        for y in range(BOARD_H):
            if not fits(board, shape, x, y):
                return y - 1 if y > 0 else None
        return BOARD_H - shape.shape[0]

class GameState(Enum):
    MENU = "menu"
    PLAYING = "playing"
    GAME_OVER = "game_over"

class PvPTetrisEnv:
    """Competitive Tetris environment with FIXED rotations and attack mechanics"""
    
    def __init__(self, player_name: str, seed: Optional[int] = None):
        self.player_name = player_name
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
        self.level = 1
        
        # PvP specific mechanics
        self.pending_garbage = []  # Lines to add to bottom
        self.garbage_delay = 0     # Delay before adding garbage
        self.combo_count = 0       # Current combo chain
        self.back_to_back = False  # T-spin/Tetris bonus
        
        # Drop timing
        self.drop_timer = 0
        self.drop_interval = max(5, 60 - self.level * 3)  # Gets faster with level
        self.lock_delay = 30       # Frames before piece locks
        self.lock_timer = 0
        self.move_reset_count = 0  # Number of moves that reset lock timer
        
        # Visual effects
        self.line_clear_animation = 0
        self.cleared_rows = []
        self.attack_lines_sent = 0

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
        self.lock_timer = 0
        self.move_reset_count = 0
        
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
        
        # Calculate attack lines to send
        attack_lines = self._calculate_attack(lines)
        
        # Update combo
        if lines > 0:
            self.combo_count += 1
            # Combo bonus increases attack
            combo_bonus = min(self.combo_count - 1, 4)  # Max +4 lines
            attack_lines += combo_bonus
        else:
            self.combo_count = 0
        
        # Update score with official Tetris scoring
        level_multiplier = self.level
        base_scores = {0: 0, 1: 100, 2: 300, 3: 500, 4: 800}
        self.score += base_scores.get(lines, 0) * level_multiplier
        
        # Combo bonus score
        if lines > 0 and self.combo_count > 1:
            combo_score = 50 * self.combo_count * level_multiplier
            self.score += combo_score
        
        self.total_lines += lines
        
        # Update level (every 10 lines)
        new_level = (self.total_lines // 10) + 1
        if new_level > self.level:
            self.level = new_level
            self.drop_interval = max(5, 60 - self.level * 3)
        
        # Store cleared rows for animation
        if lines > 0:
            self.cleared_rows = self._get_full_rows()
            self.line_clear_animation = 30  # 30 frames animation
        
        self._spawn_piece()
        return attack_lines

    def _get_full_rows(self):
        """Get rows that are about to be cleared"""
        return [i for i in range(BOARD_H) if np.all(self.board[i] == 1)]

    def _calculate_attack(self, lines_cleared: int) -> int:
        """Calculate how many attack lines to send based on lines cleared"""
        if lines_cleared == 0:
            return 0
        
        # Base attack values (official Tetris guidelines)
        base_attack = {
            1: 0,  # Single sends no attack in modern Tetris
            2: 1,  # Double sends 1 line
            3: 2,  # Triple sends 2 lines  
            4: 4   # Tetris sends 4 lines
        }
        
        attack = base_attack.get(lines_cleared, 0)
        
        # Back-to-back bonus (consecutive Tetrises or T-spins)
        if lines_cleared == 4:  # Tetris
            if self.back_to_back:
                attack += 1  # B2B Tetris sends +1 extra
            self.back_to_back = True
        else:
            self.back_to_back = False
        
        return attack

    def add_garbage_lines(self, count: int):
        """Add garbage lines to the bottom of the board"""
        if count <= 0:
            return
        
        # Add to pending garbage (will be added after current piece locks)
        self.pending_garbage.append(count)

    def _execute_garbage(self):
        """Actually add pending garbage lines to board"""
        total_garbage = sum(self.pending_garbage)
        if total_garbage == 0:
            return
        
        # Remove rows from top
        rows_to_remove = min(total_garbage, BOARD_H)
        self.board = self.board[rows_to_remove:]
        
        # Add garbage rows at bottom
        for _ in range(rows_to_remove):
            garbage_row = np.ones(BOARD_W, dtype=np.uint8)
            # Add one random hole in the garbage line
            hole_pos = self.rng.randint(0, BOARD_W - 1)
            garbage_row[hole_pos] = 0
            self.board = np.vstack([self.board, garbage_row])
        
        # Clear pending garbage
        self.pending_garbage = []
        
        # Check if game over after adding garbage
        if np.any(self.board[:4] != 0):  # Top 4 rows have blocks
            self.done = True

    def reset(self):
        self.board[:] = 0
        self.done = False
        self.bag = []
        self.next_piece = None
        self.total_lines = 0
        self.total_pieces = 0
        self.score = 0
        self.level = 1
        self.combo_count = 0
        self.back_to_back = False
        self.pending_garbage = []
        self.garbage_delay = 0
        self.drop_timer = 0
        self.lock_timer = 0
        self.line_clear_animation = 0
        self.cleared_rows = []
        self.attack_lines_sent = 0
        self.drop_interval = 57  # Starting speed
        self._spawn_piece()

    def update(self):
        """Update game state"""
        if self.done:
            return 0
        
        # Handle line clear animation
        if self.line_clear_animation > 0:
            self.line_clear_animation -= 1
            return 0
        
        # Add pending garbage after animation
        if self.pending_garbage and self.line_clear_animation == 0:
            self._execute_garbage()
        
        # Handle piece dropping
        self.drop_timer += 1
        piece_should_drop = self.drop_timer >= self.drop_interval
        
        if piece_should_drop:
            self.drop_timer = 0
            if self._piece_fits(self.piece_x, self.piece_y + 1, self.piece_rotation):
                self.piece_y += 1
                self.lock_timer = 0  # Reset lock timer when piece moves down
            else:
                # Piece hit bottom, start lock delay
                self.lock_timer += 1
        
        # Handle lock delay
        if not self._piece_fits(self.piece_x, self.piece_y + 1, self.piece_rotation):
            self.lock_timer += 1
            if self.lock_timer >= self.lock_delay:
                return self._place_piece()  # Return attack lines
        
        return 0

    def move_left(self):
        if (not self.done and self.line_clear_animation == 0 and 
            self._piece_fits(self.piece_x - 1, self.piece_y, self.piece_rotation)):
            self.piece_x -= 1
            self._reset_lock_timer()

    def move_right(self):
        if (not self.done and self.line_clear_animation == 0 and 
            self._piece_fits(self.piece_x + 1, self.piece_y, self.piece_rotation)):
            self.piece_x += 1
            self._reset_lock_timer()

    def rotate(self):
        """FIXED ROTATION with comprehensive wall kicks"""
        if self.done or self.line_clear_animation > 0 or self.cur_piece is None:
            return
        
        available_rotations = PIECES[self.cur_piece]
        if len(available_rotations) <= 1:
            return  # Can't rotate (O piece)
        
        current_rotation = self.piece_rotation
        new_rotation = (current_rotation + 1) % len(available_rotations)
        
        # Comprehensive wall kick tests - FIXED VERSION
        wall_kicks = [
            (0, 0),   # No movement
            (-1, 0), (1, 0), (-2, 0), (2, 0),  # Horizontal
            (0, -1), (0, -2), (0, -3),         # Up
            (-1, -1), (1, -1), (-2, -1), (2, -1),  # Diagonal up
            (-1, 1), (1, 1),                   # Diagonal down
            (0, 1),                            # Down
        ]
        
        # For I piece, add more horizontal tests
        if self.cur_piece == "I":
            wall_kicks.extend([(-3, 0), (3, 0), (-4, 0), (4, 0)])
        
        for dx, dy in wall_kicks:
            test_x = self.piece_x + dx
            test_y = self.piece_y + dy
            
            if self._piece_fits(test_x, test_y, new_rotation):
                self.piece_x = test_x
                self.piece_y = test_y
                self.piece_rotation = new_rotation
                self._reset_lock_timer()
                return

    def _reset_lock_timer(self):
        """Reset lock timer when piece moves (limited resets)"""
        if self.move_reset_count < 15:  # Limit infinite lock delay abuse
            self.lock_timer = 0
            self.move_reset_count += 1

    def soft_drop(self):
        if (not self.done and self.line_clear_animation == 0 and 
            self._piece_fits(self.piece_x, self.piece_y + 1, self.piece_rotation)):
            self.piece_y += 1
            self.score += 1
            self.lock_timer = 0

    def hard_drop(self):
        if self.done or self.line_clear_animation > 0:
            return 0
        
        drop_distance = 0
        while self._piece_fits(self.piece_x, self.piece_y + 1, self.piece_rotation):
            self.piece_y += 1
            drop_distance += 1
        
        self.score += drop_distance * 2
        return self._place_piece()  # Immediately place and return attack lines

    def get_board_with_piece(self):
        """Get board with current piece rendered"""
        if self.done or self.cur_piece is None:
            return self.board.copy()
            
        board_with_piece = self.board.copy()
        
        # Don't show piece during line clear animation
        if self.line_clear_animation > 0:
            return board_with_piece
        
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
        if self.done or self.cur_piece is None or self.line_clear_animation > 0:
            return None
        
        ghost_y = self.piece_y
        while self._piece_fits(self.piece_x, ghost_y + 1, self.piece_rotation):
            ghost_y += 1
        return (self.piece_x, ghost_y, self.piece_rotation)

    def get_danger_level(self):
        """Get how close to game over (0-1, where 1 is very dangerous)"""
        # Check height of highest column
        max_height = 0
        for c in range(BOARD_W):
            col = self.board[:, c]
            filled_rows = np.where(col != 0)[0]
            if len(filled_rows) > 0:
                height = BOARD_H - filled_rows[0]
                max_height = max(max_height, height)
        
        return min(max_height / BOARD_H, 1.0)

class TetrisPvPGame:
    """FIXED PvP Tetris game with proper piece rotations"""
    
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
        "GARBAGE": (128, 128, 128),    # Gray for garbage
        "DANGER": (255, 100, 100),     # Red for danger zone
        "CLEARED": (255, 255, 0),      # Yellow for cleared lines
    }
    
    def __init__(self, checkpoint_path: str = None):
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
        pygame.display.set_caption("FIXED Tetris PvP - Official Style")
        
        # Fonts
        self.font_large = pygame.font.SysFont("Arial", 32, bold=True)
        self.font_medium = pygame.font.SysFont("Arial", 20, bold=True)
        self.font_small = pygame.font.SysFont("Arial", 16)
        
        self.clock = pygame.time.Clock()
        
        # Game state
        self.state = GameState.MENU
        self.player1 = PvPTetrisEnv("Player 1")
        self.player2 = PvPTetrisEnv("Player 2")
        
        # AI agent (optional)
        self.ai_agent = None
        if checkpoint_path:
            self.ai_agent = self.load_ai_agent(checkpoint_path)
            self.player2.player_name = "FIXED AI"
        
        # Game timing
        self.fps = 60
        self.ai_move_timer = 0
        self.ai_move_interval = 20  # AI moves every 20 frames
        
        # UI positions
        self.p1_board_x = self.gap
        self.p2_board_x = self.p1_board_x + self.board_width + self.gap
        self.sidebar_x = self.p2_board_x + self.board_width + self.gap
        self.board_y = 50
        
        # Visual effects
        self.attack_effects = []  # Store attack animations

    def load_ai_agent(self, checkpoint_path: str):
        """Load AI agent if available"""
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}")
            return None
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        agent = PlacementDQNAgent(
            n_actions=len(ALL_PLACEMENTS),
            device=device,
            eps_start=0.0,
            eps_end=0.0,
            eps_decay=1
        )
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
        except:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        agent.q_net.load_state_dict(checkpoint['q_net'])
        agent.target_net.load_state_dict(checkpoint['target_net'])
        
        print(f"✅ FIXED AI agent loaded from {checkpoint_path}")
        return agent

    def draw_board(self, env: PvPTetrisEnv, offset_x: int, offset_y: int, is_p2: bool = False):
        """Draw a Tetris board with PvP features"""
        # Background
        board_rect = pygame.Rect(offset_x, offset_y, self.board_width, self.board_height)
        pygame.draw.rect(self.screen, (20, 20, 30), board_rect)
        
        # Danger zone indicator (top 4 rows)
        danger_level = env.get_danger_level()
        if danger_level > 0.7:  # High danger
            danger_rect = pygame.Rect(offset_x, offset_y, self.board_width, 4 * self.cell_size)
            danger_color = (255, int(100 * (1 - danger_level)), int(100 * (1 - danger_level)))
            pygame.draw.rect(self.screen, danger_color, danger_rect, 3)
        
        # Get board with piece
        board = env.get_board_with_piece()
        
        # Draw cells
        for y in range(BOARD_H):
            for x in range(BOARD_W):
                cell_value = board[y, x]
                
                if cell_value != 0:
                    rect = pygame.Rect(
                        offset_x + x * self.cell_size,
                        offset_y + y * self.cell_size,
                        self.cell_size, self.cell_size
                    )
                    
                    # Color based on cell type
                    if cell_value == 1:  # Placed piece
                        color = self.COLORS["PLACED"]
                    elif cell_value == 2:  # Falling piece
                        color = self.COLORS["FALLING"]
                    else:
                        color = self.COLORS["GARBAGE"]
                    
                    # Line clear animation effect
                    if env.line_clear_animation > 0 and y in env.cleared_rows:
                        flash_intensity = int(255 * (env.line_clear_animation / 30))
                        color = (flash_intensity, flash_intensity, 0)  # Yellow flash
                    
                    pygame.draw.rect(self.screen, color, rect)
                    pygame.draw.rect(self.screen, (50, 50, 50), rect, 1)
        
        # Draw ghost piece
        self.draw_ghost_piece(env, offset_x, offset_y)
        
        # Draw grid
        for x in range(BOARD_W + 1):
            pygame.draw.line(self.screen, (40, 40, 40),
                           (offset_x + x * self.cell_size, offset_y),
                           (offset_x + x * self.cell_size, offset_y + self.board_height))
        
        for y in range(BOARD_H + 1):
            pygame.draw.line(self.screen, (40, 40, 40),
                           (offset_x, offset_y + y * self.cell_size),
                           (offset_x + self.board_width, offset_y + y * self.cell_size))
        
        # Draw pending garbage indicator
        if env.pending_garbage:
            total_pending = sum(env.pending_garbage)
            warning_height = total_pending * self.cell_size
            warning_rect = pygame.Rect(
                offset_x - 5, 
                offset_y + self.board_height - warning_height,
                5, warning_height
            )
            pygame.draw.rect(self.screen, (255, 100, 100), warning_rect)

    def draw_ghost_piece(self, env: PvPTetrisEnv, offset_x: int, offset_y: int):
        """Draw ghost piece"""
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

    def draw_next_piece(self, piece_name: str, x: int, y: int):
        """Draw next piece preview"""
        if not piece_name or piece_name not in PIECES:
            return
        
        shape = PIECES[piece_name][0]
        color = self.COLORS[piece_name]
        
        # Center the piece
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
        title = self.font_large.render("FIXED TETRIS PvP", True, (100, 255, 100))
        title_rect = title.get_rect(center=(self.screen_width // 2, 100))
        self.screen.blit(title, title_rect)
        
        subtitle = self.font_medium.render("All Pieces Rotate Properly!", True, (200, 255, 200))
        subtitle_rect = subtitle.get_rect(center=(self.screen_width // 2, 140))
        self.screen.blit(subtitle, subtitle_rect)
        
        # Instructions
        instructions = [
            "PLAYER 1 (Left):",
            "A/D - Move Left/Right",
            "S - Soft Drop",
            "W - Rotate (FIXED!)", 
            "SPACE - Hard Drop",
            "",
            "PLAYER 2 (Right):",
            "←/→ - Move Left/Right",
            "↓ - Soft Drop",
            "↑ - Rotate (FIXED!)",
            "ENTER - Hard Drop",
            "",
            "COMPETITIVE RULES:",
            "• Clear lines to send attacks",
            "• 2 lines = 1 attack, 3 = 2, 4 = 4",
            "• Combos increase attack power",
            "• Garbage has random holes",
            "• I pieces rotate horizontally/vertically!",
            "• L pieces rotate through all 4 orientations!",
            "",
            "Press SPACE to Start!",
            "Press ESC to Quit"
        ]
        
        y = 180
        for instruction in instructions:
            if instruction.startswith("PLAYER") or instruction.startswith("COMPETITIVE"):
                color = (255, 255, 100)
                font = self.font_small
            elif "Press" in instruction:
                color = (100, 255, 100)
                font = self.font_medium
            elif "FIXED" in instruction or "rotate" in instruction:
                color = (100, 255, 100)  # Green for fixed features
                font = self.font_small
            else:
                color = (200, 200, 200)
                font = self.font_small
            
            text = font.render(instruction, True, color)
            text_rect = text.get_rect(center=(self.screen_width // 2, y))
            self.screen.blit(text, text_rect)
            y += 22

    def draw_game_over(self):
        """Draw game over screen"""
        # Semi-transparent overlay
        overlay = pygame.Surface((self.screen_width, self.screen_height))
        overlay.set_alpha(128)
        overlay.fill((0, 0, 0))
        self.screen.blit(overlay, (0, 0))
        
        # Determine winner
        if self.player1.done and not self.player2.done:
            winner_text = self.font_large.render(f"{self.player2.player_name} WINS!", True, (100, 255, 100))
        elif self.player2.done and not self.player1.done:
            winner_text = self.font_large.render(f"{self.player1.player_name} WINS!", True, (100, 255, 100))
        else:
            winner_text = self.font_large.render("TIE GAME!", True, (255, 255, 100))
        
        winner_rect = winner_text.get_rect(center=(self.screen_width // 2, 200))
        self.screen.blit(winner_text, winner_rect)
        
        # Final scores
        p1_score = self.font_small.render(f"{self.player1.player_name}: {self.player1.score} pts, {self.player1.total_lines} lines", True, (200, 200, 200))
        p2_score = self.font_small.render(f"{self.player2.player_name}: {self.player2.score} pts, {self.player2.total_lines} lines", True, (200, 200, 200))
        
        p1_rect = p1_score.get_rect(center=(self.screen_width // 2, 260))
        p2_rect = p2_score.get_rect(center=(self.screen_width // 2, 285))
        
        self.screen.blit(p1_score, p1_rect)
        self.screen.blit(p2_score, p2_rect)
        
        # Restart instruction
        restart_text = self.font_small.render("Press R to Restart or ESC to Quit", True, (255, 255, 100))
        restart_rect = restart_text.get_rect(center=(self.screen_width // 2, 350))
        self.screen.blit(restart_text, restart_rect)

    def draw_game(self):
        """Draw the main game"""
        self.screen.fill((40, 40, 60))
        
        # Draw both boards
        self.draw_board(self.player1, self.p1_board_x, self.board_y, False)
        self.draw_board(self.player2, self.p2_board_x, self.board_y, True)
        
        # Player labels
        p1_label = self.font_medium.render(self.player1.player_name, True, (255, 255, 255))
        p2_label = self.font_medium.render(self.player2.player_name, True, (100, 255, 100) if "AI" in self.player2.player_name else (255, 255, 255))
        
        p1_rect = p1_label.get_rect(center=(self.p1_board_x + self.board_width // 2, 20))
        p2_rect = p2_label.get_rect(center=(self.p2_board_x + self.board_width // 2, 20))
        
        self.screen.blit(p1_label, p1_rect)
        self.screen.blit(p2_label, p2_rect)
        
        # Sidebar stats
        y = self.board_y
        
        stats = [
            f"{self.player1.player_name.upper()}:",
            f"Score: {self.player1.score:,}",
            f"Lines: {self.player1.total_lines}",
            f"Level: {self.player1.level}",
            f"Combo: {self.player1.combo_count}",
            "",
            f"{self.player2.player_name.upper()}:",
            f"Score: {self.player2.score:,}",
            f"Lines: {self.player2.total_lines}", 
            f"Level: {self.player2.level}",
            f"Combo: {self.player2.combo_count}",
            "",
            "ROTATIONS: FIXED ✅",
            "NEXT PIECES:"
        ]
        
        for stat in stats:
            if stat.endswith(":"):
                color = (255, 255, 100)
            elif "FIXED" in stat:
                color = (100, 255, 100)
            else:
                color = (200, 200, 200)
            text = self.font_small.render(stat, True, color)
            self.screen.blit(text, (self.sidebar_x, y))
            y += 22
        
        # Next piece previews
        if self.player1.next_piece:
            text = self.font_small.render("P1:", True, (200, 200, 200))
            self.screen.blit(text, (self.sidebar_x, y))
            self.draw_next_piece(self.player1.next_piece, self.sidebar_x, y + 20)
            y += 90
        
        if self.player2.next_piece:
            text = self.font_small.render("P2:", True, (200, 200, 200))
            self.screen.blit(text, (self.sidebar_x, y))
            self.draw_next_piece(self.player2.next_piece, self.sidebar_x, y + 20)

    def start_game(self):
        """Start a new game"""
        self.player1.reset()
        self.player2.reset()
        self.state = GameState.PLAYING
        self.attack_effects = []

    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                
                if self.state == GameState.MENU:
                    if event.key == pygame.K_SPACE:
                        self.start_game()
                
                elif self.state == GameState.PLAYING:
                    # Player 1 controls (WASD + Space)
                    if event.key == pygame.K_a:
                        self.player1.move_left()
                    elif event.key == pygame.K_d:
                        self.player1.move_right()
                    elif event.key == pygame.K_w:
                        print("🔄 Player 1 rotating...")
                        self.player1.rotate()
                    elif event.key == pygame.K_s:
                        self.player1.soft_drop()
                    elif event.key == pygame.K_SPACE:
                        attack_lines = self.player1.hard_drop()
                        if attack_lines > 0:
                            self.player2.add_garbage_lines(attack_lines)
                            self.add_attack_effect(1, attack_lines)
                    
                    # Player 2 controls (Arrows + Enter) - only if no AI
                    if not self.ai_agent:
                        if event.key == pygame.K_LEFT:
                            self.player2.move_left()
                        elif event.key == pygame.K_RIGHT:
                            self.player2.move_right()
                        elif event.key == pygame.K_UP:
                            print("🔄 Player 2 rotating...")
                            self.player2.rotate()
                        elif event.key == pygame.K_DOWN:
                            self.player2.soft_drop()
                        elif event.key == pygame.K_RETURN:
                            attack_lines = self.player2.hard_drop()
                            if attack_lines > 0:
                                self.player1.add_garbage_lines(attack_lines)
                                self.add_attack_effect(2, attack_lines)
                
                elif self.state == GameState.GAME_OVER:
                    if event.key == pygame.K_r:
                        self.start_game()
        
        return True

    def add_attack_effect(self, player: int, lines: int):
        """Add visual effect for attack lines sent"""
        effect = {
            'player': player,
            'lines': lines,
            'timer': 60,  # Show for 1 second
            'x': self.p1_board_x if player == 1 else self.p2_board_x,
            'y': self.board_y + self.board_height // 2
        }
        self.attack_effects.append(effect)

    def update_attack_effects(self):
        """Update attack visual effects"""
        for effect in self.attack_effects[:]:
            effect['timer'] -= 1
            if effect['timer'] <= 0:
                self.attack_effects.remove(effect)

    def draw_attack_effects(self):
        """Draw attack line effects"""
        for effect in self.attack_effects:
            alpha = int(255 * (effect['timer'] / 60))
            
            # Create text
            text = f"+{effect['lines']} ATTACK!"
            font_surface = self.font_medium.render(text, True, (255, 100, 100))
            
            # Create alpha surface
            alpha_surface = pygame.Surface(font_surface.get_size())
            alpha_surface.set_alpha(alpha)
            alpha_surface.blit(font_surface, (0, 0))
            
            # Animate position (move up)
            y_offset = (60 - effect['timer']) * 2
            self.screen.blit(alpha_surface, (effect['x'], effect['y'] - y_offset))

    def update_ai(self):
        """Update AI player with FIXED compatibility"""
        if not self.ai_agent or self.player2.done:
            return
        
        self.ai_move_timer += 1
        if self.ai_move_timer >= self.ai_move_interval:
            self.ai_move_timer = 0
            
            # Create adapter for AI compatibility
            ai_adapter = PvPToAIAdapter(self.player2)
            
            if not ai_adapter.done and self.player2.cur_piece:
                try:
                    # Get AI action using adapter
                    state = ai_adapter.get_state()
                    mask = ai_adapter.get_valid_actions()
                    action_idx = self.ai_agent._get_best_action(state, mask)
                    
                    # Convert placement action to PvP moves
                    if action_idx < len(ALL_PLACEMENTS):
                        action = ALL_PLACEMENTS[action_idx]
                        if action.piece_name == self.player2.cur_piece:
                            self.execute_ai_placement(action)
                            
                except Exception as e:
                    print(f"AI error: {e}")
                    # Fallback: just hard drop
                    attack_lines = self.player2.hard_drop()
                    if attack_lines > 0:
                        self.player1.add_garbage_lines(attack_lines)
                        self.add_attack_effect(2, attack_lines)

    def execute_ai_placement(self, action):
        """Execute the AI's chosen placement"""
        target_rotation = action.rotation
        target_x = action.x
        
        # Rotate to target rotation
        current_rotation = self.player2.piece_rotation
        rotations_needed = (target_rotation - current_rotation) % len(PIECES[self.player2.cur_piece])
        
        for _ in range(rotations_needed):
            self.player2.rotate()
        
        # Move to target x position
        current_x = self.player2.piece_x
        while current_x < target_x:
            self.player2.move_right()
            current_x += 1
        while current_x > target_x:
            self.player2.move_left()
            current_x -= 1
        
        # Hard drop
        attack_lines = self.player2.hard_drop()
        if attack_lines > 0:
            self.player1.add_garbage_lines(attack_lines)
            self.add_attack_effect(2, attack_lines)
            
        # Log AI rotation usage
        if action.rotation > 0:
            print(f"🔄 AI used rotation {action.rotation} for {action.piece_name} piece!")

    def update_game(self):
        """Update game state"""
        if self.state != GameState.PLAYING:
            return
        
        # Update both players
        attack1 = self.player1.update()
        attack2 = self.player2.update()
        
        # Handle attacks
        if attack1 > 0:
            self.player2.add_garbage_lines(attack1)
            self.add_attack_effect(1, attack1)
        
        if attack2 > 0:
            self.player1.add_garbage_lines(attack2)
            self.add_attack_effect(2, attack2)
        
        # Update AI
        if self.ai_agent:
            self.update_ai()
        
        # Update visual effects
        self.update_attack_effects()
        
        # Check for game over
        if self.player1.done or self.player2.done:
            self.state = GameState.GAME_OVER

    def run(self):
        """Main game loop"""
        running = True
        
        print("🎮 FIXED TETRIS PvP - All pieces rotate properly!")
        if self.ai_agent:
            print("🤖 Playing against FIXED AI agent")
        else:
            print("👥 Playing human vs human")
        print("🔧 I pieces rotate horizontally/vertically")
        print("🔧 L pieces rotate through all 4 orientations")
        
        while running:
            running = self.handle_events()
            
            if self.state == GameState.PLAYING:
                self.update_game()
            
            # Draw everything
            if self.state == GameState.MENU:
                self.draw_menu()
            elif self.state == GameState.PLAYING:
                self.draw_game()
                self.draw_attack_effects()
            elif self.state == GameState.GAME_OVER:
                self.draw_game()
                self.draw_attack_effects()
                self.draw_game_over()
            
            pygame.display.flip()
            self.clock.tick(self.fps)
        
        pygame.quit()

def find_latest_checkpoint(checkpoint_dir="checkpoints_fixed", prefix="placement_tetris_fixed_ep"):
    """Find the most recent FIXED checkpoint"""
    if not os.path.exists(checkpoint_dir):
        return find_latest_old_checkpoint()
    
    files = [f for f in os.listdir(checkpoint_dir) 
             if f.startswith(prefix) and f.endswith(".pth")]
    
    if not files:
        return find_latest_old_checkpoint()
    
    episodes = []
    for f in files:
        try:
            ep_str = f.replace(prefix, "").replace(".pth", "")
            episodes.append((int(ep_str), f))
        except ValueError:
            continue
    
    if episodes:
        latest_ep, latest_file = max(episodes)
        return os.path.join(checkpoint_dir, latest_file)
    
    return find_latest_old_checkpoint()

def find_latest_old_checkpoint():
    """Fallback to old checkpoints"""
    checkpoint_dir = "checkpoints"
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
    parser = argparse.ArgumentParser(description="Play FIXED PvP Tetris")
    parser.add_argument("--ai", action="store_true", help="Play against AI (uses latest checkpoint)")
    parser.add_argument("--checkpoint", type=str, help="Specific AI checkpoint to use")
    
    args = parser.parse_args()
    
    checkpoint_path = None
    if args.ai:
        if args.checkpoint:
            checkpoint_path = args.checkpoint
        else:
            checkpoint_path = find_latest_checkpoint()
            if checkpoint_path is None:
                print("❌ No checkpoint found! Playing human vs human.")
            else:
                if "checkpoints_fixed" in checkpoint_path:
                    print(f"✅ Using FIXED AI checkpoint: {checkpoint_path}")
                else:
                    print(f"⚠️  Using OLD AI checkpoint: {checkpoint_path}")
                    print("   Note: AI may have broken rotations!")
    
    # Start the game
    print("\n🚀 Starting FIXED Tetris PvP!")
    game = TetrisPvPGame(checkpoint_path)
    game.run()# tetris_pvp_FIXED.py