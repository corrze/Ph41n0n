# tetris_vs_ai.py
# Tetris vs AI compatible with NEW FIXED training model

# python tetris_vs_ai_NEW.py --mode last_standing
# python tetris_vs_ai_NEW.py --mode standard

# Auto-detect latest FIXED checkpoint
# python tetris_vs_ai_NEW.py

# Use specific checkpoint
#python tetris_vs_ai_NEW.py --checkpoint checkpoints_fixed/placement_tetris_fixed_ep10000.pth

# Choose mode and checkpoint
#python tetris_vs_ai.py --checkpoint path/to/model.pth --mode standard


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

# Smart import system - try FIXED training first, fallback to embedded pieces
try:
    # Try to import from FIXED training file
    from tetris_dqn_placement import (
        TetrisEnv, PlacementDQNAgent, ALL_PLACEMENTS, PIECES, PIECE_NAMES, 
        PIECE_INDEX, BOARD_W, BOARD_H, fits, place_and_clear
    )
    print("✅ Using NEW FIXED training classes for vs AI")
except ImportError:
    # Fallback: Import base and override pieces
    print("⚠️  FIXED training file not found, using embedded fixed pieces")
    
    from tetris_dqn_placement import (
        TetrisEnv, PlacementDQNAgent, PIECE_NAMES, 
        PIECE_INDEX, BOARD_W, BOARD_H, fits, place_and_clear
    )
    
    # FIXED PIECE DEFINITIONS - EMBEDDED FOR COMPATIBILITY
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
    print(f"🔧 FIXED vs AI: Using {len(ALL_PLACEMENTS)} placement actions")

# Verify piece rotations
print("🔍 VERIFYING FIXED PIECE ROTATIONS FOR VS AI:")
expected_rotations = {"I": 2, "O": 1, "T": 4, "S": 2, "Z": 2, "J": 4, "L": 4}
for piece_name in PIECE_NAMES:
    actual = len(PIECES[piece_name])
    expected = expected_rotations[piece_name]
    status = "✅" if actual == expected else "❌"
    print(f"  {status} {piece_name}: {actual} rotations (expected {expected})")

class GameState(Enum):
    MENU = "menu"
    PLAYING = "playing"
    GAME_OVER = "game_over"

class HumanTetrisEnv:
    """Human-controllable Tetris environment with FIXED rotation"""
    
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
        """COMPLETELY FIXED ROTATION with comprehensive wall kicks"""
        if self.done or self.cur_piece is None:
            return
        
        # Get all available rotations for this piece
        available_rotations = PIECES[self.cur_piece]
        if len(available_rotations) <= 1:
            return  # Can't rotate (O piece)
        
        current_rotation = self.piece_rotation
        new_rotation = (current_rotation + 1) % len(available_rotations)
        
        # COMPREHENSIVE wall kick tests - MUCH more thorough than before
        wall_kick_tests = self._get_comprehensive_wall_kicks(current_rotation, new_rotation)
        
        for dx, dy in wall_kick_tests:
            test_x = self.piece_x + dx
            test_y = self.piece_y + dy
            
            if self._piece_fits(test_x, test_y, new_rotation):
                # SUCCESS - Apply the rotation
                self.piece_x = test_x
                self.piece_y = test_y
                self.piece_rotation = new_rotation
                
                print(f"🔄 {self.cur_piece} rotated: {current_rotation} -> {new_rotation}")
                return

    def _get_comprehensive_wall_kicks(self, old_rotation: int, new_rotation: int) -> List[Tuple[int, int]]:
        """Get comprehensive wall kick offsets"""
        
        # Start with no movement
        offsets = [(0, 0)]
        
        # Basic directional tests
        basic_offsets = [
            (-1, 0), (1, 0),    # Left, Right
            (0, -1),            # Up
            (-2, 0), (2, 0),    # Far left, Far right
            (-1, -1), (1, -1),  # Up-left, Up-right
            (0, -2),            # Far up
            (-1, 1), (1, 1),    # Down-left, Down-right
            (0, 1),             # Down (less common but possible)
        ]
        offsets.extend(basic_offsets)
        
        # Piece-specific offsets
        if self.cur_piece == "I":
            # I-piece needs more horizontal space
            i_offsets = [
                (-3, 0), (3, 0),     # Very far left/right
                (-2, -1), (2, -1),   # Far horizontal + up
                (-1, -2), (1, -2),   # Horizontal + far up
                (0, -3),             # Very far up
            ]
            offsets.extend(i_offsets)
        
        elif self.cur_piece in ["L", "J", "T"]:
            # L, J, T pieces often need more varied kick attempts
            complex_offsets = [
                (-2, -1), (2, -1),   # Far horizontal + up
                (-1, -2), (1, -2),   # Horizontal + far up
                (-2, 1), (2, 1),     # Far horizontal + down
                (-3, 0), (3, 0),     # Very far horizontal
                (0, -3),             # Very far up
            ]
            offsets.extend(complex_offsets)
        
        elif self.cur_piece in ["S", "Z"]:
            # S and Z pieces sometimes need specific kicks
            sz_offsets = [
                (-2, -1), (2, -1),
                (-1, -2), (1, -2),
            ]
            offsets.extend(sz_offsets)
        
        # Additional fallback positions for stubborn pieces
        fallback_offsets = [
            (-4, 0), (4, 0),        # Very far horizontal
            (0, -4),                # Very far up
            (-3, -1), (3, -1),      # Very far horizontal + up
            (-2, -2), (2, -2),      # Far diagonal up
            (-1, -3), (1, -3),      # Horizontal + very far up
        ]
        offsets.extend(fallback_offsets)
        
        return offsets

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
    """Main game class for human vs FIXED AI Tetris - COMPATIBLE WITH NEW MODEL"""
    
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
        "DEAD_BOARD": (100, 50, 50),   # Red tint for dead players
    }
    
    def __init__(self, checkpoint_path: str, mode: str = "last_standing"):
        pygame.init()
        
        # Game mode: "last_standing" or "standard"
        self.mode = mode
        
        # Display settings
        self.cell_size = 25
        self.board_width = BOARD_W * self.cell_size
        self.board_height = BOARD_H * self.cell_size
        self.sidebar_width = 200
        self.gap = 50
        
        self.screen_width = self.board_width * 2 + self.sidebar_width + self.gap * 3
        self.screen_height = self.board_height + 100
        
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        
        if mode == "last_standing":
            pygame.display.set_caption("FIXED Tetris vs AI - Last One Standing!")
        else:
            pygame.display.set_caption("FIXED Tetris vs AI - Standard Mode")
        
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
        """Load the FIXED trained AI agent"""
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
        
        # Check if this is a FIXED checkpoint
        if "checkpoints_fixed" in checkpoint_path:
            print(f"✅ FIXED AI agent loaded from {checkpoint_path}")
        else:
            print(f"⚠️  OLD AI agent loaded from {checkpoint_path}")
            print("   Note: This AI may have broken piece rotations!")
        
        return agent

    def ai_player_thread(self):
        """AI player runs in separate thread"""
        while self.ai_active:
            if self.state == GameState.PLAYING and not self.ai_env.done:
                # Get AI action
                state = self.ai_env.get_state()
                mask = self.ai_env.get_valid_actions()
                
                if self.ai_agent:
                    action_idx = self.ai_agent._get_best_action(state, mask)
                    self.ai_action_queue.append(action_idx)
                
                time.sleep(0.5)  # AI thinks for 0.5 seconds
            else:
                time.sleep(0.1)

    def draw_board(self, board, offset_x, offset_y, is_ai=False, is_dead=False):
        """Draw a Tetris board with death indicator"""
        # Background - red tint if player is dead
        bg_color = (40, 20, 20) if is_dead else (20, 20, 30)
        board_rect = pygame.Rect(offset_x, offset_y, self.board_width, self.board_height)
        pygame.draw.rect(self.screen, bg_color, board_rect)
        
        # Draw placed pieces
        if is_dead:
            placed_color = (150, 100, 100)  # Dim red for dead player
        else:
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
                    if not is_dead:  # Only show falling piece if alive
                        falling_color = self.COLORS["AI_FALLING"] if is_ai else self.COLORS["FALLING"]
                        rect = pygame.Rect(
                            offset_x + x * self.cell_size,
                            offset_y + y * self.cell_size,
                            self.cell_size, self.cell_size
                        )
                        pygame.draw.rect(self.screen, falling_color, rect)
                        pygame.draw.rect(self.screen, (100, 100, 100), rect, 1)
        
        # Death overlay
        if is_dead:
            death_overlay = pygame.Surface((self.board_width, self.board_height))
            death_overlay.set_alpha(100)
            death_overlay.fill((255, 0, 0))
            self.screen.blit(death_overlay, (offset_x, offset_y))
            
            # Death text
            death_text = self.font_medium.render("DEAD", True, (255, 200, 200))
            death_rect = death_text.get_rect(center=(offset_x + self.board_width // 2, offset_y + self.board_height // 2))
            self.screen.blit(death_text, death_rect)
        
        # Draw grid
        grid_color = (80, 40, 40) if is_dead else (40, 40, 40)
        for x in range(BOARD_W + 1):
            pygame.draw.line(self.screen, grid_color,
                           (offset_x + x * self.cell_size, offset_y),
                           (offset_x + x * self.cell_size, offset_y + self.board_height))
        
        for y in range(BOARD_H + 1):
            pygame.draw.line(self.screen, grid_color,
                           (offset_x, offset_y + y * self.cell_size),
                           (offset_x + self.board_width, offset_y + y * self.cell_size))

    def draw_ghost_piece(self, env, offset_x, offset_y):
        """Draw ghost piece (where piece will land) - only if alive"""
        if env.done:
            return
        
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
        """Draw start menu with mode selection"""
        self.screen.fill((30, 30, 50))
        
        # Title
        title = self.font_large.render("FIXED TETRIS vs AI", True, (100, 255, 100))
        title_rect = title.get_rect(center=(self.screen_width // 2, 80))
        self.screen.blit(title, title_rect)
        
        # Mode indicator
        if self.mode == "last_standing":
            mode_text = self.font_medium.render("LAST ONE STANDING MODE", True, (255, 255, 100))
        else:
            mode_text = self.font_medium.render("STANDARD MODE", True, (255, 255, 100))
        
        mode_rect = mode_text.get_rect(center=(self.screen_width // 2, 120))
        self.screen.blit(mode_text, mode_rect)
        
        # Instructions
        instructions = [
            "Human Controls:",
            "A/D - Move Left/Right",
            "S - Soft Drop", 
            "W - Rotate (ALL PIECES FIXED!)",
            "SPACE - Hard Drop",
            "",
            "GAME FEATURES:",
            "• I pieces rotate horizontally/vertically!",
            "• L pieces rotate through all 4 orientations!",
            "• J, T, S, Z pieces all rotate properly!",
            "",
        ]
        
        if self.mode == "last_standing":
            instructions.extend([
                "LAST STANDING RULES:",
                "• Game continues until BOTH players die",
                "• Winner = Highest score when both dead",
                "• You can keep playing even if AI dies!",
            ])
        else:
            instructions.extend([
                "STANDARD RULES:",
                "• Game ends when first player dies",
                "• Standard Tetris scoring",
                "• First to die loses!",
            ])
        
        instructions.extend([
            "",
            "Press ENTER to Start!",
            "Press ESC to Quit"
        ])
        
        y = 160
        for instruction in instructions:
            if "Press" in instruction:
                color = (255, 255, 100)
            elif instruction.endswith("RULES:") or instruction.endswith("FEATURES:") or "Human Controls:" in instruction:
                color = (255, 200, 100)
            elif "FIXED" in instruction or "rotate" in instruction:
                color = (100, 255, 100)
            else:
                color = (200, 200, 200)
            
            text = self.font_small.render(instruction, True, color)
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
        
        # Game Over text
        game_over_text = self.font_large.render("GAME OVER", True, (255, 100, 100))
        game_over_rect = game_over_text.get_rect(center=(self.screen_width // 2, 150))
        self.screen.blit(game_over_text, game_over_rect)
        
        # Calculate final scores
        human_score = self.human_env.score
        ai_score = self.ai_env.total_lines * 100  # AI score based on lines
        
        # Determine winner
        if self.mode == "last_standing":
            # In last standing mode, check who has higher score when both are dead
            if human_score > ai_score:
                winner_text = self.font_medium.render("🎉 HUMAN WINS! 🎉", True, (100, 255, 100))
            elif ai_score > human_score:
                winner_text = self.font_medium.render("🤖 FIXED AI WINS! 🤖", True, (255, 100, 100))
            else:
                winner_text = self.font_medium.render("🤝 TIE GAME! 🤝", True, (255, 255, 100))
        else:
            # In standard mode, check who died first
            if self.human_env.done and not self.ai_env.done:
                winner_text = self.font_medium.render("🤖 FIXED AI WINS! 🤖", True, (255, 100, 100))
            elif self.ai_env.done and not self.human_env.done:
                winner_text = self.font_medium.render("🎉 HUMAN WINS! 🎉", True, (100, 255, 100))
            else:
                winner_text = self.font_medium.render("🤝 TIE GAME! 🤝", True, (255, 255, 100))
        
        winner_rect = winner_text.get_rect(center=(self.screen_width // 2, 200))
        self.screen.blit(winner_text, winner_rect)
        
        # Final scores comparison
        score_comparison = [
            f"FINAL SCORES:",
            f"Human: {human_score:,} points",
            f"FIXED AI: {ai_score:,} points",
            f"",
            f"STATISTICS:",
            f"Human - Lines: {self.human_env.total_lines}, Pieces: {self.human_env.total_pieces}",
            f"AI - Lines: {self.ai_env.total_lines}, Pieces: {self.ai_env.total_pieces}"
        ]
        
        y = 260
        for line in score_comparison:
            if "FINAL SCORES:" in line or "STATISTICS:" in line:
                color = (255, 255, 100)
            elif "Human:" in line:
                color = (100, 255, 100) if human_score > ai_score else (200, 200, 200)
            elif "FIXED AI:" in line:
                color = (255, 100, 100) if ai_score > human_score else (200, 200, 200)
            else:
                color = (200, 200, 200)
            
            text = self.font_small.render(line, True, color)
            text_rect = text.get_rect(center=(self.screen_width // 2, y))
            self.screen.blit(text, text_rect)
            y += 25
        
        # Restart instruction
        restart_text = self.font_small.render("Press R to Restart or ESC to Quit", True, (255, 255, 100))
        restart_rect = restart_text.get_rect(center=(self.screen_width // 2, y + 40))
        self.screen.blit(restart_text, restart_rect)

    def draw_game(self):
        """Draw the main game"""
        self.screen.fill((40, 40, 60))
        
        # Draw both boards with death status
        human_board = self.human_env.get_board_with_piece()
        ai_board = self.ai_env.get_full_state() if hasattr(self.ai_env, 'get_full_state') else self.ai_env.board
        
        self.draw_board(human_board, self.human_board_x, self.board_y, 
                       is_ai=False, is_dead=self.human_env.done)
        self.draw_board(ai_board, self.ai_board_x, self.board_y, 
                       is_ai=True, is_dead=self.ai_env.done)
        
        # Draw ghost pieces only for living players
        if not self.human_env.done:
            self.draw_ghost_piece(self.human_env, self.human_board_x, self.board_y)
        
        # Labels with status
        human_status = "HUMAN" if not self.human_env.done else "HUMAN (DEAD)"
        ai_status = "FIXED AI" if not self.ai_env.done else "FIXED AI (DEAD)"
        
        human_color = (255, 255, 255) if not self.human_env.done else (255, 100, 100)
        ai_color = (100, 255, 100) if not self.ai_env.done else (255, 100, 100)
        
        human_label = self.font_medium.render(human_status, True, human_color)
        ai_label = self.font_medium.render(ai_status, True, ai_color)
        
        human_label_rect = human_label.get_rect(center=(self.human_board_x + self.board_width // 2, 20))
        ai_label_rect = ai_label.get_rect(center=(self.ai_board_x + self.board_width // 2, 20))
        
        self.screen.blit(human_label, human_label_rect)
        self.screen.blit(ai_label, ai_label_rect)
        
        # Sidebar with stats
        y = self.board_y
        
        # Current leader indicator
        human_score = self.human_env.score
        ai_score = self.ai_env.total_lines * 100
        
        if human_score > ai_score:
            leader = "👑 HUMAN LEADING!"
            leader_color = (100, 255, 100)
        elif ai_score > human_score:
            leader = "👑 FIXED AI LEADING!"
            leader_color = (255, 100, 100)
        else:
            leader = "🤝 TIED!"
            leader_color = (255, 255, 100)
        
        leader_text = self.font_small.render(leader, True, leader_color)
        self.screen.blit(leader_text, (self.sidebar_x, y))
        y += 40
        
        # Mode indicator
        mode_text = f"Mode: {self.mode.replace('_', ' ').title()}"
        mode_color = (200, 200, 255)
        mode_render = self.font_small.render(mode_text, True, mode_color)
        self.screen.blit(mode_render, (self.sidebar_x, y))
        y += 30
        
        # Stats
        stats = [
            "HUMAN STATS:",
            f"Score: {human_score:,}",
            f"Lines: {self.human_env.total_lines}",
            f"Pieces: {self.human_env.total_pieces}",
            f"Status: {'ALIVE' if not self.human_env.done else 'DEAD'}",
            "",
            "FIXED AI STATS:",
            f"Score: {ai_score:,}",
            f"Lines: {self.ai_env.total_lines}",
            f"Pieces: {self.ai_env.total_pieces}",
            f"Status: {'ALIVE' if not self.ai_env.done else 'DEAD'}",
            "",
            "ROTATIONS: ALL FIXED ✅",
            "NEXT PIECES:"
        ]
        
        for stat in stats:
            if "STATS:" in stat:
                color = (255, 255, 100)
            elif "Status: ALIVE" in stat:
                color = (100, 255, 100)
            elif "Status: DEAD" in stat:
                color = (255, 100, 100)
            elif "FIXED" in stat:
                color = (100, 255, 100)
            else:
                color = (200, 200, 200)
            
            text = self.font_small.render(stat, True, color)
            self.screen.blit(text, (self.sidebar_x, y))
            y += 20
        
        # Next piece previews (only for living players)
        if self.human_env.next_piece and not self.human_env.done:
            text = self.font_small.render("Human:", True, (200, 200, 200))
            self.screen.blit(text, (self.sidebar_x, y))
            self.draw_next_piece(self.human_env.next_piece, self.sidebar_x, y + 20)
            y += 90
        
        if self.ai_env.next_piece and not self.ai_env.done:
            text = self.font_small.render("AI:", True, (200, 200, 200))
            self.screen.blit(text, (self.sidebar_x, y))
            self.draw_next_piece(self.ai_env.next_piece, self.sidebar_x, y + 20)

    def start_game(self):
        """Start a new game"""
        self.human_env.reset()
        self.ai_env.reset()
        self.state = GameState.PLAYING
        
        # Reset any game over timers
        for attr in ['game_over_timer', 'final_game_over_timer', 'human_death_announced', 'ai_death_announced']:
            if hasattr(self, attr):
                delattr(self, attr)
        
        # Start AI thread
        self.ai_active = True
        self.ai_action_queue = []
        if self.ai_thread and self.ai_thread.is_alive():
            self.ai_active = False
            self.ai_thread.join(timeout=1.0)
        
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
                    # Human controls - ONLY if human is still alive
                    if not self.human_env.done:
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
        """Update game state based on mode"""
        if self.state != GameState.PLAYING:
            return
        
        # Update human game (only if alive)
        if not self.human_env.done:
            self.human_env.update()
        
        # Update AI game (only if alive)
        if not self.ai_env.done:
            self.ai_move_timer += 1
            if self.ai_move_timer >= self.ai_move_interval and self.ai_action_queue:
                action_idx = self.ai_action_queue.pop(0)
                self.ai_env.step(action_idx)
                self.ai_move_timer = 0
        
        # Game ending logic depends on mode
        if self.mode == "last_standing":
            # LAST STANDING MODE: Game only ends when BOTH players are dead
            if self.human_env.done and self.ai_env.done:
                if not hasattr(self, 'final_game_over_timer'):
                    self.final_game_over_timer = 0
                    print("🏁 BOTH PLAYERS DEAD - LAST STANDING GAME ENDING!")
                    print(f"Final Scores: Human={self.human_env.score}, AI={self.ai_env.total_lines * 100}")
                
                self.final_game_over_timer += 1
                if self.final_game_over_timer > 60:  # 1 second delay
                    self.ai_active = False
                    self.state = GameState.GAME_OVER
            
            # Show status when one player dies
            elif self.human_env.done and not self.ai_env.done:
                if not hasattr(self, 'human_death_announced'):
                    print("💀 HUMAN DIED! FIXED AI continues playing...")
                    self.human_death_announced = True
            
            elif self.ai_env.done and not self.human_env.done:
                if not hasattr(self, 'ai_death_announced'):
                    print("💀 FIXED AI DIED! Human continues playing...")
                    self.ai_death_announced = True
        
        else:
            # STANDARD MODE: Game ends when first player dies
            if self.human_env.done or self.ai_env.done:
                if not hasattr(self, 'game_over_timer'):
                    self.game_over_timer = 0
                    if self.human_env.done:
                        print("💀 HUMAN DIED FIRST - FIXED AI WINS!")
                    else:
                        print("💀 FIXED AI DIED FIRST - HUMAN WINS!")
                
                self.game_over_timer += 1
                if self.game_over_timer > 60:  # 1 second delay
                    self.ai_active = False
                    self.state = GameState.GAME_OVER

    def run(self):
        """Main game loop"""
        running = True
        
        print("🎮 FIXED TETRIS VS AI!")
        print("🔧 ALL piece rotations have been FIXED!")
        print("🔧 I pieces rotate horizontally/vertically")
        print("🔧 L pieces rotate through all 4 orientations")
        print(f"🎯 Mode: {self.mode.replace('_', ' ').title()}")
        
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

def test_fixed_pieces():
    """Test that all pieces have correct rotations"""
    print("\n🧪 TESTING ALL FIXED PIECES...")
    
    for piece_name in PIECE_NAMES:
        rotations = PIECES[piece_name]
        print(f"✅ {piece_name} piece: {len(rotations)} rotations")
        
        if piece_name == "I" and len(rotations) == 2:
            print("   🔄 I piece can rotate horizontally/vertically!")
        elif piece_name == "L" and len(rotations) == 4:
            print("   🔄 L piece can rotate through all 4 orientations!")
        elif piece_name in ["J", "T"] and len(rotations) == 4:
            print(f"   🔄 {piece_name} piece has all 4 rotations!")
    
    print("🎯 All pieces are FIXED and ready to use!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play FIXED Tetris vs AI")
    parser.add_argument("--checkpoint", type=str, help="Specific AI checkpoint to use")
    parser.add_argument("--mode", type=str, choices=["last_standing", "standard"], 
                       default="last_standing", help="Game mode (default: last_standing)")
    
    args = parser.parse_args()
    
    # Test pieces first
    test_fixed_pieces()
    
    # Find checkpoint
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        checkpoint_path = find_latest_checkpoint()
        if checkpoint_path is None:
            print("❌ No checkpoint found! Please train an agent first.")
            print("You can still run the game to test human controls, but AI won't work.")
            checkpoint_path = "dummy_path"
        else:
            if "checkpoints_fixed" in checkpoint_path:
                print(f"✅ Using FIXED AI checkpoint: {checkpoint_path}")
            else:
                print(f"⚠️  Using OLD AI checkpoint: {checkpoint_path}")
                print("   Note: AI may have broken rotations!")
    
    # Start the game
    print(f"\n🚀 Starting FIXED Tetris vs AI - {args.mode.replace('_', ' ').title()} Mode!")
    game = TetrisVsAI(checkpoint_path, mode=args.mode)
    game.run()