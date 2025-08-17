# tetris_dqn_placement.py
# Tetris DQN with placement-based actions (no rendering during training)

import os
import numpy as np
import random
import time
from collections import deque, namedtuple
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim

# ============================== TETRIS CORE =====================================

BOARD_W, BOARD_H = 10, 20

BASE_PIECES = {
    "I": np.array([[1, 1, 1, 1]], dtype=np.uint8),
    "O": np.array([[1, 1], [1, 1]], dtype=np.uint8),
    "T": np.array([[0,1,0], [1,1,1]], dtype=np.uint8),
    "S": np.array([[0,1,1], [1,1,0]], dtype=np.uint8),
    "Z": np.array([[1,1,0], [0,1,1]], dtype=np.uint8),
    "J": np.array([[1,0,0], [1,1,1]], dtype=np.uint8),
    "L": np.array([[0,0,1], [1,1,1]], dtype=np.uint8),
}

def unique_rotations(shape: np.ndarray) -> List[np.ndarray]:
    rots, seen = [], set()
    cur = shape
    for _ in range(4):
        key = cur.tobytes()
        if key not in seen:
            rots.append(cur.copy())
            seen.add(key)
        cur = np.rot90(cur)
    return rots

PIECES = {name: unique_rotations(shape) for name, shape in BASE_PIECES.items()}
PIECE_NAMES = ["I","O","T","S","Z","J","L"]
PIECE_INDEX = {n:i for i,n in enumerate(PIECE_NAMES)}

def fits(board: np.ndarray, shape: np.ndarray, x: int, y: int) -> bool:
    h, w = shape.shape
    if x < 0 or x + w > BOARD_W or y < 0 or y + h > BOARD_H:
        return False
    region = board[y:y+h, x:x+w]
    return np.all(region + shape <= 1)

def hard_drop(board: np.ndarray, shape: np.ndarray, x: int) -> Optional[int]:
    """Find the lowest valid y position for the piece at column x"""
    for y in range(BOARD_H):
        if not fits(board, shape, x, y):
            return y - 1 if y > 0 else None
    return BOARD_H - shape.shape[0]

def place_and_clear(board: np.ndarray, shape: np.ndarray, x: int, y: int) -> Tuple[np.ndarray, int]:
    new_board = board.copy()
    h, w = shape.shape
    new_board[y:y+h, x:x+w] |= shape
    full_rows = np.where(np.all(new_board == 1, axis=1))[0]
    lines = len(full_rows)
    if lines > 0:
        new_board = np.delete(new_board, full_rows, axis=0)
        new_board = np.vstack([np.zeros((lines, BOARD_W), dtype=np.uint8), new_board])
    return new_board, lines

# ============================== ACTION SPACE ===================================

class PlacementAction:
    def __init__(self, piece_name: str, rotation: int, x: int):
        self.piece_name = piece_name
        self.rotation = rotation
        self.x = x
    
    def __repr__(self):
        return f"Place({self.piece_name}, rot={self.rotation}, x={self.x})"

def generate_all_placements():
    """Generate all possible placement actions for all pieces"""
    all_actions = []
    for piece_name, rotations in PIECES.items():
        for rot_idx, shape in enumerate(rotations):
            shape_width = shape.shape[1]
            for x in range(BOARD_W - shape_width + 1):
                all_actions.append(PlacementAction(piece_name, rot_idx, x))
    return all_actions

ALL_PLACEMENTS = generate_all_placements()
print(f"Total placement actions: {len(ALL_PLACEMENTS)}")

def get_valid_placements(board: np.ndarray, piece_name: str) -> List[int]:
    """Get list of valid placement action indices for the current piece"""
    valid_indices = []
    for i, action in enumerate(ALL_PLACEMENTS):
        if action.piece_name == piece_name:
            shape = PIECES[piece_name][action.rotation]
            y = hard_drop(board, shape, action.x)
            if y is not None and y >= 0:
                valid_indices.append(i)
    return valid_indices

def create_action_mask(board: np.ndarray, piece_name: str) -> np.ndarray:
    """Create binary mask for valid actions"""
    mask = np.zeros(len(ALL_PLACEMENTS), dtype=np.float32)
    valid_indices = get_valid_placements(board, piece_name)
    mask[valid_indices] = 1.0
    return mask

# ============================== TETRIS ENVIRONMENT ===============================

class TetrisEnv:
    """Tetris environment with placement-based actions"""
    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        self.board = np.zeros((BOARD_H, BOARD_W), dtype=np.uint8)
        self.cur_piece = None
        self.next_piece = None
        self.done = False
        self.bag = []
        self.total_lines = 0
        self.total_pieces = 0

    def _refill_bag(self):
        self.bag = PIECE_NAMES[:]
        self.rng.shuffle(self.bag)

    def _draw_piece(self):
        if not self.bag:
            self._refill_bag()
        return self.bag.pop()

    def _spawn_piece(self):
        """Spawn new piece"""
        self.cur_piece = self.next_piece if self.next_piece else self._draw_piece()
        self.next_piece = self._draw_piece()
        self.total_pieces += 1
        
        # Check if any valid placements exist (game over check)
        valid_placements = get_valid_placements(self.board, self.cur_piece)
        if not valid_placements:
            self.done = True

    def reset(self):
        self.board[:] = 0
        self.done = False
        self.bag = []
        self.next_piece = None
        self.total_lines = 0
        self.total_pieces = 0
        self._spawn_piece()
        return self.get_state()

    def step(self, action_idx: int) -> Tuple:
        """Take a placement action"""
        if self.done:
            return self.get_state(), 0.0, True, {}
        
        # Validate action
        if action_idx >= len(ALL_PLACEMENTS):
            return self.get_state(), -10.0, True, {}
        
        action = ALL_PLACEMENTS[action_idx]
        
        # Check if action is for current piece
        if action.piece_name != self.cur_piece:
            return self.get_state(), -10.0, True, {}
        
        # Execute placement
        shape = PIECES[self.cur_piece][action.rotation]
        y = hard_drop(self.board, shape, action.x)
        
        if y is None or y < 0:
            # Invalid placement
            return self.get_state(), -10.0, True, {}
        
        # Place piece and clear lines
        self.board, lines_cleared = place_and_clear(self.board, shape, action.x, y)
        self.total_lines += lines_cleared
        
        # Calculate reward
        reward = self._calculate_reward(lines_cleared, y)
        
        # Spawn next piece
        self._spawn_piece()
        
        info = {
            "lines": lines_cleared,
            "total_lines": self.total_lines,
            "total_pieces": self.total_pieces,
            "placement_height": y
        }
        
        return self.get_state(), reward, self.done, info

    def _calculate_reward(self, lines_cleared: int, placement_y: int) -> float:
        """Calculate reward for the placement"""
        if self.done:
            return -50.0  # Heavy game over penalty
        
        # Line clearing rewards (exponential)
        line_rewards = {0: 0.0, 1: 10.0, 2: 25.0, 3: 50.0, 4: 100.0}
        reward = line_rewards.get(lines_cleared, 0.0)
        
        # Height bonus - reward placing pieces lower
        height_bonus = (BOARD_H - placement_y) * 0.1
        
        # Small survival bonus
        reward += 1.0
        
        return reward + height_bonus

    def get_state(self):
        """Get current state representation"""
        # One-hot encode pieces
        cur_piece_vec = np.zeros(7, dtype=np.float32)
        next_piece_vec = np.zeros(7, dtype=np.float32)
        
        if self.cur_piece:
            cur_piece_vec[PIECE_INDEX[self.cur_piece]] = 1.0
        if self.next_piece:
            next_piece_vec[PIECE_INDEX[self.next_piece]] = 1.0
            
        return {
            "board": self.board.astype(np.float32),
            "cur_piece": cur_piece_vec,
            "next_piece": next_piece_vec,
        }

    def get_valid_actions(self):
        """Get mask of valid placement actions"""
        if self.done or self.cur_piece is None:
            return np.zeros(len(ALL_PLACEMENTS), dtype=np.float32)
        return create_action_mask(self.board, self.cur_piece)

# ============================== BOARD ANALYSIS =================================

def analyze_board(board: np.ndarray) -> np.ndarray:
    """Extract board features for the neural network"""
    features = []
    
    # Column heights
    heights = np.zeros(BOARD_W)
    for c in range(BOARD_W):
        col = board[:, c]
        filled_rows = np.where(col != 0)[0]
        if len(filled_rows) > 0:
            heights[c] = BOARD_H - filled_rows[0]
    
    features.extend(heights)  # 10 features
    
    # Aggregate height features
    features.append(heights.max())  # Max height
    features.append(heights.mean())  # Average height
    features.append(heights.std())   # Height variance
    
    # Holes (empty cells with filled cells above)
    holes = 0
    for c in range(BOARD_W):
        col = board[:, c]
        filled_rows = np.where(col != 0)[0]
        if len(filled_rows) > 0:
            top_filled = filled_rows[0]
            holes += (col[top_filled:] == 0).sum()
    features.append(holes)
    
    # Bumpiness (height differences between adjacent columns)
    bumpiness = np.abs(np.diff(heights)).sum()
    features.append(bumpiness)
    
    # Complete lines
    complete_lines = np.sum(np.all(board == 1, axis=1))
    features.append(complete_lines)
    
    # Wells (deep single-width gaps)
    wells = 0
    for c in range(BOARD_W):
        if c == 0:  # Left edge
            if heights[c] < heights[c+1]:
                wells += heights[c+1] - heights[c]
        elif c == BOARD_W-1:  # Right edge
            if heights[c] < heights[c-1]:
                wells += heights[c-1] - heights[c]
        else:  # Middle columns
            left_higher = heights[c-1] > heights[c]
            right_higher = heights[c+1] > heights[c]
            if left_higher and right_higher:
                wells += min(heights[c-1], heights[c+1]) - heights[c]
    features.append(wells)
    
    return np.array(features, dtype=np.float32)

# ============================== NEURAL NETWORK ====================================

class TetrisPlacementNet(nn.Module):
    """Neural network for placement-based Tetris"""
    def __init__(self, n_actions: int):
        super().__init__()
        
        # CNN for board representation
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((5, 5))
        )
        
        # Process board features
        self.board_features_fc = nn.Sequential(
            nn.Linear(17, 32),  # Board analysis features
            nn.ReLU()
        )
        
        # Process piece information
        self.piece_fc = nn.Sequential(
            nn.Linear(14, 32),  # 7 + 7 for cur + next piece
            nn.ReLU()
        )
        
        # Combine all features
        conv_features = 64 * 5 * 5  # From conv layers
        total_features = conv_features + 32 + 32  # conv + board_features + pieces
        
        self.fc = nn.Sequential(
            nn.Linear(total_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, n_actions)
        )

    def forward(self, board, cur_piece, next_piece, board_features, mask=None):
        batch_size = board.size(0)
        
        # Process board through CNN
        conv_out = self.conv(board.unsqueeze(1))
        conv_features = conv_out.view(batch_size, -1)
        
        # Process board features
        board_feat = self.board_features_fc(board_features)
        
        # Process pieces
        piece_combined = torch.cat([cur_piece, next_piece], dim=1)
        piece_features = self.piece_fc(piece_combined)
        
        # Combine all features
        combined = torch.cat([conv_features, board_feat, piece_features], dim=1)
        q_values = self.fc(combined)
        
        # Apply action mask
        if mask is not None:
            q_values = q_values * mask + (-1e9) * (1 - mask)
            
        return q_values

# ============================== REPLAY BUFFER ===================================

Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done", "mask", "next_mask"])

class ReplayBuffer:
    def __init__(self, capacity=200000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, *args):
        self.buffer.append(Transition(*args))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))
    
    def __len__(self):
        return len(self.buffer)

# ============================== AGENT ==========================================

class PlacementDQNAgent:
    """DQN Agent for placement-based Tetris"""
    def __init__(self, n_actions, device="cpu", lr=1e-4, gamma=0.99, 
                 eps_start=1.0, eps_end=0.01, eps_decay=200000):
        self.device = device
        self.n_actions = n_actions
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.steps = 0
        
        # Networks
        self.q_net = TetrisPlacementNet(n_actions).to(device)
        self.target_net = TetrisPlacementNet(n_actions).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        
        # Buffer
        self.buffer = ReplayBuffer()

    def get_epsilon(self):
        if self.steps >= self.eps_decay:
            return self.eps_end
        return self.eps_start - (self.eps_start - self.eps_end) * (self.steps / self.eps_decay)

    def select_action(self, env):
        state = env.get_state()
        mask = env.get_valid_actions()
        
        # Random valid action with probability epsilon
        if random.random() < self.get_epsilon():
            valid_actions = np.where(mask == 1)[0]
            if len(valid_actions) > 0:
                return random.choice(valid_actions)
            return 0
        
        # Greedy action
        return self._get_best_action(state, mask)

    def _get_best_action(self, state, mask):
        with torch.no_grad():
            board = torch.FloatTensor(state["board"]).unsqueeze(0).to(self.device)
            cur_piece = torch.FloatTensor(state["cur_piece"]).unsqueeze(0).to(self.device)
            next_piece = torch.FloatTensor(state["next_piece"]).unsqueeze(0).to(self.device)
            
            # Analyze board
            board_features = analyze_board(state["board"])
            board_feat_tensor = torch.FloatTensor(board_features).unsqueeze(0).to(self.device)
            
            mask_tensor = torch.FloatTensor(mask).unsqueeze(0).to(self.device)
            
            q_values = self.q_net(board, cur_piece, next_piece, board_feat_tensor, mask_tensor)
            return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done, mask, next_mask):
        self.buffer.push(state, action, reward, next_state, done, mask, next_mask)

    def train(self, batch_size=64):
        if len(self.buffer) < batch_size:
            return None
            
        self.steps += 1
        batch = self.buffer.sample(batch_size)
        
        # Convert batch to tensors (optimized)
        boards = torch.tensor(np.stack([s["board"] for s in batch.state]), dtype=torch.float32, device=self.device)
        cur_pieces = torch.tensor(np.stack([s["cur_piece"] for s in batch.state]), dtype=torch.float32, device=self.device)
        next_pieces = torch.tensor(np.stack([s["next_piece"] for s in batch.state]), dtype=torch.float32, device=self.device)
        
        # Analyze boards (optimized)
        board_features = torch.tensor(np.stack([analyze_board(s["board"]) for s in batch.state]), dtype=torch.float32, device=self.device)
        next_board_features = torch.tensor(np.stack([analyze_board(s["board"]) for s in batch.next_state]), dtype=torch.float32, device=self.device)
        
        actions = torch.tensor(batch.action, dtype=torch.long, device=self.device)
        rewards = torch.tensor(batch.reward, dtype=torch.float32, device=self.device)
        dones = torch.tensor(batch.done, dtype=torch.bool, device=self.device)
        masks = torch.tensor(np.stack(batch.mask), dtype=torch.float32, device=self.device)
        
        next_boards = torch.tensor(np.stack([s["board"] for s in batch.next_state]), dtype=torch.float32, device=self.device)
        next_cur_pieces = torch.tensor(np.stack([s["cur_piece"] for s in batch.next_state]), dtype=torch.float32, device=self.device)
        next_next_pieces = torch.tensor(np.stack([s["next_piece"] for s in batch.next_state]), dtype=torch.float32, device=self.device)
        next_masks = torch.tensor(np.stack(batch.next_mask), dtype=torch.float32, device=self.device)
        
        # Current Q values
        current_q = self.q_net(boards, cur_pieces, next_pieces, board_features, masks)
        current_q = current_q.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Next Q values (Double DQN)
        with torch.no_grad():
            # Use online network to select actions
            next_q_online = self.q_net(next_boards, next_cur_pieces, next_next_pieces, next_board_features, next_masks)
            next_actions = next_q_online.argmax(1)
            
            # Use target network to evaluate actions
            next_q_target = self.target_net(next_boards, next_cur_pieces, next_next_pieces, next_board_features, next_masks)
            next_q_values = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            
            target_q = rewards + (self.gamma * next_q_values * ~dones)
        
        # Loss
        loss = nn.functional.mse_loss(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def save(self, path, episode=None, metrics=None):
        """Save agent state with optional training metadata"""
        save_dict = {
            'q_net': self.q_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps': self.steps,
            'hyperparams': {
                'gamma': self.gamma,
                'eps_start': self.eps_start,
                'eps_end': self.eps_end,
                'eps_decay': self.eps_decay,
                'learning_rate': self.optimizer.param_groups[0]['lr']
            }
        }
        
        # Add training metadata if provided
        if episode is not None:
            save_dict['episode'] = episode
        if metrics is not None:
            save_dict['metrics'] = metrics
            
        # Add RNG states for reproducibility
        save_dict['random_states'] = {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state()
        }
        
        torch.save(save_dict, path)

    def load(self, path, reset_epsilon=False):
        """Load agent state and return training metadata"""
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load network states
        self.q_net.load_state_dict(checkpoint['q_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        
        # Handle epsilon schedule reset
        if reset_epsilon:
            print("Resetting epsilon schedule for continued exploration")
            self.steps = 0  # Reset to start of epsilon decay
        else:
            self.steps = checkpoint.get('steps', 0)
        
        # Restore hyperparameters if they were saved
        if 'hyperparams' in checkpoint:
            hp = checkpoint['hyperparams']
            if not reset_epsilon:  # Only restore old hyperparams if not resetting
                self.gamma = hp.get('gamma', self.gamma)
                self.eps_start = hp.get('eps_start', self.eps_start)
                self.eps_end = hp.get('eps_end', self.eps_end)
                self.eps_decay = hp.get('eps_decay', self.eps_decay)
        
        # Restore RNG states for reproducibility
        if 'random_states' in checkpoint and not reset_epsilon:
            try:
                random.setstate(checkpoint['random_states']['python'])
                np.random.set_state(checkpoint['random_states']['numpy'])
                torch.set_rng_state(checkpoint['random_states']['torch'])
            except Exception as e:
                print(f"Warning: Could not restore RNG states: {e}")
        
        # Return training metadata for resumption
        return {
            'episode': checkpoint.get('episode', 0),
            'metrics': checkpoint.get('metrics', {}),
            'steps': self.steps,
            'epsilon_reset': reset_epsilon
        }

def find_latest_checkpoint(checkpoint_dir="checkpoints", prefix="placement_tetris_ep"):
    """Find the most recent checkpoint file"""
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith(prefix) and f.endswith('.pth')]
    if not checkpoint_files:
        return None
    
    # Extract episode numbers and find the highest
    episodes = []
    for filename in checkpoint_files:
        try:
            # Extract episode number from filename like "placement_tetris_ep15000.pth"
            episode_str = filename.replace(prefix, "").replace(".pth", "")
            episodes.append((int(episode_str), filename))
        except ValueError:
            continue
    
    if episodes:
        latest_episode, latest_file = max(episodes)
        return os.path.join(checkpoint_dir, latest_file), latest_episode
    
    return None

def save_training_state(agent, episode, episode_rewards, episode_lines, episode_pieces, 
                       checkpoint_dir="checkpoints", prefix="placement_tetris"):
    """Save complete training state including metrics"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save the main checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f"{prefix}_ep{episode}.pth")
    metrics = {
        'episode_rewards': episode_rewards,
        'episode_lines': episode_lines, 
        'episode_pieces': episode_pieces,
        'training_time': time.time()
    }
    
    agent.save(checkpoint_path, episode=episode, metrics=metrics)
    
    # Also save a "latest" checkpoint for easy resuming
    latest_path = os.path.join(checkpoint_dir, f"{prefix}_latest.pth")
    agent.save(latest_path, episode=episode, metrics=metrics)
    
    print(f"Saved checkpoint: {checkpoint_path}")
    return checkpoint_path

def load_training_state(agent, checkpoint_path=None, checkpoint_dir="checkpoints", reset_epsilon=False):
    """Load training state and return where to resume from"""
    if checkpoint_path is None:
        # Try to find latest checkpoint
        result = find_latest_checkpoint(checkpoint_dir)
        if result is None:
            print("No checkpoint found, starting fresh training")
            return None
        checkpoint_path, _ = result
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return None
    
    print(f"Loading checkpoint: {checkpoint_path}")
    if reset_epsilon:
        print("RESETTING EPSILON SCHEDULE - Will restart exploration")
    
    resume_data = agent.load(checkpoint_path, reset_epsilon=reset_epsilon)
    
    if resume_data and 'metrics' in resume_data:
        metrics = resume_data['metrics']
        print(f"Resumed from episode {resume_data['episode']}")
        print(f"Training steps: {resume_data['steps']}")
        print(f"Current epsilon: {agent.get_epsilon():.3f}")
        if 'episode_rewards' in metrics:
            recent_rewards = metrics['episode_rewards'][-100:] if len(metrics['episode_rewards']) > 100 else metrics['episode_rewards']
            print(f"Recent average reward: {np.mean(recent_rewards):.2f}")
        
        return {
            'start_episode': resume_data['episode'] + 1,
            'episode_rewards': metrics.get('episode_rewards', []),
            'episode_lines': metrics.get('episode_lines', []),
            'episode_pieces': metrics.get('episode_pieces', [])
        }
    
    return None

def train_placement_tetris(episodes=50000, save_every=5000, eval_every=1000, resume=True, reset_epsilon=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Total placement actions: {len(ALL_PLACEMENTS)}")
    
    env = TetrisEnv(seed=42)
    agent = PlacementDQNAgent(
        n_actions=len(ALL_PLACEMENTS), 
        device=device,
        lr=1e-4,
        eps_start=0.95,
        eps_end=0.05,
        eps_decay=100000
    )
    
    # Initialize with empty lists
    episode_rewards = []
    episode_lines = []
    episode_pieces = []
    start_episode = 1
    
    # FIXED: Proper checkpoint loading
    if resume and os.path.exists("checkpoints"):
        checkpoint_files = [f for f in os.listdir("checkpoints") 
                          if f.startswith("placement_tetris_ep") and f.endswith(".pth")]
        
        if checkpoint_files:
            # Find latest checkpoint
            episodes_found = []
            for f in checkpoint_files:
                try:
                    ep_str = f.replace("placement_tetris_ep", "").replace(".pth", "")
                    episodes_found.append((int(ep_str), f))
                except:
                    continue
            
            if episodes_found:
                latest_ep, latest_file = max(episodes_found)
                checkpoint_path = os.path.join("checkpoints", latest_file)
                print(f"Loading checkpoint: {checkpoint_path}")
                
                try:
                    checkpoint = torch.load(checkpoint_path, map_location=device)
                    
                    # Load agent state
                    agent.q_net.load_state_dict(checkpoint['q_net'])
                    agent.target_net.load_state_dict(checkpoint['target_net'])
                    agent.optimizer.load_state_dict(checkpoint['optimizer'])
                    
                    # Handle epsilon
                    if reset_epsilon:
                        agent.steps = 0
                        print("🔄 EPSILON RESET - Exploration re-enabled!")
                    else:
                        agent.steps = checkpoint.get('steps', 0)
                    
                    # CRITICAL: Restore episode number AND metrics
                    if 'episode' in checkpoint:
                        start_episode = checkpoint.get('episode', 0) + 1
                        print(f"✅ Resumed from episode {start_episode}")
                    else:
                        print("⚠️ No episode info in checkpoint, starting from episode 1")

                    if 'metrics' in checkpoint:
                        metrics = checkpoint['metrics']
                        episode_rewards = metrics.get('episode_rewards', [])
                        episode_lines = metrics.get('episode_lines', [])
                        episode_pieces = metrics.get('episode_pieces', [])
                        
                        print(f"✅ Loaded {len(episode_rewards)} episodes of history")
                        
                        # Show recent performance
                        if len(episode_rewards) >= 100:
                            recent_reward = np.mean(episode_rewards[-100:])
                            recent_lines = np.mean(episode_lines[-100:])
                            print(f"✅ Recent avg reward: {recent_reward:.2f}")
                            print(f"✅ Recent avg lines: {recent_lines:.1f}")
                    else:
                        print("⚠️ No metrics in checkpoint - will build new history")

                    print(f"✅ Current epsilon: {agent.get_epsilon():.3f}")
                    
                except Exception as e:
                    print(f"❌ Failed to load checkpoint: {e}")
                    print("Starting fresh...")
                    start_episode = 1
    
    print("Training in progress...")
    start_time = time.time()
    
    
    for episode in range(start_episode, episodes + 1):
        state = env.reset()
        total_reward = 0
        step = 0
        loss = 0
        
        while not env.done and step < 10000:  # Max steps per episode
            # Get current action mask
            mask = env.get_valid_actions()
            
            # Select action
            action = agent.select_action(env)
            
            # Take step
            next_state, reward, done, info = env.step(action)
            next_mask = env.get_valid_actions()
            
            # Store transition
            agent.store_transition(state, action, reward, next_state, done, mask, next_mask)
            
            # Train (reduced frequency for speed)
            if len(agent.buffer) > 10000 and step % 3 == 0:  # Train every 3 steps instead of every step
                train_loss = agent.train(batch_size=32)  # Smaller batch size for speed
                if train_loss is not None:
                    loss = train_loss
            
            state = next_state
            total_reward += reward
            step += 1
        
        episode_rewards.append(total_reward)
        episode_lines.append(info.get("total_lines", 0))
        episode_pieces.append(info.get("total_pieces", 0))
        
        # Update target network
        if episode % 1000 == 0:
            agent.update_target()
        
        # Print progress
        if episode % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_lines = np.mean(episode_lines[-100:])
            avg_pieces = np.mean(episode_pieces[-100:])
            elapsed = time.time() - start_time
            
            print(f"Episode {episode:5d} | "
                  f"Avg Reward: {avg_reward:8.2f} | "
                  f"Avg Lines: {avg_lines:6.1f} | "
                  f"Avg Pieces: {avg_pieces:6.1f} | "
                  f"Epsilon: {agent.get_epsilon():.3f} | "
                  f"Buffer: {len(agent.buffer):6d} | "
                  f"Time: {elapsed/60:.1f}m")
        
        # Evaluation
        if episode % eval_every == 0:
            eval_reward, eval_lines, eval_pieces = evaluate_agent(agent, episodes=10)
            print(f"EVAL - Reward: {eval_reward:.2f}, Lines: {eval_lines:.1f}, Pieces: {eval_pieces:.1f}")
        
        # Save model
        if episode % save_every == 0:
            os.makedirs("checkpoints", exist_ok=True)
            agent.save(f"checkpoints/placement_tetris_ep{episode}.pth")
            print(f"Model saved at episode {episode}")
    
    return agent, episode_rewards, episode_lines

def evaluate_agent(agent, episodes=10):
    """Evaluate agent performance"""
    total_rewards = []
    total_lines = []
    total_pieces = []
    
    original_epsilon = agent.get_epsilon
    agent.get_epsilon = lambda: 0.0  # No exploration during evaluation
    
    for _ in range(episodes):
        env = TetrisEnv(seed=random.randint(0, 9999))
        state = env.reset()
        episode_reward = 0
        step = 0
        
        while not env.done and step < 10000:
            mask = env.get_valid_actions()
            action = agent._get_best_action(state, mask)
            state, reward, done, info = env.step(action)
            episode_reward += reward
            step += 1
        
        total_rewards.append(episode_reward)
        total_lines.append(info.get("total_lines", 0))
        total_pieces.append(info.get("total_pieces", 0))
    
    agent.get_epsilon = original_epsilon  # Restore original epsilon function
    
    return np.mean(total_rewards), np.mean(total_lines), np.mean(total_pieces)

if __name__ == "__main__":
    # Train the agent with automatic resume capability
    print("Training Placement-Based Tetris DQN")
    print("=" * 50)
    
    # OPTION 1: Continue with better epsilon schedule (RECOMMENDED)
    agent, rewards, lines = train_placement_tetris(
        episodes=50000,
        save_every=5000,
        eval_every=1000,
        resume=True,           # Load your existing progress
        reset_epsilon=True     # BUT reset exploration schedule
    )
    
    # OPTION 2: If you want normal resume (keep old epsilon)
    # agent, rewards, lines = train_placement_tetris(
    #     episodes=50000,
    #     save_every=5000,
    #     eval_every=1000,
    #     resume=True,
    #     reset_epsilon=False
    # )
    
    # OPTION 3: Start completely fresh
    # agent, rewards, lines = train_placement_tetris(
    #     episodes=50000,
    #     save_every=5000,
    #     eval_every=1000,
    #     resume=False
    # )
    
    print("\nTraining completed!")
    print(f"Final average reward: {np.mean(rewards[-1000:]):.2f}")
    print(f"Final average lines: {np.mean(lines[-1000:]):.1f}")
    
    # Final evaluation
    final_reward, final_lines, final_pieces = evaluate_agent(agent, episodes=100)
    print(f"Final evaluation over 100 games:")
    print(f"  Average reward: {final_reward:.2f}")
    print(f"  Average lines cleared: {final_lines:.1f}")
    print(f"  Average pieces placed: {final_pieces:.1f}")
    
    # Instructions for manual loading
    print("\n" + "=" * 50)
    print("CHECKPOINT USAGE:")
    print("To resume with reset exploration: set reset_epsilon=True")
    print("To resume normally: set reset_epsilon=False") 
    print("To start fresh: set resume=False")
    print("Current settings use OPTION 1 (recommended for your situation)")
    print("=" * 50)