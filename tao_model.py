#!/usr/bin/env python3
"""
TAO: Transformer Against Opponent

A transformer-based opponent modeling system inspired by:
- "Towards Offline Opponent Modeling with In-Context Learning" (ICLR 2024)
- In-context learning for fast adaptation

Architecture:
- Input: Last N rounds of (my_move, opp_move) pairs
- Transformer encoder with attention
- Output: Predicted opponent move distribution

Designed to run on RTX 4090 (24GB VRAM).
"""

import os
import json
import math
import random
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Constants
MOVES = ['rock', 'paper', 'scissors']
MOVE_TO_IDX = {'rock': 0, 'paper': 1, 'scissors': 2}
IDX_TO_MOVE = {0: 'rock', 1: 'paper', 2: 'scissors'}
BEATS = {'rock': 'paper', 'paper': 'scissors', 'scissors': 'rock'}

# Model config
MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)


@dataclass
class TAOConfig:
    """Configuration for TAO model."""
    embed_dim: int = 64
    num_heads: int = 4
    num_layers: int = 3
    history_length: int = 16
    dropout: float = 0.1
    vocab_size: int = 6  # 3 moves × 2 players
    output_size: int = 3  # rock, paper, scissors


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""

    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TAOModel(nn.Module):
    """
    Transformer Against Opponent model.

    Takes history of (my_move, opp_move) pairs and predicts opponent's next move.
    """

    def __init__(self, config: TAOConfig = None):
        super().__init__()

        if config is None:
            config = TAOConfig()

        self.config = config

        # Embeddings for moves
        # We encode: my_rock=0, my_paper=1, my_scissors=2, opp_rock=3, opp_paper=4, opp_scissors=5
        self.move_embedding = nn.Embedding(config.vocab_size, config.embed_dim)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(
            config.embed_dim,
            max_len=config.history_length * 2,
            dropout=config.dropout
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.num_heads,
            dim_feedforward=config.embed_dim * 4,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)

        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.embed_dim, config.output_size)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def encode_history(self, my_moves: List[int], opp_moves: List[int]) -> torch.Tensor:
        """
        Encode move history into token sequence.

        Args:
            my_moves: List of my move indices (0, 1, 2)
            opp_moves: List of opponent move indices (0, 1, 2)

        Returns:
            Tensor of shape (seq_len,) with encoded tokens
        """
        tokens = []
        for my, opp in zip(my_moves, opp_moves):
            tokens.append(my)  # my move: 0, 1, 2
            tokens.append(opp + 3)  # opp move: 3, 4, 5

        return torch.tensor(tokens, dtype=torch.long)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Tensor of shape (batch, seq_len) with encoded move tokens

        Returns:
            Tensor of shape (batch, 3) with move probabilities
        """
        # Embed moves
        x = self.move_embedding(x)  # (batch, seq_len, embed_dim)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Transformer
        x = self.transformer(x)  # (batch, seq_len, embed_dim)

        # Use last token for prediction
        x = x[:, -1, :]  # (batch, embed_dim)

        # Output
        logits = self.output_head(x)  # (batch, 3)

        return logits

    def predict(self, my_moves: List[str], opp_moves: List[str]) -> torch.Tensor:
        """
        Predict opponent's next move.

        Args:
            my_moves: List of my moves as strings
            opp_moves: List of opponent moves as strings

        Returns:
            Tensor of shape (3,) with move probabilities
        """
        # Convert to indices
        my_idx = [MOVE_TO_IDX[m] for m in my_moves[-self.config.history_length:]]
        opp_idx = [MOVE_TO_IDX[m] for m in opp_moves[-self.config.history_length:]]

        # Pad if needed
        while len(my_idx) < self.config.history_length:
            my_idx.insert(0, 0)
            opp_idx.insert(0, 0)

        # Encode
        tokens = self.encode_history(my_idx, opp_idx).unsqueeze(0)

        # Move to device
        device = next(self.parameters()).device
        tokens = tokens.to(device)

        # Predict
        with torch.no_grad():
            logits = self(tokens)
            probs = F.softmax(logits, dim=-1)

        return probs.squeeze(0)

    def get_best_counter(self, my_moves: List[str], opp_moves: List[str]) -> Tuple[str, float]:
        """
        Get best counter move.

        Args:
            my_moves: List of my moves
            opp_moves: List of opponent moves

        Returns:
            Tuple of (best_move, confidence)
        """
        probs = self.predict(my_moves, opp_moves)
        pred_idx = probs.argmax().item()
        pred_move = IDX_TO_MOVE[pred_idx]
        confidence = probs[pred_idx].item()

        # Counter the predicted move
        counter = BEATS[pred_move]

        return counter, confidence

    def save(self, path: str = None):
        """Save model to disk."""
        if path is None:
            path = MODEL_DIR / "tao_model.pt"

        torch.save({
            'config': self.config,
            'state_dict': self.state_dict()
        }, path)
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str = None) -> 'TAOModel':
        """Load model from disk."""
        if path is None:
            path = MODEL_DIR / "tao_model.pt"

        checkpoint = torch.load(path, map_location='cpu')
        model = cls(checkpoint['config'])
        model.load_state_dict(checkpoint['state_dict'])

        return model


class RPSDataset(Dataset):
    """Dataset for training TAO model."""

    def __init__(self, matches: List[Dict], history_length: int = 16):
        """
        Args:
            matches: List of match dicts with 'history' key
            history_length: Number of rounds to use as context
        """
        self.samples = []
        self.history_length = history_length

        for match in matches:
            history = match.get('history', [])
            if len(history) < history_length + 1:
                continue

            # Create samples from sliding windows
            for i in range(history_length, len(history)):
                context = history[i-history_length:i]
                target = history[i]

                my_moves = [r.get('your_move', 'rock') for r in context]
                opp_moves = [r.get('opponent_move', 'rock') for r in context]
                target_move = target.get('opponent_move', 'rock')

                self.samples.append({
                    'my_moves': [MOVE_TO_IDX.get(m, 0) for m in my_moves],
                    'opp_moves': [MOVE_TO_IDX.get(m, 0) for m in opp_moves],
                    'target': MOVE_TO_IDX.get(target_move, 0)
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]

        # Encode history
        tokens = []
        for my, opp in zip(sample['my_moves'], sample['opp_moves']):
            tokens.append(my)
            tokens.append(opp + 3)

        x = torch.tensor(tokens, dtype=torch.long)
        y = torch.tensor(sample['target'], dtype=torch.long)

        return x, y


def generate_synthetic_data(n_matches: int = 100, rounds_per_match: int = 99) -> List[Dict]:
    """
    Generate synthetic training data with various opponent strategies.

    This is used when real match data is unavailable.
    """
    matches = []

    strategies = [
        ('random', lambda _: random.choice(MOVES)),
        ('rock_bias', lambda _: random.choices(MOVES, weights=[0.5, 0.25, 0.25])[0]),
        ('paper_bias', lambda _: random.choices(MOVES, weights=[0.25, 0.5, 0.25])[0]),
        ('scissors_bias', lambda _: random.choices(MOVES, weights=[0.25, 0.25, 0.5])[0]),
        ('wsls', None),  # Special handling
        ('cycle', None),  # Special handling
        ('counter', None),  # Special handling
    ]

    for i in range(n_matches):
        strat_name, strat_fn = random.choice(strategies)
        history = []

        opp_last = random.choice(MOVES)
        my_last = random.choice(MOVES)
        last_result = 'tie'

        for r in range(rounds_per_match):
            my_move = random.choice(MOVES)  # Simulated player

            # Determine opponent move based on strategy
            if strat_name == 'wsls':
                if last_result == 'opponent':  # Opp won
                    opp_move = opp_last  # Stay
                else:
                    opp_move = random.choice([m for m in MOVES if m != opp_last])
            elif strat_name == 'cycle':
                cycle = ['rock', 'paper', 'scissors']
                opp_move = cycle[(cycle.index(opp_last) + 1) % 3]
            elif strat_name == 'counter':
                # Counter our last move
                opp_move = BEATS.get(my_last, random.choice(MOVES))
            else:
                opp_move = strat_fn(r)

            # Determine result
            if BEATS[opp_move] == my_move:
                result = 'you'
            elif BEATS[my_move] == opp_move:
                result = 'opponent'
            else:
                result = 'tie'

            history.append({
                'round': r + 1,
                'your_move': my_move,
                'opponent_move': opp_move,
                'winner': result
            })

            opp_last = opp_move
            my_last = my_move
            last_result = result

        matches.append({
            'id': f'synthetic_{i}',
            'strategy': strat_name,
            'history': history
        })

    return matches


def train_tao(
    model: TAOModel,
    train_data: List[Dict],
    val_data: List[Dict] = None,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-4,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> TAOModel:
    """
    Train TAO model.

    Args:
        model: TAO model to train
        train_data: List of match dicts for training
        val_data: List of match dicts for validation
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        device: Device to train on

    Returns:
        Trained model
    """
    model = model.to(device)

    # Create datasets
    train_dataset = RPSDataset(train_data, model.config.history_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    if val_data:
        val_dataset = RPSDataset(val_data, model.config.history_length)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Loss
    criterion = nn.CrossEntropyLoss()

    print(f"Training on {device}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print("-" * 50)

    best_val_acc = 0
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_correct += (logits.argmax(dim=-1) == y).sum().item()
            train_total += y.size(0)

        train_acc = train_correct / train_total

        # Validation
        if val_data:
            model.eval()
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    logits = model(x)
                    val_correct += (logits.argmax(dim=-1) == y).sum().item()
                    val_total += y.size(0)

            val_acc = val_correct / val_total

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                model.save(MODEL_DIR / "tao_best.pt")

            print(f"Epoch {epoch+1}/{epochs}: Loss={train_loss/len(train_loader):.4f} "
                  f"Train Acc={train_acc:.1%} Val Acc={val_acc:.1%}")
        else:
            print(f"Epoch {epoch+1}/{epochs}: Loss={train_loss/len(train_loader):.4f} "
                  f"Train Acc={train_acc:.1%}")

        scheduler.step()

    print("-" * 50)
    print(f"Training complete! Best val acc: {best_val_acc:.1%}")

    return model


def evaluate_tao(model: TAOModel, test_data: List[Dict], device: str = 'cpu') -> Dict:
    """Evaluate TAO model on test data."""
    model = model.to(device)
    model.eval()

    dataset = RPSDataset(test_data, model.config.history_length)
    loader = DataLoader(dataset, batch_size=64)

    correct = 0
    total = 0
    predictions = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=-1)

            correct += (preds == y).sum().item()
            total += y.size(0)
            predictions.extend(preds.cpu().tolist())

    accuracy = correct / total

    return {
        'accuracy': accuracy,
        'total_samples': total,
        'predictions': predictions
    }


# ============================================================================
# CLI Interface
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description='TAO Model Training')
    parser.add_argument('--train', action='store_true', help='Train model')
    parser.add_argument('--synthetic', action='store_true', help='Use synthetic data')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--test', action='store_true', help='Test model')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()

    if args.train:
        print("TAO Model Training")
        print("=" * 50)

        # Generate synthetic data if no real data
        print("Generating synthetic training data...")
        train_data = generate_synthetic_data(n_matches=200)
        val_data = generate_synthetic_data(n_matches=50)

        print(f"Generated {len(train_data)} training matches")
        print(f"Generated {len(val_data)} validation matches")

        # Create model
        config = TAOConfig(
            embed_dim=64,
            num_heads=4,
            num_layers=3,
            history_length=16
        )
        model = TAOModel(config)

        param_count = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {param_count:,}")

        # Train
        model = train_tao(
            model,
            train_data,
            val_data,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=args.device
        )

        # Save
        model.save()

    elif args.test:
        print("TAO Model Testing")
        print("=" * 50)

        # Load model
        try:
            model = TAOModel.load()
            model = model.to(args.device)
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Train a model first with --train")
            return

        # Generate test data
        test_data = generate_synthetic_data(n_matches=50)

        # Evaluate
        results = evaluate_tao(model, test_data, args.device)
        print(f"Test Accuracy: {results['accuracy']:.1%}")
        print(f"Total Samples: {results['total_samples']}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
