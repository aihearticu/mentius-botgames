#!/usr/bin/env python3
"""
Mentius RPS Brain v9.0 - Back to Basics

Key insight from research:
- IO2_fightinguuu (78.74% WR) and v5 worked because they're SIMPLE
- v8 overthought late-game and became predictable
- Best strategy: multiple predictors + simple selection + fixed noise

Changes from v8:
1. REMOVE all late-game special handling (was making us predictable)
2. Fixed 10% noise (not dynamic)
3. Simpler predictor selection (no phase weighting)
4. Keep the good: spam detection, meta-predictors, pattern matching

Based on:
- Iocaine Powder meta-strategy (P.0, P.1, P.2 levels)
- Raymond Hettinger's multi-arm bandit approach
- Our v5 simplicity that worked
"""

import random
import math
from collections import Counter
from typing import Optional, Tuple, List, Dict

VERSION = "9.0"

# Constants
MOVES = ['rock', 'paper', 'scissors']
BEATS = {'rock': 'paper', 'paper': 'scissors', 'scissors': 'rock'}
LOSES_TO = {'rock': 'scissors', 'paper': 'rock', 'scissors': 'paper'}
MOVE_TO_IDX = {'rock': 0, 'paper': 1, 'scissors': 2}
IDX_TO_MOVE = {0: 'rock', 1: 'paper', 2: 'scissors'}


class Predictor:
    """Simple predictor with rolling 6-round score window."""
    
    def __init__(self, name: str):
        self.name = name
        self.results: List[int] = []  # 1=correct, 0=wrong
        self.last_prediction: Optional[str] = None
    
    def predict(self, brain: 'RPSBrainV9') -> Optional[str]:
        raise NotImplementedError
    
    def update(self, actual_opp: str):
        """Update score based on last prediction."""
        if self.last_prediction is not None:
            correct = 1 if self.last_prediction == actual_opp else 0
            self.results.append(correct)
            # Keep only last 6 results
            if len(self.results) > 6:
                self.results.pop(0)
    
    def get_score(self) -> float:
        """Recent accuracy with recency weighting."""
        if len(self.results) < 2:
            return 0.0
        score = 0.0
        for i, r in enumerate(self.results):
            weight = 1.0 + i * 0.3  # More recent = higher weight
            score += weight if r else -weight * 0.3
        return score


class FrequencyPredictor(Predictor):
    """Predict opponent's most common move."""
    
    def __init__(self, decay: float = 0.9):
        super().__init__('freq')
        self.counts = [0.1, 0.1, 0.1]
        self.decay = decay
    
    def predict(self, brain: 'RPSBrainV9') -> Optional[str]:
        total = sum(self.counts)
        if total < 1:
            return None
        idx = self.counts.index(max(self.counts))
        # Only predict if one move is dominant
        if self.counts[idx] / total < 0.35:
            return None
        self.last_prediction = IDX_TO_MOVE[idx]
        return self.last_prediction
    
    def observe(self, opp_move: str):
        for i in range(3):
            self.counts[i] *= self.decay
        self.counts[MOVE_TO_IDX[opp_move]] += 1


class TransitionPredictor(Predictor):
    """Predict based on opponent's transitions."""
    
    def __init__(self, decay: float = 0.85):
        super().__init__('trans')
        self.trans = [[0.1, 0.1, 0.1] for _ in range(3)]
        self.decay = decay
        self.last_opp: Optional[str] = None
    
    def predict(self, brain: 'RPSBrainV9') -> Optional[str]:
        if self.last_opp is None:
            return None
        counts = self.trans[MOVE_TO_IDX[self.last_opp]]
        total = sum(counts)
        if total < 1:
            return None
        idx = counts.index(max(counts))
        if counts[idx] / total < 0.38:
            return None
        self.last_prediction = IDX_TO_MOVE[idx]
        return self.last_prediction
    
    def observe(self, opp_move: str):
        if self.last_opp is not None:
            last_idx = MOVE_TO_IDX[self.last_opp]
            for i in range(3):
                self.trans[last_idx][i] *= self.decay
            self.trans[last_idx][MOVE_TO_IDX[opp_move]] += 1
        self.last_opp = opp_move


class ResponsePredictor(Predictor):
    """Predict based on how opponent responds to our moves."""
    
    def __init__(self, decay: float = 0.85):
        super().__init__('resp')
        self.resp = [[0.1, 0.1, 0.1] for _ in range(3)]
        self.decay = decay
        self.last_my: Optional[str] = None
    
    def predict(self, brain: 'RPSBrainV9') -> Optional[str]:
        if self.last_my is None:
            return None
        counts = self.resp[MOVE_TO_IDX[self.last_my]]
        total = sum(counts)
        if total < 1:
            return None
        idx = counts.index(max(counts))
        if counts[idx] / total < 0.38:
            return None
        self.last_prediction = IDX_TO_MOVE[idx]
        return self.last_prediction
    
    def observe(self, my_move: str, opp_move: str):
        if self.last_my is not None:
            last_idx = MOVE_TO_IDX[self.last_my]
            for i in range(3):
                self.resp[last_idx][i] *= self.decay
            self.resp[last_idx][MOVE_TO_IDX[opp_move]] += 1
        self.last_my = my_move


class PatternPredictor(Predictor):
    """N-gram pattern matching."""
    
    def __init__(self, n: int = 3):
        super().__init__(f'pat{n}')
        self.n = n
        self.patterns: Dict[str, List[int]] = {}
        self.history: List[str] = []
    
    def predict(self, brain: 'RPSBrainV9') -> Optional[str]:
        if len(self.history) < self.n:
            return None
        key = ''.join(self.history[-self.n:])
        if key not in self.patterns:
            return None
        counts = self.patterns[key]
        total = sum(counts)
        if total < 2:
            return None
        idx = counts.index(max(counts))
        if counts[idx] / total < 0.40:
            return None
        self.last_prediction = IDX_TO_MOVE[idx]
        return self.last_prediction
    
    def observe(self, opp_move: str):
        if len(self.history) >= self.n:
            key = ''.join(self.history[-self.n:])
            if key not in self.patterns:
                self.patterns[key] = [0, 0, 0]
            self.patterns[key][MOVE_TO_IDX[opp_move]] += 1
        self.history.append(opp_move)
        if len(self.history) > 60:
            self.history.pop(0)


class MetaPredictor(Predictor):
    """Meta-level predictor - Iocaine Powder style."""
    
    def __init__(self, base: Predictor, level: int):
        super().__init__(f'{base.name}_p{level}')
        self.base = base
        self.level = level
    
    def predict(self, brain: 'RPSBrainV9') -> Optional[str]:
        base_pred = self.base.predict(brain)
        if base_pred is None:
            return None
        # Apply meta-level
        pred = base_pred
        for _ in range(self.level):
            pred = BEATS[pred]  # Go one level deeper
        self.last_prediction = pred
        return pred


class RPSBrainV9:
    VERSION = "9.0"
    
    def __init__(self):
        self.my_history: List[str] = []
        self.opp_history: List[str] = []
        self.results: List[str] = []
        
        self.my_score = 0
        self.opp_score = 0
        self.consecutive_losses = 0
        
        # Simple predictors
        self.freq = FrequencyPredictor()
        self.trans = TransitionPredictor()
        self.resp = ResponsePredictor()
        self.pat2 = PatternPredictor(2)
        self.pat3 = PatternPredictor(3)
        
        # All predictors including meta-levels (Iocaine style)
        self.predictors: List[Predictor] = [
            self.freq,
            MetaPredictor(self.freq, 1),
            MetaPredictor(self.freq, 2),
            self.trans,
            MetaPredictor(self.trans, 1),
            MetaPredictor(self.trans, 2),
            self.resp,
            MetaPredictor(self.resp, 1),
            MetaPredictor(self.resp, 2),
            self.pat2,
            MetaPredictor(self.pat2, 1),
            self.pat3,
            MetaPredictor(self.pat3, 1),
        ]
        
        # Move tracking
        self.last_moves: List[str] = []
        self.move_losses: Dict[str, int] = {'rock': 0, 'paper': 0, 'scissors': 0}
        self.move_counts: Dict[str, int] = {'rock': 0, 'paper': 0, 'scissors': 0}
    
    def _detect_spam(self, window: int = 10, threshold: float = 0.55) -> Optional[str]:
        """Detect if opponent is spamming a single move."""
        if len(self.opp_history) < window:
            return None
        recent = self.opp_history[-window:]
        counts = Counter(recent)
        for move, count in counts.items():
            if count / window >= threshold:
                return move
        return None
    
    def _is_exploited(self, move: str) -> bool:
        """Check if we're being exploited on a move."""
        if self.move_counts[move] < 5:
            return False
        return self.move_losses[move] / self.move_counts[move] > 0.55
    
    def update(self, my_move: str, opp_move: str, winner: str):
        """Update all state."""
        self.my_history.append(my_move)
        self.opp_history.append(opp_move)
        self.results.append(winner)
        
        # Scores
        if winner == 'you':
            self.my_score += 1
            self.consecutive_losses = 0
        elif winner == 'opponent':
            self.opp_score += 1
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        
        # Move tracking
        self.move_counts[my_move] += 1
        if winner == 'opponent':
            self.move_losses[my_move] += 1
        
        self.last_moves.append(my_move)
        if len(self.last_moves) > 10:
            self.last_moves.pop(0)
        
        # Update all predictors
        for p in self.predictors:
            p.update(opp_move)
        
        # Update base predictor observations
        self.freq.observe(opp_move)
        self.trans.observe(opp_move)
        self.resp.observe(my_move, opp_move)
        self.pat2.observe(opp_move)
        self.pat3.observe(opp_move)
    
    def choose_move(self) -> Tuple[str, str]:
        """Choose move - SIMPLE approach."""
        round_num = len(self.opp_history) + 1
        
        # Early game: random with slight paper bias
        if round_num <= 5:
            move = random.choices(MOVES, [0.30, 0.40, 0.30])[0]
            return move, f"R{round_num}: Early random. Score: {self.my_score}-{self.opp_score}"
        
        # Emergency: 5+ consecutive losses = full random reset
        if self.consecutive_losses >= 5:
            self.consecutive_losses = 0
            move = random.choice(MOVES)
            return move, f"R{round_num}: EMERGENCY reset. Score: {self.my_score}-{self.opp_score}"
        
        # Priority 1: Counter spam (if detected)
        spam_move = self._detect_spam()
        if spam_move:
            counter = BEATS[spam_move]
            # 85% counter, 15% noise
            if random.random() < 0.85:
                return counter, f"R{round_num}: SPAM({spam_move})→{counter}. Score: {self.my_score}-{self.opp_score}"
            else:
                move = random.choice(MOVES)
                return move, f"R{round_num}: Spam noise. Score: {self.my_score}-{self.opp_score}"
        
        # REMOVED: Avoid exploited move - this made us 2-move predictable!
        # The predictors will naturally adapt; explicit avoidance is worse
        
        # Main strategy: Get predictions, pick best scorer
        predictions = []
        for p in self.predictors:
            pred = p.predict(self)
            if pred is not None:
                score = p.get_score()
                predictions.append((score, pred, p.name))
        
        if predictions:
            # Sort by score, pick best
            predictions.sort(reverse=True)
            best_score, best_pred, best_name = predictions[0]
            
            if best_score > 0:
                move = BEATS[best_pred]  # Counter the predicted move
                
                # Fixed 10% noise - simple, effective
                if random.random() < 0.10:
                    move = random.choice(MOVES)
                    return move, f"R{round_num}: Noise. Score: {self.my_score}-{self.opp_score}"
                
                return move, f"R{round_num}: {best_name}({best_score:.1f})→{move}. Score: {self.my_score}-{self.opp_score}"
        
        # Fallback: frequency-based or random
        if self.opp_history:
            counts = Counter(self.opp_history[-15:])
            most_common = counts.most_common(1)[0][0]
            move = BEATS[most_common]
            return move, f"R{round_num}: Freq fallback→{move}. Score: {self.my_score}-{self.opp_score}"
        
        move = random.choice(MOVES)
        return move, f"R{round_num}: Random. Score: {self.my_score}-{self.opp_score}"


# Compatibility
brain = RPSBrainV9()

def get_brain():
    return brain

def reset_brain():
    global brain
    brain = RPSBrainV9()
    return brain


if __name__ == "__main__":
    print(f"RPS Brain v{VERSION}")
    b = RPSBrainV9()
    
    # Quick test
    test_moves = ['rock', 'rock', 'rock', 'paper', 'paper', 'scissors']
    for opp in test_moves:
        my_move, reason = b.choose_move()
        winner = 'you' if BEATS[opp] == my_move else ('opponent' if LOSES_TO[opp] == my_move else 'tie')
        b.update(my_move, opp, winner)
        print(f"{my_move} vs {opp} = {winner} | {reason}")
