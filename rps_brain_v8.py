#!/usr/bin/env python3
"""
Mentius RPS Brain v8.0 - Championship Edition

Key improvements over v7:
1. Late-game adaptation (rounds 40+ special handling)
2. Faster predictor windows (3 rounds with exponential decay)
3. Dynamic noise injection based on game state
4. Score-aware predictor weighting
5. Forced diversity after repetition
6. Opponent spam detection with adaptive thresholds
7. Multi-pattern detection (short + long term)
8. Emergency mode improvements
9. Momentum tracking and exploitation
10. Prepared for TAO model integration

Based on research from:
- Multi-AI Markov Models (Wang et al. 2020)
- DeepMind Population-based RPS Benchmark (2023)
- IO2_fightinguuu (78.74% win rate)
- TAO: Transformer Against Opponent (ICLR 2024)
"""

import random
import math
from collections import defaultdict, Counter
from typing import Optional, Tuple, List, Dict

# Constants
MOVES = ['rock', 'paper', 'scissors']
BEATS = {'rock': 'paper', 'paper': 'scissors', 'scissors': 'rock'}
LOSES_TO = {'rock': 'scissors', 'paper': 'rock', 'scissors': 'paper'}
MOVE_TO_IDX = {'rock': 0, 'paper': 1, 'scissors': 2}
IDX_TO_MOVE = {0: 'rock', 1: 'paper', 2: 'scissors'}


class Predictor:
    """Base class for predictors with fast adaptation."""

    def __init__(self, name: str, window: int = 8):
        self.name = name
        self.predictions: List[Optional[str]] = []
        self.scores: List[int] = []  # 1 if correct, 0 if wrong
        self.window = window

    def predict(self, brain: 'RPSBrainV8') -> Optional[str]:
        """Return predicted opponent move or None."""
        raise NotImplementedError

    def update(self, actual_opp: str, prediction: Optional[str]):
        """Update score based on prediction accuracy."""
        if prediction is not None:
            correct = 1 if prediction == actual_opp else 0
            self.scores.append(correct)
            if len(self.scores) > self.window:
                self.scores.pop(0)
        self.predictions.append(prediction)

    def get_score(self) -> float:
        """Exponentially weighted recent score."""
        if not self.scores:
            return 0.0
        score = 0.0
        for i, s in enumerate(self.scores):
            # Exponential recency weighting
            weight = math.exp(0.3 * i)
            score += weight if s else -weight * 0.5  # Penalize wrong less than reward correct
        return score

    def get_accuracy(self) -> float:
        """Get recent accuracy (0-1)."""
        if len(self.scores) < 3:
            return 0.5
        return sum(self.scores[-5:]) / min(5, len(self.scores))

    def get_confidence(self) -> float:
        """Confidence based on consistency."""
        if len(self.scores) < 3:
            return 0.3
        recent = self.scores[-5:]
        if len(recent) < 3:
            return 0.3
        return sum(recent) / len(recent)


class FrequencyPredictor(Predictor):
    """Predict opponent's most common move with fast decay."""

    def __init__(self, decay: float = 0.85, window: int = 8):
        super().__init__('freq', window)
        self.counts = [0.0, 0.0, 0.0]
        self.decay = decay

    def predict(self, brain: 'RPSBrainV8') -> Optional[str]:
        total = sum(self.counts)
        if total < 2:
            return None
        idx = self.counts.index(max(self.counts))
        # Only predict if confident
        if self.counts[idx] / total < 0.35:
            return None
        return IDX_TO_MOVE[idx]

    def observe(self, opp_move: str):
        for i in range(3):
            self.counts[i] *= self.decay
        self.counts[MOVE_TO_IDX[opp_move]] += 1


class TransitionPredictor(Predictor):
    """Predict based on opponent's transition patterns."""

    def __init__(self, decay: float = 0.8, window: int = 8):
        super().__init__('trans', window)
        self.trans = [[0.0, 0.0, 0.0] for _ in range(3)]
        self.decay = decay
        self.last_opp: Optional[str] = None

    def predict(self, brain: 'RPSBrainV8') -> Optional[str]:
        if self.last_opp is None:
            return None
        counts = self.trans[MOVE_TO_IDX[self.last_opp]]
        total = sum(counts)
        if total < 2:
            return None
        idx = counts.index(max(counts))
        if counts[idx] / total < 0.40:
            return None
        return IDX_TO_MOVE[idx]

    def observe(self, opp_move: str):
        if self.last_opp is not None:
            last_idx = MOVE_TO_IDX[self.last_opp]
            for i in range(3):
                self.trans[last_idx][i] *= self.decay
            self.trans[last_idx][MOVE_TO_IDX[opp_move]] += 1
        self.last_opp = opp_move


class ResponsePredictor(Predictor):
    """Predict based on how opponent responds to our moves."""

    def __init__(self, decay: float = 0.8, window: int = 8):
        super().__init__('resp', window)
        self.resp = [[0.0, 0.0, 0.0] for _ in range(3)]
        self.decay = decay
        self.last_my: Optional[str] = None

    def predict(self, brain: 'RPSBrainV8') -> Optional[str]:
        if self.last_my is None:
            return None
        counts = self.resp[MOVE_TO_IDX[self.last_my]]
        total = sum(counts)
        if total < 2:
            return None
        idx = counts.index(max(counts))
        if counts[idx] / total < 0.40:
            return None
        return IDX_TO_MOVE[idx]

    def observe(self, my_move: str, opp_move: str):
        if self.last_my is not None:
            last_idx = MOVE_TO_IDX[self.last_my]
            for i in range(3):
                self.resp[last_idx][i] *= self.decay
            self.resp[last_idx][MOVE_TO_IDX[opp_move]] += 1
        self.last_my = my_move


class WSLSPredictor(Predictor):
    """Win-Stay-Lose-Shift detector with pattern tracking."""

    def __init__(self, window: int = 8):
        super().__init__('wsls', window)
        self.win_stay = 0
        self.win_shift = 0
        self.lose_stay = 0
        self.lose_shift = 0
        self.last_opp: Optional[str] = None
        self.last_result: Optional[str] = None
        # Track specific shift destinations
        self.shift_to = {m: [0, 0, 0] for m in MOVES}

    def predict(self, brain: 'RPSBrainV8') -> Optional[str]:
        if self.last_opp is None or self.last_result is None:
            return None

        if self.last_result == 'win':
            total = self.win_stay + self.win_shift
            if total < 3:
                return None
            if self.win_stay > self.win_shift * 1.3:
                return self.last_opp
            elif self.win_shift > self.win_stay * 1.3:
                # Check shift destination
                shift_counts = self.shift_to[self.last_opp]
                if sum(shift_counts) >= 3:
                    best_idx = shift_counts.index(max(shift_counts))
                    if shift_counts[best_idx] / sum(shift_counts) > 0.45:
                        return IDX_TO_MOVE[best_idx]
                return random.choice([m for m in MOVES if m != self.last_opp])

        elif self.last_result == 'lose':
            total = self.lose_stay + self.lose_shift
            if total < 3:
                return None
            if self.lose_shift > self.lose_stay * 1.3:
                shift_counts = self.shift_to[self.last_opp]
                if sum(shift_counts) >= 3:
                    best_idx = shift_counts.index(max(shift_counts))
                    if shift_counts[best_idx] / sum(shift_counts) > 0.45:
                        return IDX_TO_MOVE[best_idx]
                return random.choice([m for m in MOVES if m != self.last_opp])
            elif self.lose_stay > self.lose_shift * 1.3:
                return self.last_opp

        return None

    def observe(self, my_move: str, opp_move: str, winner: str):
        if self.last_opp is not None and self.last_result is not None:
            stayed = (opp_move == self.last_opp)

            if self.last_result == 'win':
                if stayed:
                    self.win_stay += 1
                else:
                    self.win_shift += 1
                    self.shift_to[self.last_opp][MOVE_TO_IDX[opp_move]] += 1
            elif self.last_result == 'lose':
                if stayed:
                    self.lose_stay += 1
                else:
                    self.lose_shift += 1
                    self.shift_to[self.last_opp][MOVE_TO_IDX[opp_move]] += 1

        self.last_opp = opp_move
        if winner == 'opponent':
            self.last_result = 'win'
        elif winner == 'you':
            self.last_result = 'lose'
        else:
            self.last_result = 'tie'


class PatternPredictor(Predictor):
    """N-gram pattern matching with variable lengths."""

    def __init__(self, n: int = 3, window: int = 8):
        super().__init__(f'pat{n}', window)
        self.n = n
        self.opp_history = ""

    def predict(self, brain: 'RPSBrainV8') -> Optional[str]:
        if len(self.opp_history) < self.n + 2:
            return None

        pattern = self.opp_history[-self.n:]
        counts = {'R': 0, 'P': 0, 'S': 0}

        for i in range(len(self.opp_history) - self.n):
            if self.opp_history[i:i+self.n] == pattern:
                if i + self.n < len(self.opp_history):
                    next_move = self.opp_history[i + self.n]
                    counts[next_move] += 1

        total = sum(counts.values())
        if total < 2:
            return None

        best = max(counts, key=counts.get)
        if counts[best] / total < 0.45:
            return None

        return {'R': 'rock', 'P': 'paper', 'S': 'scissors'}[best]

    def observe(self, opp_move: str):
        self.opp_history += {'rock': 'R', 'paper': 'P', 'scissors': 'S'}[opp_move]


class MomentumPredictor(Predictor):
    """Detect momentum and streaks."""

    def __init__(self, window: int = 8):
        super().__init__('momentum', window)
        self.recent_opp_moves: List[str] = []
        self.recent_results: List[str] = []
        self.streak_window = 6

    def predict(self, brain: 'RPSBrainV8') -> Optional[str]:
        if len(self.recent_opp_moves) < 4:
            return None

        # Check for opponent streak (same move 3+ times)
        if len(set(self.recent_opp_moves[-3:])) == 1:
            return self.recent_opp_moves[-1]

        # Check for cycling pattern (R-P-S or S-P-R)
        if len(self.recent_opp_moves) >= 3:
            last3 = self.recent_opp_moves[-3:]
            if len(set(last3)) == 3:  # All different
                # Forward cycle: R->P->S->R
                fwd_next = {'rock': 'paper', 'paper': 'scissors', 'scissors': 'rock'}
                # Backward cycle: R->S->P->R
                bwd_next = {'rock': 'scissors', 'scissors': 'paper', 'paper': 'rock'}

                # Check if following forward cycle
                if last3[1] == fwd_next[last3[0]] and last3[2] == fwd_next[last3[1]]:
                    return fwd_next[last3[2]]

                # Check if following backward cycle
                if last3[1] == bwd_next[last3[0]] and last3[2] == bwd_next[last3[1]]:
                    return bwd_next[last3[2]]

        # Check if opponent is on winning streak (might get overconfident)
        opp_wins = self.recent_results[-5:].count('opponent')
        if opp_wins >= 4 and len(set(self.recent_opp_moves[-3:])) == 1:
            return self.recent_opp_moves[-1]  # They'll likely repeat

        return None

    def observe(self, opp_move: str, winner: str):
        self.recent_opp_moves.append(opp_move)
        self.recent_results.append(winner)
        if len(self.recent_opp_moves) > self.streak_window:
            self.recent_opp_moves.pop(0)
            self.recent_results.pop(0)


class MetaPredictor(Predictor):
    """Meta-level predictor (P.1, P.2)."""

    def __init__(self, base: Predictor, level: int):
        super().__init__(f'{base.name}_p{level}', base.window)
        self.base = base
        self.level = level

    def predict(self, brain: 'RPSBrainV8') -> Optional[str]:
        base_pred = self.base.predict(brain)
        if base_pred is None:
            return None

        pred = base_pred
        for _ in range(self.level):
            pred = BEATS[pred]
        return pred


class RPSBrainV8:
    """Championship-level RPS brain with all optimizations."""

    VERSION = "8.0"

    def __init__(self, focus_length: int = 5):
        self.my_history: List[str] = []
        self.opp_history: List[str] = []
        self.results: List[str] = []

        # Focus length for adaptation speed
        self.focus_length = focus_length

        # Game state tracking
        self.my_score = 0
        self.opp_score = 0
        self.consecutive_losses = 0
        self.consecutive_wins = 0

        # Move tracking for anti-exploitation
        self.move_counts: Dict[str, int] = {'rock': 0, 'paper': 0, 'scissors': 0}
        self.move_losses: Dict[str, int] = {'rock': 0, 'paper': 0, 'scissors': 0}
        self.last_moves: List[str] = []  # Our last N moves

        # Performance tracking
        self.performance_window: List[str] = []  # W/L/T

        # Initialize predictors with faster windows
        self.freq = FrequencyPredictor(decay=0.80, window=6)
        self.trans = TransitionPredictor(decay=0.75, window=6)
        self.resp = ResponsePredictor(decay=0.75, window=6)
        self.wsls = WSLSPredictor(window=6)
        self.pat2 = PatternPredictor(n=2, window=6)
        self.pat3 = PatternPredictor(n=3, window=6)
        self.pat4 = PatternPredictor(n=4, window=6)
        self.momentum = MomentumPredictor(window=6)

        # Long-term predictors (slower adaptation)
        self.freq_long = FrequencyPredictor(decay=0.92, window=12)
        self.trans_long = TransitionPredictor(decay=0.90, window=12)

        # Build predictor list with meta-levels
        self.predictors: List[Predictor] = [
            # Fast predictors
            self.freq,
            MetaPredictor(self.freq, 1),
            MetaPredictor(self.freq, 2),
            self.trans,
            MetaPredictor(self.trans, 1),
            MetaPredictor(self.trans, 2),
            self.resp,
            MetaPredictor(self.resp, 1),
            MetaPredictor(self.resp, 2),
            self.wsls,
            MetaPredictor(self.wsls, 1),
            self.pat2,
            MetaPredictor(self.pat2, 1),
            self.pat3,
            MetaPredictor(self.pat3, 1),
            self.pat4,
            self.momentum,
            MetaPredictor(self.momentum, 1),
            # Long-term predictors
            self.freq_long,
            MetaPredictor(self.freq_long, 1),
            self.trans_long,
            MetaPredictor(self.trans_long, 1),
        ]

        # TAO model placeholder (for future integration)
        self.tao_model = None

    def _get_round(self) -> int:
        """Current round number (1-indexed)."""
        return len(self.opp_history) + 1

    def _get_game_phase(self) -> str:
        """Determine game phase for strategy adjustment."""
        round_num = self._get_round()
        if round_num <= 10:
            return 'early'
        elif round_num <= 40:
            return 'mid'
        elif round_num <= 70:
            return 'late'
        else:
            return 'endgame'

    def _get_recent_win_rate(self, n: int = 10) -> float:
        """Get win rate over last n rounds."""
        if len(self.results) < n:
            recent = self.results
        else:
            recent = self.results[-n:]

        if not recent:
            return 0.5

        wins = recent.count('you')
        losses = recent.count('opponent')
        total = wins + losses

        if total == 0:
            return 0.5
        return wins / total

    def _detect_opponent_spam(self, window: int = 12, threshold: float = 0.50) -> Optional[str]:
        """Detect if opponent is spamming a single move."""
        if len(self.opp_history) < window:
            return None

        recent = self.opp_history[-window:]
        counts = Counter(recent)

        for move, count in counts.items():
            if count / window >= threshold:
                return move
        return None

    def _detect_our_exploitation(self) -> Optional[str]:
        """Detect if we're being exploited on a specific move."""
        for move in MOVES:
            if self.move_counts[move] >= 5:
                loss_rate = self.move_losses[move] / self.move_counts[move]
                if loss_rate > 0.55:
                    return move
        return None

    def _get_dynamic_noise(self) -> float:
        """Calculate noise rate based on game state - SCORE AWARE."""
        round_num = self._get_round()
        phase = self._get_game_phase()
        score_diff = self.my_score - self.opp_score
        recent_wr = self._get_recent_win_rate(8)

        base_noise = 0.08

        # Phase adjustments
        if phase == 'early':
            base_noise = 0.12  # More exploration early
        elif phase == 'mid':
            base_noise = 0.10
        elif phase in ('late', 'endgame'):
            # SCORE-BASED noise in late game
            if score_diff >= 5:
                base_noise = 0.30  # Big lead - be unpredictable
            elif score_diff >= 2:
                base_noise = 0.20  # Small lead - moderate noise
            elif score_diff <= -3:
                base_noise = 0.05  # Behind - trust predictions, minimize noise
            elif score_diff < 0:
                base_noise = 0.08  # Slightly behind - low noise
            else:
                base_noise = 0.12  # Close game - balanced

        # Consecutive loss adjustment - but NOT when ahead
        if self.consecutive_losses >= 3 and score_diff < 3:
            base_noise = max(base_noise, 0.15)
        if self.consecutive_losses >= 5 and score_diff < 0:
            base_noise = max(base_noise, 0.20)

        return min(base_noise, 0.35)  # Cap at 35%

    def _force_move_diversity(self, proposed_move: str) -> str:
        """Force diversity if we've been too predictable."""
        if len(self.last_moves) < 4:
            return proposed_move

        # Check if we've played same move 3+ times recently
        if self.last_moves[-3:].count(proposed_move) >= 3:
            alternatives = [m for m in MOVES if m != proposed_move]
            return random.choice(alternatives)

        # Check overall recent distribution
        recent = self.last_moves[-8:]
        counts = Counter(recent)
        if counts[proposed_move] >= 5:
            alternatives = [m for m in MOVES if m != proposed_move]
            return random.choice(alternatives)

        return proposed_move

    def update(self, my_move: str, opp_move: str, winner: str):
        """Update all state with match result."""
        self.my_history.append(my_move)
        self.opp_history.append(opp_move)
        self.results.append(winner)

        # Update scores
        if winner == 'you':
            self.my_score += 1
            self.consecutive_losses = 0
            self.consecutive_wins += 1
        elif winner == 'opponent':
            self.opp_score += 1
            self.consecutive_losses += 1
            self.consecutive_wins = 0
        else:
            self.consecutive_losses = 0
            self.consecutive_wins = 0

        # Update move tracking
        self.move_counts[my_move] += 1
        if winner == 'opponent':
            self.move_losses[my_move] += 1

        self.last_moves.append(my_move)
        if len(self.last_moves) > 15:
            self.last_moves.pop(0)

        # Update performance window
        self.performance_window.append('W' if winner == 'you' else 'L' if winner == 'opponent' else 'T')
        if len(self.performance_window) > 20:
            self.performance_window.pop(0)

        # Update all predictors
        for p in self.predictors:
            pred = p.predictions[-1] if p.predictions else None
            p.update(opp_move, pred)

        # Update base predictor observations
        self.freq.observe(opp_move)
        self.freq_long.observe(opp_move)
        self.trans.observe(opp_move)
        self.trans_long.observe(opp_move)
        self.resp.observe(my_move, opp_move)
        self.wsls.observe(my_move, opp_move, winner)
        self.pat2.observe(opp_move)
        self.pat3.observe(opp_move)
        self.pat4.observe(opp_move)
        self.momentum.observe(opp_move, winner)

    def choose_move(self) -> Tuple[str, str]:
        """Choose the best move based on all available information."""
        round_num = self._get_round()
        phase = self._get_game_phase()

        # Phase 1: Early game exploration (rounds 1-5)
        if round_num <= 5:
            move = random.choice(MOVES)
            reason = f"R{round_num}: Early explore. Score: {self.my_score}-{self.opp_score}"
            for p in self.predictors:
                p.predictions.append(None)
            return move, reason

        # Emergency mode: 5+ consecutive losses
        if self.consecutive_losses >= 5:
            move = random.choice(MOVES)
            reason = f"R{round_num}: EMERGENCY ({self.consecutive_losses} losses). Score: {self.my_score}-{self.opp_score}"
            for p in self.predictors:
                p.predictions.append(None)
            self.consecutive_losses = 0  # Reset to avoid infinite emergency
            return move, reason

        # Check for opponent spam (priority counter)
        spam_move = self._detect_opponent_spam()
        if spam_move:
            counter = BEATS[spam_move]
            # High confidence counter, but with some noise
            if random.random() < 0.85:
                move = counter
                reason = f"R{round_num}: SPAM({spam_move})→{counter}. Score: {self.my_score}-{self.opp_score}"
            else:
                move = random.choice(MOVES)
                reason = f"R{round_num}: Spam noise. Score: {self.my_score}-{self.opp_score}"
            for p in self.predictors:
                p.predictions.append(None)
            return self._force_move_diversity(move), reason

        # Check if we're being exploited
        exploited_move = self._detect_our_exploitation()
        if exploited_move:
            alternatives = [m for m in MOVES if m != exploited_move]
            move = random.choice(alternatives)
            reason = f"R{round_num}: AVOID({exploited_move}). Score: {self.my_score}-{self.opp_score}"
            for p in self.predictors:
                p.predictions.append(None)
            # Reset exploitation tracking for this move
            self.move_counts[exploited_move] = 0
            self.move_losses[exploited_move] = 0
            return move, reason

        # Late-game special handling - SCORE-AWARE
        if phase in ('late', 'endgame'):
            recent_wr = self._get_recent_win_rate(8)
            very_recent_wr = self._get_recent_win_rate(5)
            score_diff = self.my_score - self.opp_score
            
            # AHEAD: Protect lead with controlled unpredictability
            if score_diff >= 5:
                # Big lead - pure random to prevent being read
                if random.random() < 0.35:
                    move = random.choice(MOVES)
                    reason = f"R{round_num}: Lead protect (+{score_diff}). Score: {self.my_score}-{self.opp_score}"
                    for p in self.predictors:
                        p.predictions.append(None)
                    return move, reason
            elif score_diff >= 2:
                # Small lead - some randomness
                if random.random() < 0.20:
                    move = random.choice(MOVES)
                    reason = f"R{round_num}: Hold lead (+{score_diff}). Score: {self.my_score}-{self.opp_score}"
                    for p in self.predictors:
                        p.predictions.append(None)
                    return move, reason
            
            # BEHIND or CLOSE: Be aggressive but smart
            if score_diff <= -3:
                # Behind - trust best predictor MORE, reduce noise
                # Don't reset randomly when behind - we need wins not chaos
                pass  # Let predictor logic handle it with lower noise
            elif very_recent_wr < 0.25 and score_diff < 0:
                # Badly losing momentum AND behind - try meta-shift
                # But don't full reset - that's giving up
                if random.random() < 0.30:
                    move = random.choice(MOVES)
                    reason = f"R{round_num}: Desperation ({score_diff}). Score: {self.my_score}-{self.opp_score}"
                    for p in self.predictors:
                        p.predictions.append(None)
                    return self._force_move_diversity(move), reason

        # Get predictions from all predictors
        predictions: Dict[str, Tuple[str, float, float]] = {}
        for p in self.predictors:
            pred = p.predict(self)
            p.predictions.append(pred)
            if pred is not None:
                score = p.get_score()
                confidence = p.get_confidence()
                predictions[p.name] = (pred, score, confidence)

        # Find best predictor - PHASE-AWARE SELECTION
        best_pred = None
        best_weighted_score = -float('inf')
        best_name = None

        for name, (pred, score, confidence) in predictions.items():
            weighted = score * (0.5 + confidence)

            # Phase-specific predictor weighting
            if phase == 'early':
                # Trust frequency-based early
                if 'freq' in name:
                    weighted *= 1.3
            elif phase == 'mid':
                # Balanced approach
                pass
            elif phase in ('late', 'endgame'):
                # In late game, DISTRUST predictors that worked early
                # They've likely been figured out
                if score > 5:  # If predictor was doing well...
                    weighted *= 0.6  # ...opponent probably adapted to it
                # Prefer meta-level predictors in late game
                if '_p1' in name or '_p2' in name:
                    weighted *= 1.4
                # Prefer momentum-based in endgame
                if 'momentum' in name and phase == 'endgame':
                    weighted *= 1.5

            if weighted > best_weighted_score:
                best_weighted_score = weighted
                best_pred = pred
                best_name = name

        # Apply noise injection
        noise_rate = self._get_dynamic_noise()

        # Late-game counter-prediction: if we're being read, go one level deeper
        if phase in ('late', 'endgame') and best_pred is not None:
            recent_wr = self._get_recent_win_rate(6)
            score_diff = self.my_score - self.opp_score
            
            # Only meta-shift if BEHIND and losing - don't mess with what works when ahead
            if recent_wr < 0.40 and score_diff < 0:
                # They're predicting us AND we're behind - counter their counter
                if random.random() < 0.40:
                    best_pred = BEATS[best_pred]  # Shift to next meta-level
                    best_name = f"{best_name}+meta"
            elif score_diff >= 3:
                # We're ahead - inject pure randomness instead of meta-gaming
                if random.random() < 0.25:
                    best_pred = None  # Force random fallback

        if best_pred is not None and best_weighted_score > 0:
            move = BEATS[best_pred]

            if random.random() < noise_rate:
                move = random.choice(MOVES)
                reason = f"R{round_num}: Noise({noise_rate:.0%}). Score: {self.my_score}-{self.opp_score}"
            else:
                reason = f"R{round_num}: {best_name}({best_weighted_score:.1f})→{move}. Score: {self.my_score}-{self.opp_score}"
        else:
            # Fallback to frequency or random
            freq_pred = self.freq.predict(self)
            if freq_pred:
                move = BEATS[freq_pred]
                reason = f"R{round_num}: Freq fallback→{move}. Score: {self.my_score}-{self.opp_score}"
            else:
                move = random.choice(MOVES)
                reason = f"R{round_num}: Random fallback. Score: {self.my_score}-{self.opp_score}"

        # Force diversity check
        move = self._force_move_diversity(move)

        return move, reason

    def load_tao_model(self, model_path: str):
        """Load TAO model for inference (future integration)."""
        # Placeholder for TAO model loading
        # self.tao_model = TAOModel.load(model_path)
        pass


# Singleton instance
brain = RPSBrainV8()


def get_move(history_list: List[dict]) -> Tuple[str, str]:
    """Entry point for play_match.py."""
    global brain

    for i, h in enumerate(history_list):
        if i >= len(brain.opp_history):
            brain.update(h['your_move'], h['opponent_move'], h['winner'])

    return brain.choose_move()


def reset_brain():
    """Reset for new match."""
    global brain
    brain = RPSBrainV8()


# =============================================================================
# Testing Suite
# =============================================================================

def test_against_rock_bias():
    """Test against rock-biased opponent."""
    brain = RPSBrainV8()
    wins, losses, ties = 0, 0, 0

    for i in range(100):
        move, _ = brain.choose_move()
        opp = random.choices(MOVES, weights=[0.55, 0.25, 0.20])[0]

        if move == BEATS[opp]:
            winner = 'you'
            wins += 1
        elif opp == BEATS[move]:
            winner = 'opponent'
            losses += 1
        else:
            winner = 'tie'
            ties += 1

        brain.update(move, opp, winner)

    wr = wins / (wins + losses) * 100 if wins + losses > 0 else 0
    return wins, losses, ties, wr


def test_against_counter_exploiter():
    """Test against opponent who counters our most common move."""
    brain = RPSBrainV8()
    wins, losses, ties = 0, 0, 0
    my_moves = []

    for i in range(100):
        move, _ = brain.choose_move()
        my_moves.append(move)

        if len(my_moves) >= 8:
            counts = Counter(my_moves[-12:])
            most_common = counts.most_common(1)[0][0]
            opp = BEATS[most_common]
        else:
            opp = random.choice(MOVES)

        if move == BEATS[opp]:
            winner = 'you'
            wins += 1
        elif opp == BEATS[move]:
            winner = 'opponent'
            losses += 1
        else:
            winner = 'tie'
            ties += 1

        brain.update(move, opp, winner)

    wr = wins / (wins + losses) * 100 if wins + losses > 0 else 0
    return wins, losses, ties, wr


def test_against_scissors_spam():
    """Test against scissors spammer (like JARVIS)."""
    brain = RPSBrainV8()
    wins, losses, ties = 0, 0, 0

    for i in range(100):
        move, _ = brain.choose_move()

        # Spam scissors 60% of the time
        if random.random() < 0.60:
            opp = 'scissors'
        else:
            opp = random.choice(['rock', 'paper'])

        if move == BEATS[opp]:
            winner = 'you'
            wins += 1
        elif opp == BEATS[move]:
            winner = 'opponent'
            losses += 1
        else:
            winner = 'tie'
            ties += 1

        brain.update(move, opp, winner)

    wr = wins / (wins + losses) * 100 if wins + losses > 0 else 0
    return wins, losses, ties, wr


def test_against_wsls():
    """Test against Win-Stay-Lose-Shift opponent."""
    brain = RPSBrainV8()
    wins, losses, ties = 0, 0, 0
    opp_move = random.choice(MOVES)
    last_opp_won = False

    for i in range(100):
        move, _ = brain.choose_move()

        # WSLS logic
        if i > 0 and not last_opp_won:
            opp_move = random.choice([m for m in MOVES if m != opp_move])

        if move == BEATS[opp_move]:
            winner = 'you'
            wins += 1
            last_opp_won = False
        elif opp_move == BEATS[move]:
            winner = 'opponent'
            losses += 1
            last_opp_won = True
        else:
            winner = 'tie'
            ties += 1

        brain.update(move, opp_move, winner)

    wr = wins / (wins + losses) * 100 if wins + losses > 0 else 0
    return wins, losses, ties, wr


def test_late_game_adaptation():
    """Test late-game performance against adapting opponent."""
    brain = RPSBrainV8()
    wins, losses, ties = 0, 0, 0
    my_moves = []

    for i in range(99):  # Full match
        move, _ = brain.choose_move()
        my_moves.append(move)

        # Opponent adapts more aggressively after round 40
        if i < 40:
            # Random-ish early
            opp = random.choices(MOVES, weights=[0.35, 0.35, 0.30])[0]
        else:
            # Counter our most common move aggressively
            if len(my_moves) >= 5:
                counts = Counter(my_moves[-8:])
                most_common = counts.most_common(1)[0][0]
                if random.random() < 0.75:  # 75% counter
                    opp = BEATS[most_common]
                else:
                    opp = random.choice(MOVES)
            else:
                opp = random.choice(MOVES)

        if move == BEATS[opp]:
            winner = 'you'
            wins += 1
        elif opp == BEATS[move]:
            winner = 'opponent'
            losses += 1
        else:
            winner = 'tie'
            ties += 1

        brain.update(move, opp, winner)

    # Calculate early vs late performance
    early_wins = sum(1 for r in brain.results[:40] if r == 'you')
    early_losses = sum(1 for r in brain.results[:40] if r == 'opponent')
    late_wins = sum(1 for r in brain.results[40:] if r == 'you')
    late_losses = sum(1 for r in brain.results[40:] if r == 'opponent')

    early_wr = early_wins / (early_wins + early_losses) * 100 if early_wins + early_losses > 0 else 0
    late_wr = late_wins / (late_wins + late_losses) * 100 if late_wins + late_losses > 0 else 0
    total_wr = wins / (wins + losses) * 100 if wins + losses > 0 else 0

    return {
        'total': (wins, losses, ties, total_wr),
        'early': (early_wins, early_losses, early_wr),
        'late': (late_wins, late_losses, late_wr)
    }


if __name__ == "__main__":
    print(f"Mentius RPS Brain v{RPSBrainV8.VERSION} - Test Suite")
    print("=" * 60)

    print("\nTest 1: Rock-biased opponent (55% rock)")
    w, l, t, wr = test_against_rock_bias()
    print(f"  Result: {w}W-{l}L-{t}T = {wr:.1f}% win rate")

    print("\nTest 2: Counter-exploiter (counters our most common)")
    w, l, t, wr = test_against_counter_exploiter()
    print(f"  Result: {w}W-{l}L-{t}T = {wr:.1f}% win rate")

    print("\nTest 3: Scissors spammer (60% scissors)")
    w, l, t, wr = test_against_scissors_spam()
    print(f"  Result: {w}W-{l}L-{t}T = {wr:.1f}% win rate")

    print("\nTest 4: Win-Stay-Lose-Shift opponent")
    w, l, t, wr = test_against_wsls()
    print(f"  Result: {w}W-{l}L-{t}T = {wr:.1f}% win rate")

    print("\nTest 5: Late-game adaptation (opponent adapts after R40)")
    results = test_late_game_adaptation()
    w, l, t, wr = results['total']
    print(f"  Total: {w}W-{l}L-{t}T = {wr:.1f}% win rate")
    ew, el, ewr = results['early']
    print(f"  Early (R1-40): {ew}W-{el}L = {ewr:.1f}%")
    lw, ll, lwr = results['late']
    print(f"  Late (R41-99): {lw}W-{ll}L = {lwr:.1f}%")

    print("\n" + "=" * 60)
    print("Tests complete!")
