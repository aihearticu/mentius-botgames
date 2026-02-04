#!/usr/bin/env python3
"""
Opponent Profiler System

Real-time opponent profiling during matches:
- Load saved profiles by opponent name
- Update profiles from match history
- Provide counter-strategy recommendations
- Integrate with RPSBrainV8
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter
from dataclasses import dataclass, field

MOVES = ['rock', 'paper', 'scissors']
BEATS = {'rock': 'paper', 'paper': 'scissors', 'scissors': 'rock'}
LOSES_TO = {'rock': 'scissors', 'paper': 'rock', 'scissors': 'paper'}

DATA_DIR = Path(__file__).parent / "opponent_data"


@dataclass
class LiveProfile:
    """Real-time opponent profile built during a match."""

    name: str
    moves: List[str] = field(default_factory=list)
    transitions: Dict[str, Counter] = field(default_factory=lambda: {m: Counter() for m in MOVES})
    our_moves: List[str] = field(default_factory=list)
    results: List[str] = field(default_factory=list)

    # WSLS tracking
    win_stay: int = 0
    win_shift: int = 0
    lose_stay: int = 0
    lose_shift: int = 0
    last_opp_result: Optional[str] = None

    # Pattern cache
    _cached_strategy: Optional[str] = None
    _cache_valid_until: int = 0

    def update(self, my_move: str, opp_move: str, result: str):
        """Update profile with new round data."""
        # Update move history
        if self.moves:
            last_opp = self.moves[-1]
            self.transitions[last_opp][opp_move] += 1

        self.moves.append(opp_move)
        self.our_moves.append(my_move)
        self.results.append(result)

        # Update WSLS tracking
        if self.last_opp_result and len(self.moves) >= 2:
            stayed = (self.moves[-1] == self.moves[-2])
            if self.last_opp_result == 'win':
                if stayed:
                    self.win_stay += 1
                else:
                    self.win_shift += 1
            elif self.last_opp_result == 'lose':
                if stayed:
                    self.lose_stay += 1
                else:
                    self.lose_shift += 1

        # Track opponent's result perspective
        if result == 'you':
            self.last_opp_result = 'lose'
        elif result == 'opponent':
            self.last_opp_result = 'win'
        else:
            self.last_opp_result = 'tie'

        # Invalidate cache
        self._cached_strategy = None

    def get_frequency(self) -> Dict[str, float]:
        """Get move frequency distribution."""
        if not self.moves:
            return {m: 1/3 for m in MOVES}

        total = len(self.moves)
        return {m: self.moves.count(m) / total for m in MOVES}

    def get_dominant_move(self, threshold: float = 0.45) -> Optional[str]:
        """Get dominant move if above threshold."""
        freq = self.get_frequency()
        for move, pct in freq.items():
            if pct >= threshold:
                return move
        return None

    def get_transition_prediction(self) -> Optional[str]:
        """Predict next move based on transitions."""
        if not self.moves:
            return None

        last_move = self.moves[-1]
        trans = self.transitions[last_move]

        if sum(trans.values()) < 3:
            return None

        total = sum(trans.values())
        for move, count in trans.most_common(1):
            if count / total >= 0.50:
                return move

        return None

    def detect_wsls_pattern(self) -> Optional[str]:
        """Detect Win-Stay-Lose-Shift patterns."""
        total_win = self.win_stay + self.win_shift
        total_lose = self.lose_stay + self.lose_shift

        if total_win < 5 and total_lose < 5:
            return None

        # Check for strong patterns
        patterns = []

        if total_win >= 5:
            if self.win_stay / total_win > 0.65:
                patterns.append('win_stay')
            elif self.win_shift / total_win > 0.65:
                patterns.append('win_shift')

        if total_lose >= 5:
            if self.lose_stay / total_lose > 0.65:
                patterns.append('lose_stay')
            elif self.lose_shift / total_lose > 0.65:
                patterns.append('lose_shift')

        if patterns:
            return '_'.join(patterns)
        return None

    def classify_strategy(self) -> str:
        """Classify opponent's strategy in real-time."""
        if len(self.moves) < 8:
            return 'unknown'

        # Check cache
        if self._cached_strategy and len(self.moves) <= self._cache_valid_until:
            return self._cached_strategy

        # Check for spam first
        dominant = self.get_dominant_move(0.50)
        if dominant:
            self._cached_strategy = f'spam_{dominant}'
            self._cache_valid_until = len(self.moves) + 5
            return self._cached_strategy

        # Check for WSLS
        wsls = self.detect_wsls_pattern()
        if wsls:
            self._cached_strategy = f'wsls_{wsls}'
            self._cache_valid_until = len(self.moves) + 5
            return self._cached_strategy

        # Check for strong transitions
        for from_move, to_counts in self.transitions.items():
            total = sum(to_counts.values())
            if total >= 5:
                for to_move, count in to_counts.most_common(1):
                    if count / total >= 0.60:
                        self._cached_strategy = f'trans_{from_move}_{to_move}'
                        self._cache_valid_until = len(self.moves) + 5
                        return self._cached_strategy

        # Check for pattern (3+ consecutive same moves)
        if len(self.moves) >= 3:
            if len(set(self.moves[-3:])) == 1:
                self._cached_strategy = f'streak_{self.moves[-1]}'
                self._cache_valid_until = len(self.moves) + 2
                return self._cached_strategy

        # Default to adaptive
        self._cached_strategy = 'adaptive'
        self._cache_valid_until = len(self.moves) + 3
        return self._cached_strategy

    def get_counter_recommendation(self) -> Dict:
        """Get counter-strategy recommendation."""
        strategy = self.classify_strategy()

        # Spam counter
        if strategy.startswith('spam_'):
            move = strategy.split('_')[1]
            return {
                'move': BEATS[move],
                'confidence': 0.80,
                'reason': f'Counter {move} spam'
            }

        # Streak counter
        if strategy.startswith('streak_'):
            move = strategy.split('_')[1]
            return {
                'move': BEATS[move],
                'confidence': 0.75,
                'reason': f'Counter {move} streak'
            }

        # Transition counter
        if strategy.startswith('trans_'):
            parts = strategy.split('_')
            if len(parts) >= 3:
                from_move, to_move = parts[1], parts[2]
                if self.moves and self.moves[-1] == from_move:
                    return {
                        'move': BEATS[to_move],
                        'confidence': 0.70,
                        'reason': f'After {from_move}, expect {to_move}'
                    }

        # WSLS counter
        if strategy.startswith('wsls_'):
            if self.last_opp_result == 'win' and 'win_stay' in strategy:
                # They'll repeat their last move
                return {
                    'move': BEATS[self.moves[-1]] if self.moves else None,
                    'confidence': 0.65,
                    'reason': 'WSLS: They win-stay'
                }
            elif self.last_opp_result == 'lose' and 'lose_shift' in strategy:
                # They'll shift away from last move
                if self.moves:
                    # Can't predict exactly where, but not their last move
                    last = self.moves[-1]
                    options = [m for m in MOVES if m != last]
                    return {
                        'move': None,  # Can't be certain
                        'confidence': 0.50,
                        'reason': f'WSLS: They lose-shift from {last}'
                    }

        # Default: frequency-based
        freq = self.get_frequency()
        most_common = max(freq, key=freq.get)
        return {
            'move': BEATS[most_common],
            'confidence': 0.55,
            'reason': f'Counter most common: {most_common}'
        }


class OpponentProfiler:
    """Manages opponent profiles across matches."""

    def __init__(self):
        self.saved_profiles: Dict[str, Dict] = {}
        self.live_profile: Optional[LiveProfile] = None
        self._load_saved_profiles()

    def _load_saved_profiles(self):
        """Load saved profiles from disk."""
        DATA_DIR.mkdir(exist_ok=True)
        for filepath in DATA_DIR.glob("*.json"):
            try:
                with open(filepath) as f:
                    data = json.load(f)
                name = data.get('name', filepath.stem)
                self.saved_profiles[name.lower()] = data
            except Exception as e:
                print(f"Error loading {filepath}: {e}")

    def start_match(self, opponent_name: str) -> LiveProfile:
        """Start tracking a new match."""
        self.live_profile = LiveProfile(name=opponent_name)

        # Load any saved intel
        key = opponent_name.lower().replace(' ', '_')
        if key in self.saved_profiles:
            saved = self.saved_profiles[key]
            print(f"[Profiler] Loaded saved intel for {opponent_name}")
            print(f"  Known strategy: {saved.get('strategy_type', 'unknown')}")
            print(f"  Counter: {saved.get('counter_strategy', {}).get('play', 'unknown')}")

        return self.live_profile

    def update(self, my_move: str, opp_move: str, result: str):
        """Update live profile with round result."""
        if self.live_profile:
            self.live_profile.update(my_move, opp_move, result)

    def get_recommendation(self) -> Optional[Dict]:
        """Get move recommendation based on profile."""
        if not self.live_profile:
            return None

        return self.live_profile.get_counter_recommendation()

    def get_strategy(self) -> str:
        """Get current opponent strategy classification."""
        if not self.live_profile:
            return 'unknown'
        return self.live_profile.classify_strategy()

    def save_match_results(self):
        """Save match results to disk for future reference."""
        if not self.live_profile or len(self.live_profile.moves) < 20:
            return

        key = self.live_profile.name.lower().replace(' ', '_')
        filepath = DATA_DIR / f"{key}.json"

        # Merge with existing data
        existing = self.saved_profiles.get(key, {
            'name': self.live_profile.name,
            'total_moves': {'rock': 0, 'paper': 0, 'scissors': 0},
            'match_count': 0
        })

        # Update totals
        freq = self.live_profile.get_frequency()
        for move in MOVES:
            existing['total_moves'][move] = existing.get('total_moves', {}).get(move, 0) + \
                                            self.live_profile.moves.count(move)

        existing['match_count'] = existing.get('match_count', 0) + 1
        existing['strategy_type'] = self.live_profile.classify_strategy()
        existing['counter_strategy'] = self.live_profile.get_counter_recommendation()

        # Save
        with open(filepath, 'w') as f:
            json.dump(existing, f, indent=2)

        self.saved_profiles[key] = existing
        print(f"[Profiler] Saved profile for {self.live_profile.name}")


# ============================================================================
# Integration with RPSBrain
# ============================================================================

# Global profiler instance
_profiler: Optional[OpponentProfiler] = None


def get_profiler() -> OpponentProfiler:
    """Get or create global profiler instance."""
    global _profiler
    if _profiler is None:
        _profiler = OpponentProfiler()
    return _profiler


def start_match(opponent_name: str) -> LiveProfile:
    """Start profiling a new match."""
    return get_profiler().start_match(opponent_name)


def update_profile(my_move: str, opp_move: str, result: str):
    """Update profile with round result."""
    get_profiler().update(my_move, opp_move, result)


def get_counter_move() -> Optional[Dict]:
    """Get recommended counter move."""
    return get_profiler().get_recommendation()


def end_match():
    """End match and save results."""
    get_profiler().save_match_results()


# ============================================================================
# Testing
# ============================================================================

def test_profiler():
    """Test the profiler with simulated matches."""
    print("Testing Opponent Profiler")
    print("=" * 50)

    # Test 1: Spam detection
    print("\nTest 1: Rock spammer")
    profile = LiveProfile("RockBot")
    for _ in range(20):
        if _ % 5 == 0:
            opp = 'paper'  # Occasional variation
        else:
            opp = 'rock'
        profile.update('paper', opp, 'you' if opp == 'rock' else 'opponent')

    print(f"  Strategy: {profile.classify_strategy()}")
    print(f"  Counter: {profile.get_counter_recommendation()}")

    # Test 2: Transition pattern
    print("\nTest 2: Transition pattern (rock->paper)")
    profile = LiveProfile("TransBot")
    moves = ['rock', 'paper', 'scissors', 'rock', 'paper', 'scissors',
             'rock', 'paper', 'rock', 'paper', 'rock', 'paper',
             'rock', 'paper', 'rock', 'paper', 'rock', 'paper']
    for i, opp in enumerate(moves):
        profile.update('rock', opp, 'tie')

    print(f"  Strategy: {profile.classify_strategy()}")
    print(f"  Counter: {profile.get_counter_recommendation()}")

    # Test 3: WSLS
    print("\nTest 3: Win-Stay-Lose-Shift")
    profile = LiveProfile("WSLSBot")
    opp = 'rock'
    for i in range(20):
        my = ['rock', 'paper', 'scissors'][i % 3]
        result = 'you' if BEATS[opp] == my else 'opponent' if BEATS[my] == opp else 'tie'

        profile.update(my, opp, result)

        # WSLS logic
        if result == 'opponent':  # opponent won
            pass  # stay
        else:
            opp = ['rock', 'paper', 'scissors'][(MOVES.index(opp) + 1) % 3]

    print(f"  Strategy: {profile.classify_strategy()}")
    print(f"  Counter: {profile.get_counter_recommendation()}")

    print("\n" + "=" * 50)
    print("Tests complete!")


if __name__ == "__main__":
    test_profiler()
