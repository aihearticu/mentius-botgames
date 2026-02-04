#!/usr/bin/env python3
"""
Championship Test Suite - Mentius RPS Bot

Comprehensive tests for all championship components:
- RPSBrainV8 predictor tests
- Late-game adaptation tests
- Opponent profiler tests
- TAO model tests (if PyTorch available)
- Integration tests
"""

import sys
import random
import time
from collections import Counter
from typing import List, Tuple, Dict

# Local imports
from rps_brain_v8 import RPSBrainV8, MOVES, BEATS

PROFILER_AVAILABLE = False
TAO_AVAILABLE = False

try:
    from opponent_profiler import LiveProfile, OpponentProfiler, get_profiler
    PROFILER_AVAILABLE = True
except ImportError:
    pass

try:
    import torch
    from tao_model import TAOModel, TAOConfig
    TAO_AVAILABLE = True
except ImportError:
    pass


class TestResults:
    """Track test results."""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.tests = []

    def record(self, name: str, passed: bool, details: str = ""):
        self.tests.append({"name": name, "passed": passed, "details": details})
        if passed:
            self.passed += 1
            print(f"  [PASS] {name}")
        else:
            self.failed += 1
            print(f"  [FAIL] {name}: {details}")

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"Test Results: {self.passed}/{total} passed ({100*self.passed/total:.0f}%)")
        if self.failed > 0:
            print(f"Failed tests:")
            for t in self.tests:
                if not t["passed"]:
                    print(f"  - {t['name']}: {t['details']}")
        print(f"{'='*60}")
        return self.failed == 0


def get_winner(my_move: str, opp_move: str) -> str:
    """Determine winner from moves."""
    if my_move == opp_move:
        return "tie"
    elif BEATS[opp_move] == my_move:
        return "you"
    else:
        return "opponent"


def simulate_match(brain: RPSBrainV8, opponent_strategy: callable, rounds: int = 99) -> Dict:
    """Simulate a match and return results. Creates a fresh brain instance."""
    brain = RPSBrainV8()  # Always create fresh brain for match
    wins, losses, ties = 0, 0, 0
    phase_stats = {"early": [0, 0], "mid": [0, 0], "late": [0, 0], "end": [0, 0]}

    opp_state = {"last_move": random.choice(MOVES), "won_last": False}

    for r in range(rounds):
        our_move, _ = brain.choose_move()
        opp_move = opponent_strategy(brain, opp_state, r)

        # Update opponent state
        our_won = BEATS[opp_move] == our_move
        opp_state["won_last"] = not our_won and our_move != opp_move
        opp_state["last_move"] = opp_move

        # Determine result
        winner = get_winner(our_move, opp_move)
        if winner == "tie":
            ties += 1
        elif winner == "you":
            wins += 1
        else:
            losses += 1

        # Track phase stats
        if r < 20:
            phase = "early"
        elif r < 50:
            phase = "mid"
        elif r < 80:
            phase = "late"
        else:
            phase = "end"

        if our_move != opp_move:
            if winner == "you":
                phase_stats[phase][0] += 1
            else:
                phase_stats[phase][1] += 1

        brain.update(our_move, opp_move, winner)

    return {
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "win_rate": wins / (wins + losses) if (wins + losses) > 0 else 0,
        "phase_stats": phase_stats
    }


# ============================================================================
# OPPONENT STRATEGIES
# ============================================================================

def rock_spam(brain, state, round_num):
    return "rock" if random.random() < 0.85 else random.choice(MOVES)

def paper_spam(brain, state, round_num):
    return "paper" if random.random() < 0.85 else random.choice(MOVES)

def scissors_spam(brain, state, round_num):
    return "scissors" if random.random() < 0.85 else random.choice(MOVES)

def pure_random(brain, state, round_num):
    return random.choice(MOVES)

def cycle_forward(brain, state, round_num):
    """Rock -> Paper -> Scissors cycle."""
    return MOVES[(MOVES.index(state["last_move"]) + 1) % 3]

def cycle_backward(brain, state, round_num):
    """Scissors -> Paper -> Rock cycle."""
    return MOVES[(MOVES.index(state["last_move"]) - 1) % 3]

def counter_last(brain, state, round_num):
    """Counter our last move."""
    if brain.my_history:
        return BEATS[brain.my_history[-1]]
    return random.choice(MOVES)

def meta_counter(brain, state, round_num):
    """Counter what would beat our last move (meta-level)."""
    if brain.my_history:
        what_beats_us = BEATS[brain.my_history[-1]]
        return BEATS[what_beats_us]
    return random.choice(MOVES)

def wsls_opponent(brain, state, round_num):
    """Win-Stay-Lose-Shift."""
    if round_num == 0:
        return random.choice(MOVES)

    if state["won_last"]:
        return state["last_move"]  # Stay
    else:
        return MOVES[(MOVES.index(state["last_move"]) + 1) % 3]  # Shift

def frequency_counter(brain, state, round_num):
    """Counter our most frequent move."""
    if len(brain.my_history) < 5:
        return random.choice(MOVES)

    freq = Counter(brain.my_history)
    most_common = freq.most_common(1)[0][0]
    return BEATS[most_common]

def adaptive_opponent(brain, state, round_num):
    """Switches strategy mid-match."""
    if round_num < 30:
        return rock_spam(brain, state, round_num)
    elif round_num < 60:
        return counter_last(brain, state, round_num)
    else:
        return frequency_counter(brain, state, round_num)

def late_game_crusher(brain, state, round_num):
    """Random early, aggressive late."""
    if round_num < 40:
        return pure_random(brain, state, round_num)
    else:
        # Aggressively counter patterns
        if len(brain.my_history) >= 3:
            last3 = brain.my_history[-3:]
            if len(set(last3)) == 1:
                # We're repeating - counter it
                return BEATS[last3[0]]
        return frequency_counter(brain, state, round_num)


# ============================================================================
# TEST FUNCTIONS
# ============================================================================

def test_brain_initialization(results: TestResults):
    """Test brain initialization."""
    print("\n--- Brain Initialization Tests ---")

    brain = RPSBrainV8()
    results.record("Brain instantiated", brain is not None)
    results.record("Version is 8.0", brain.VERSION == "8.0")
    results.record("Has predictors (>15)", len(brain.predictors) > 15)
    results.record("Has history tracking", hasattr(brain, 'my_history') and hasattr(brain, 'opp_history'))
    results.record("Has phase detection", hasattr(brain, '_get_game_phase') or hasattr(brain, 'get_game_phase'))


def test_spam_detection(results: TestResults):
    """Test spam move detection."""
    print("\n--- Spam Detection Tests ---")

    brain = RPSBrainV8()

    # Test rock spam detection
    for _ in range(20):
        brain.update("paper", "rock", "you")  # paper beats rock

    move, reason = brain.choose_move()
    results.record("Detects rock spam", "spam" in reason.lower() or move == "paper",
                   f"Got move={move}, reason={reason[:50]}")

    # Test against rock spam opponent
    result = simulate_match(brain, rock_spam, 99)
    results.record("Beats rock spam (>60% WR)", result["win_rate"] > 0.60,
                   f"WR={result['win_rate']:.1%}")


def test_late_game_adaptation(results: TestResults):
    """Test late-game adaptation."""
    print("\n--- Late-Game Adaptation Tests ---")

    brain = RPSBrainV8()

    # Simulate being dominated in late game
    for i in range(60):
        brain.update("rock", "paper", "opponent")  # Losing every round

    # Check that brain adapts - it may or may not change from rock depending on predictor confidence
    move, reason = brain.choose_move()
    # Brain should either change move OR inject noise in reasoning
    adapted = move != "rock" or "noise" in reason.lower() or "random" in reason.lower() or "emergency" in reason.lower()
    results.record("Adapts when losing streak", adapted,
                   f"Got move={move}, may need noise injection")

    # Test against late game crusher
    result = simulate_match(brain, late_game_crusher, 99)

    # Check late game performance specifically
    late_wins = result["phase_stats"]["late"][0]
    late_losses = result["phase_stats"]["late"][1]
    late_total = late_wins + late_losses
    late_wr = late_wins / late_total if late_total > 0 else 0

    results.record("Late-game WR > 40%", late_wr > 0.40,
                   f"Late game WR={late_wr:.1%} ({late_wins}/{late_total})")


def test_adaptive_opponents(results: TestResults):
    """Test against various adaptive opponents."""
    print("\n--- Adaptive Opponent Tests ---")

    opponents = [
        ("WSLS", wsls_opponent, 0.52),
        ("Counter-Last", counter_last, 0.42),
        ("Meta-Counter", meta_counter, 0.40),  # Meta-counter is very strong
        ("Frequency Counter", frequency_counter, 0.42),
        ("Adaptive (phase-switch)", adaptive_opponent, 0.42),
    ]

    for name, strategy, min_wr in opponents:
        brain = RPSBrainV8()
        result = simulate_match(brain, strategy, 99)
        results.record(f"vs {name} (>{min_wr:.0%} WR)", result["win_rate"] >= min_wr,
                       f"WR={result['win_rate']:.1%}")


def test_transition_detection(results: TestResults):
    """Test transition pattern detection."""
    print("\n--- Transition Detection Tests ---")

    brain = RPSBrainV8()

    # Feed a strong transition pattern: rock always followed by paper
    for _ in range(15):
        brain.update("rock", "rock", "tie")
        brain.update("scissors", "paper", "you")  # scissors beats paper

    # After rock, should predict paper and play scissors
    brain.update("rock", "rock", "tie")
    move, reason = brain.choose_move()

    results.record("Detects transition pattern", move == "scissors",
                   f"Expected scissors, got {move}")


def test_multiple_matches(results: TestResults):
    """Test consistency across multiple matches."""
    print("\n--- Consistency Tests (Multiple Matches) ---")

    opponents = [
        ("Random", pure_random),
        ("Cycle", cycle_forward),
        ("WSLS", wsls_opponent),
    ]

    for name, strategy in opponents:
        win_rates = []
        for _ in range(5):
            brain = RPSBrainV8()
            result = simulate_match(brain, strategy, 99)
            win_rates.append(result["win_rate"])

        avg_wr = sum(win_rates) / len(win_rates)
        std = (sum((wr - avg_wr)**2 for wr in win_rates) / len(win_rates)) ** 0.5

        results.record(f"vs {name} consistent (std < 0.15)", std < 0.15,
                       f"Avg WR={avg_wr:.1%}, Std={std:.2f}")


def test_profiler(results: TestResults):
    """Test opponent profiler."""
    print("\n--- Opponent Profiler Tests ---")

    if not PROFILER_AVAILABLE:
        results.record("Profiler available", False, "Module not imported")
        return

    # Test LiveProfile
    profile = LiveProfile("TestBot")
    results.record("LiveProfile created", profile is not None)

    # Feed spam pattern
    for _ in range(20):
        profile.update("paper", "rock", "you")

    strategy = profile.classify_strategy()
    results.record("Detects spam strategy", "spam_rock" in strategy,
                   f"Got strategy: {strategy}")

    rec = profile.get_counter_recommendation()
    results.record("Recommends paper vs rock spam", rec.get("move") == "paper",
                   f"Got recommendation: {rec}")

    # Test transition detection - rock -> paper -> scissors pattern
    profile2 = LiveProfile("TransBot")
    for _ in range(8):
        profile2.update("rock", "rock", "tie")
        profile2.update("paper", "paper", "tie")
        profile2.update("scissors", "scissors", "tie")

    strategy2 = profile2.classify_strategy()
    # With rock->paper->scissors cycle, check for transition or cycle detection
    detected = "trans" in strategy2.lower() or "adaptive" in strategy2.lower()
    results.record("Detects pattern (trans/adaptive)", detected,
                   f"Got strategy: {strategy2}")


def test_tao_model(results: TestResults):
    """Test TAO model."""
    print("\n--- TAO Model Tests ---")

    if not TAO_AVAILABLE:
        results.record("TAO model available", False, "PyTorch not installed")
        return

    config = TAOConfig()
    model = TAOModel(config)
    results.record("TAO model created", model is not None)

    # Test prediction
    my_moves = ["rock", "paper", "scissors"] * 5
    opp_moves = ["scissors", "rock", "paper"] * 5

    try:
        probs = model.predict(my_moves, opp_moves)
        results.record("TAO prediction works", probs is not None and len(probs) == 3,
                       f"Got probs shape: {probs.shape if hasattr(probs, 'shape') else len(probs)}")

        move, conf = model.get_best_counter(my_moves, opp_moves)
        results.record("TAO best counter works", move in MOVES,
                       f"Got move={move}, conf={conf:.2f}")
    except Exception as e:
        results.record("TAO inference", False, str(e))


def test_integration(results: TestResults):
    """Test full integration."""
    print("\n--- Integration Tests ---")

    # Test brain + profiler integration
    brain = RPSBrainV8()
    profiler = None

    if PROFILER_AVAILABLE:
        profiler = OpponentProfiler()
        profiler.start_match("IntegrationTest")

    wins, losses = 0, 0

    for r in range(50):
        move, reason = brain.choose_move()

        # Simulate opponent (rock spam)
        opp_move = "rock" if random.random() < 0.8 else random.choice(MOVES)

        # Determine result
        if BEATS[opp_move] == move:
            result = "you"
            wins += 1
        elif BEATS[move] == opp_move:
            result = "opponent"
            losses += 1
        else:
            result = "tie"

        brain.update(move, opp_move, result)
        if profiler:
            profiler.update(move, opp_move, result)

    wr = wins / (wins + losses) if (wins + losses) > 0 else 0
    results.record("Integration: brain + profiler", wr > 0.55,
                   f"WR={wr:.1%}")


def test_edge_cases(results: TestResults):
    """Test edge cases."""
    print("\n--- Edge Case Tests ---")

    brain = RPSBrainV8()

    # Empty history
    move, _ = brain.choose_move()
    results.record("Handles empty history", move in MOVES)

    # Single round
    brain.update("rock", "paper", "opponent")
    move, _ = brain.choose_move()
    results.record("Handles single round", move in MOVES)

    # Very long match
    brain = RPSBrainV8()
    for _ in range(200):
        m, _ = brain.choose_move()
        opp = random.choice(MOVES)
        winner = get_winner(m, opp)
        brain.update(m, opp, winner)

    move, _ = brain.choose_move()
    results.record("Handles 200+ rounds", move in MOVES)


def run_all_tests():
    """Run complete test suite."""
    print("=" * 60)
    print("MENTIUS CHAMPIONSHIP TEST SUITE")
    print("=" * 60)

    results = TestResults()

    # Run all test categories
    test_brain_initialization(results)
    test_spam_detection(results)
    test_late_game_adaptation(results)
    test_adaptive_opponents(results)
    test_transition_detection(results)
    test_multiple_matches(results)
    test_profiler(results)
    test_tao_model(results)
    test_integration(results)
    test_edge_cases(results)

    return results.summary()


def run_benchmark():
    """Run performance benchmark."""
    print("\n" + "=" * 60)
    print("PERFORMANCE BENCHMARK")
    print("=" * 60)

    brain = RPSBrainV8()

    # Warm up
    for _ in range(10):
        brain.choose_move()
        m, o = random.choice(MOVES), random.choice(MOVES)
        brain.update(m, o, get_winner(m, o))

    brain = RPSBrainV8()  # Fresh brain for benchmark

    # Benchmark
    iterations = 1000
    start = time.time()

    for i in range(iterations):
        move, _ = brain.choose_move()
        opp = random.choice(MOVES)
        brain.update(move, opp, get_winner(move, opp))

    elapsed = time.time() - start
    per_move = elapsed / iterations * 1000

    print(f"Benchmark: {iterations} moves in {elapsed:.2f}s")
    print(f"Average: {per_move:.2f}ms per move")
    print(f"Throughput: {iterations/elapsed:.0f} moves/sec")

    return per_move < 50  # Should be under 50ms per move


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Championship Test Suite")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmark")
    parser.add_argument("--quick", action="store_true", help="Quick test only")
    args = parser.parse_args()

    if args.benchmark:
        success = run_benchmark()
        sys.exit(0 if success else 1)

    success = run_all_tests()

    if not args.quick:
        run_benchmark()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
