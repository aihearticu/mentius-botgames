#!/usr/bin/env python3
"""
BotGames Match Player - Mentius Championship Edition

Plays RPS matches on BotGames.ai using the v8 brain.
Supports real-time matchmaking and opponent profiling.
"""

import os
import sys
import time
import json
import argparse
import requests
from typing import Dict, Optional, List, Tuple
from datetime import datetime

# Local imports
from rps_brain_v9 import RPSBrainV9

# Opponent profiler disabled for speed
PROFILER_AVAILABLE = False
# try:
#     from opponent_profiler import start_match, update_profile, get_counter_move, end_match
#     PROFILER_AVAILABLE = True
# except ImportError:
#     PROFILER_AVAILABLE = False
#     print("Warning: Opponent profiler not available", flush=True)

# Configuration
API_BASE = "https://www.botgames.ai/api/v1"
API_KEY = os.environ.get("BOTGAMES_API_KEY", "")

MOVES = ['rock', 'paper', 'scissors']


class BotGamesClient:
    """Client for BotGames.ai API."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or API_KEY
        if not self.api_key:
            raise ValueError("BOTGAMES_API_KEY environment variable not set")

        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        })

    def get_agent_info(self) -> Dict:
        """Get current agent info."""
        resp = self.session.get(f"{API_BASE}/agents/me")
        resp.raise_for_status()
        return resp.json()

    def queue_for_match(self, mode: str = "realtime") -> Dict:
        """Join matchmaking queue."""
        resp = self.session.post(f"{API_BASE}/matches/queue", json={"mode": mode})
        resp.raise_for_status()
        return resp.json()

    def get_match_state(self, match_id: str) -> Dict:
        """Get current match state."""
        resp = self.session.get(f"{API_BASE}/matches/{match_id}")
        resp.raise_for_status()
        return resp.json()

    def submit_move(self, match_id: str, move: str, reasoning: str = "") -> Dict:
        """Submit move for current round with retry on 400 errors."""
        for attempt in range(3):
            resp = self.session.post(
                f"{API_BASE}/matches/{match_id}/move",
                json={"move": move, "reasoning": reasoning}
            )
            if resp.status_code == 400:
                # May be race condition or not our turn - wait and retry
                time.sleep(0.3)
                continue
            resp.raise_for_status()
            return resp.json()
        # Final attempt
        resp = self.session.post(
            f"{API_BASE}/matches/{match_id}/move",
            json={"move": move, "reasoning": reasoning}
        )
        if resp.status_code == 400:
            # Return empty to signal skip this round
            return {"skipped": True, "error": resp.text}
        resp.raise_for_status()
        return resp.json()

    def get_leaderboard(self, limit: int = 20) -> List[Dict]:
        """Get current leaderboard."""
        resp = self.session.get(f"{API_BASE}/leaderboard", params={"limit": limit})
        resp.raise_for_status()
        data = resp.json()
        # Handle nested response format
        if isinstance(data, dict) and "leaderboard" in data:
            return data["leaderboard"]
        return data


class MatchPlayer:
    """Plays matches using the v8 brain."""

    def __init__(self, client: BotGamesClient, brain: RPSBrainV9 = None):
        self.client = client
        self.brain = brain or RPSBrainV9()
        self.current_match_id: Optional[str] = None
        self.opponent_name: Optional[str] = None
        self.match_log: List[Dict] = []

    def play_match(self, match_id: str = None) -> Dict:
        """Play a full match."""
        # Get or wait for match
        if match_id:
            self.current_match_id = match_id
        else:
            print("Queuing for match...", flush=True)
            self.client.queue_for_match()
            # Poll for match
            for _ in range(120):  # 2 minute timeout
                status = self.client.session.get(f"{API_BASE}/matches/queue/status").json()
                if status.get("matched"):
                    self.current_match_id = status.get("match_id")
                    break
                time.sleep(1)
            if not self.current_match_id:
                raise TimeoutError("No match found within 2 minutes")
            print(f"Match found: {self.current_match_id}", flush=True)

        # Reset brain for new match
        if hasattr(self.brain, 'my_history'):
            # V9 brain
            self.brain.my_history = []
            self.brain.opp_history = []
            self.brain.results = []
            self.brain.my_score = 0
            self.brain.opp_score = 0
            self.brain.consecutive_losses = 0
            self.brain.last_moves = []
            self.brain.move_losses = {'rock': 0, 'paper': 0, 'scissors': 0}
            self.brain.move_counts = {'rock': 0, 'paper': 0, 'scissors': 0}
            # Reset predictors
            for p in self.brain.predictors:
                p.results = []
                p.last_prediction = None
        elif hasattr(self.brain, 'reset'):
            self.brain.reset()
        self.match_log = []

        # Get initial state
        state = self.client.get_match_state(self.current_match_id)
        self.opponent_name = state.get("opponent", {}).get("name", "Unknown")

        print(f"\n{'='*50}", flush=True)
        print(f"Match vs {self.opponent_name}", flush=True)
        print(f"{'='*50}", flush=True)

        # Start opponent profiling if available
        if PROFILER_AVAILABLE:
            start_match(self.opponent_name)

        # Play until match ends (check both 'finished' and 'status')
        max_rounds = 120  # Safety limit
        rounds_played = 0
        while state.get("status") != "completed" and not state.get("finished") and rounds_played < max_rounds:
            if state.get("awaiting_move"):
                # Sync brain with match history
                self._sync_brain_with_history(state.get("history", []))

                # Get move from brain
                move, reasoning = self.brain.choose_move()

                # Check profiler recommendation
                if PROFILER_AVAILABLE:
                    rec = get_counter_move()
                    if rec and rec.get("confidence", 0) > 0.75:
                        prof_move = rec.get("move")
                        if prof_move and prof_move != move:
                            reasoning += f" [Profiler suggests {prof_move}: {rec.get('reason')}]"
                            # Consider overriding in high-confidence cases
                            if rec.get("confidence", 0) > 0.80:
                                move = prof_move
                                reasoning = f"Profiler override: {rec.get('reason')}"

                # Submit move
                round_num = state.get("current_round", 0)
                print(f"Round {round_num}: Playing {move} ({reasoning[:50]}...)", flush=True)

                result = self.client.submit_move(
                    self.current_match_id,
                    move,
                    reasoning
                )

                # Log round
                self.match_log.append({
                    "round": round_num,
                    "our_move": move,
                    "reasoning": reasoning,
                    "result": result
                })

                # Wait a moment before polling
                time.sleep(0.15)
                rounds_played += 1
            else:
                # Wait for opponent
                time.sleep(0.25)

            # Get updated state
            state = self.client.get_match_state(self.current_match_id)

        # Match finished
        winner = state.get("winner", "")
        our_score = state.get("your_score", 0)
        opp_score = state.get("opponent_score", 0)
        
        # Determine result (check winner field first, then score)
        if winner == "you":
            result = "win"
        elif winner == "opponent":
            result = "lose"
        elif our_score > opp_score:
            result = "win"
        elif opp_score > our_score:
            result = "lose"
        else:
            result = "tie"
            
        final_score = {
            "our_score": our_score,
            "opp_score": opp_score,
            "result": result
        }

        print(f"\n{'='*50}", flush=True)
        print(f"Match Complete: {final_score['our_score']} - {final_score['opp_score']}", flush=True)
        print(f"Result: {final_score['result'].upper()}", flush=True)
        print(f"{'='*50}\n", flush=True)

        # End profiling
        if PROFILER_AVAILABLE:
            end_match()

        return final_score

    def _sync_brain_with_history(self, history: List[Dict]):
        """Sync brain state with match history."""
        # Clear and rebuild from history
        brain_rounds = len(self.brain.opp_history)

        for i, h in enumerate(history):
            if i >= brain_rounds:
                # New round to add
                opp_move = h.get("opponent_move", "").lower()
                our_move = h.get("your_move", "").lower()
                winner = h.get("winner", "tie")

                if opp_move and our_move:
                    # Update brain with winner
                    self.brain.update(our_move, opp_move, winner)

                    # Update profiler
                    if PROFILER_AVAILABLE:
                        update_profile(our_move, opp_move, winner)


def run_local_test(brain: RPSBrainV9, opponent_type: str = "random", rounds: int = 99):
    """Run a local test match against a simulated opponent."""
    import random

    print(f"\nLocal Test: v8 Brain vs {opponent_type} ({rounds} rounds)", flush=True)
    print("=" * 50, flush=True)

    brain = RPSBrainV9()  # Fresh brain for test
    wins, losses, ties = 0, 0, 0

    # Opponent strategies
    opp_last_move = random.choice(MOVES)

    for r in range(rounds):
        # Get our move
        our_move, reasoning = brain.choose_move()

        # Generate opponent move based on type
        if opponent_type == "random":
            opp_move = random.choice(MOVES)
        elif opponent_type == "rock_spam":
            opp_move = "rock" if random.random() < 0.8 else random.choice(MOVES)
        elif opponent_type == "paper_spam":
            opp_move = "paper" if random.random() < 0.8 else random.choice(MOVES)
        elif opponent_type == "scissors_spam":
            opp_move = "scissors" if random.random() < 0.8 else random.choice(MOVES)
        elif opponent_type == "cycle":
            # Rock -> Paper -> Scissors cycle
            opp_move = MOVES[(MOVES.index(opp_last_move) + 1) % 3]
        elif opponent_type == "counter":
            # Always counters our last move
            if brain.my_history:
                last = brain.my_history[-1]
                opp_move = {'rock': 'paper', 'paper': 'scissors', 'scissors': 'rock'}[last]
            else:
                opp_move = random.choice(MOVES)
        elif opponent_type == "wsls":
            # Win-Stay-Lose-Shift
            if r > 0:
                last_opp = brain.opp_history[-1] if brain.opp_history else "rock"
                last_our = brain.my_history[-1] if brain.my_history else "rock"
                # Did opponent win last round?
                opp_won = ({'rock': 'scissors', 'paper': 'rock', 'scissors': 'paper'}[last_opp] == last_our)
                if opp_won:
                    opp_move = last_opp  # Stay
                else:
                    opp_move = MOVES[(MOVES.index(last_opp) + 1) % 3]  # Shift
            else:
                opp_move = random.choice(MOVES)
        else:
            opp_move = random.choice(MOVES)

        opp_last_move = opp_move

        # Determine winner
        if our_move == opp_move:
            winner = "tie"
            ties += 1
        elif {'rock': 'scissors', 'paper': 'rock', 'scissors': 'paper'}[our_move] == opp_move:
            winner = "you"
            wins += 1
        else:
            winner = "opponent"
            losses += 1

        # Update brain
        brain.update(our_move, opp_move, winner)

        # Progress output
        if (r + 1) % 20 == 0:
            wr = wins / (wins + losses) if (wins + losses) > 0 else 0
            print(f"Round {r+1}: {wins}W-{losses}L-{ties}T (WR: {wr:.1%})", flush=True)

    # Final results
    total = wins + losses
    win_rate = wins / total if total > 0 else 0

    print("\n" + "=" * 50, flush=True)
    print(f"Final: {wins}W - {losses}L - {ties}T", flush=True)
    print(f"Win Rate: {win_rate:.1%}", flush=True)
    print("=" * 50, flush=True)

    return {
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "win_rate": win_rate,
        "opponent": opponent_type
    }


def main():
    parser = argparse.ArgumentParser(description="BotGames Match Player")
    parser.add_argument("--test", action="store_true", help="Run local test")
    parser.add_argument("--opponent", default="random",
                       choices=["random", "rock_spam", "paper_spam", "scissors_spam",
                               "cycle", "counter", "wsls"],
                       help="Test opponent type")
    parser.add_argument("--rounds", type=int, default=99, help="Test rounds")
    parser.add_argument("--match", help="Specific match ID to join")
    parser.add_argument("--queue", action="store_true", help="Queue for matchmaking")
    parser.add_argument("--grind", type=int, default=0, help="Play N matches in a row")
    parser.add_argument("--leaderboard", action="store_true", help="Show leaderboard")

    args = parser.parse_args()

    # Initialize brain
    brain = RPSBrainV9()
    print(f"Initialized RPSBrain v{brain.VERSION}", flush=True)

    # Local test mode
    if args.test:
        run_local_test(brain, args.opponent, args.rounds)
        return

    # API modes require key
    if not API_KEY:
        print("Error: BOTGAMES_API_KEY not set", flush=True)
        print("For local testing, use: python play_match.py --test", flush=True)
        sys.exit(1)

    client = BotGamesClient()
    player = MatchPlayer(client, brain)

    # Leaderboard
    if args.leaderboard:
        lb = client.get_leaderboard()
        print("\nBotGames Leaderboard", flush=True)
        print("=" * 40, flush=True)
        for i, entry in enumerate(lb, 1):
            print(f"{i}. {entry.get('name')}: {entry.get('elo')} ELO ({entry.get('wins')}W-{entry.get('losses')}L)", flush=True)
        return

    # Grind mode
    if args.grind > 0:
        results = []
        for i in range(args.grind):
            print(f"\n--- Match {i+1}/{args.grind} ---", flush=True)
            result = player.play_match()
            results.append(result)

            # Brief pause between matches
            if i < args.grind - 1:
                time.sleep(2)

        # Summary
        wins = sum(1 for r in results if r["result"] == "win")
        print(f"\nGrind Complete: {wins}/{len(results)} wins ({wins/len(results):.1%})", flush=True)
        return

    # Single match
    if args.match:
        player.play_match(args.match)
    elif args.queue:
        player.play_match()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
