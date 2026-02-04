#!/usr/bin/env python3
"""
Play an active match using v8 brain.
"""

import os
import sys
import time
import requests
import json

from rps_brain_v8 import RPSBrainV8

API_BASE = "https://www.botgames.ai/api/v1"
API_KEY = os.environ.get("BOTGAMES_API_KEY", "")

MOVES = ['rock', 'paper', 'scissors']


def get_winner(my_move: str, opp_move: str) -> str:
    if my_move == opp_move:
        return "tie"
    elif {'rock': 'scissors', 'paper': 'rock', 'scissors': 'paper'}[my_move] == opp_move:
        return "you"
    return "opponent"


def play_match(match_id: str):
    """Play a match using v8 brain."""
    if not API_KEY:
        print("Error: BOTGAMES_API_KEY not set")
        sys.exit(1)

    session = requests.Session()
    session.headers.update({
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    })

    brain = RPSBrainV8()
    print(f"Playing match {match_id} with RPSBrain v{brain.VERSION}")

    while True:
        # Get match state
        resp = session.get(f"{API_BASE}/matches/{match_id}")
        if resp.status_code != 200:
            print(f"Error getting match state: {resp.text}")
            break

        state = resp.json()

        # Check if match is finished
        if state.get("status") == "completed" or state.get("winner"):
            print(f"\n{'='*50}")
            print(f"MATCH COMPLETE")
            print(f"Final Score: {state.get('your_score')} - {state.get('opponent_score')}")
            winner = state.get("winner")
            if winner == "you":
                print("VICTORY!")
            elif winner == "opponent":
                print(f"Defeated by {state.get('opponent', {}).get('name')}")
            else:
                print("Draw")
            print(f"{'='*50}")
            break

        # Get opponent info
        opponent = state.get("opponent", {})
        opp_name = opponent.get("name", "Unknown")
        opp_elo = opponent.get("elo", 0)

        current_round = state.get("current_round", 0)
        your_score = state.get("your_score", 0)
        opp_score = state.get("opponent_score", 0)

        # Sync brain with history
        history = state.get("history", [])
        for i, h in enumerate(history):
            if i >= len(brain.opp_history):
                our_move = h.get("your_move", "").lower()
                opp_move = h.get("opponent_move", "").lower()
                winner = h.get("winner", "tie")
                if our_move and opp_move:
                    brain.update(our_move, opp_move, winner)

        # Check if we need to submit a move
        if state.get("awaiting_move") and not state.get("your_move_submitted"):
            # Get move from brain
            move, reasoning = brain.choose_move()

            print(f"Round {current_round}: vs {opp_name} ({opp_elo}) | Score: {your_score}-{opp_score} | Playing: {move}")

            # Submit move
            resp = session.post(
                f"{API_BASE}/matches/{match_id}/move",
                json={"move": move, "reasoning": reasoning[:200]}
            )

            if resp.status_code != 200:
                print(f"Error submitting move: {resp.text}")
                # Check if match is already completed
                if "already" in resp.text.lower() or "completed" in resp.text.lower():
                    break
            else:
                result = resp.json()
                if result.get("round_result"):
                    rr = result["round_result"]
                    print(f"  -> Opponent: {rr.get('opponent_move')} | Result: {rr.get('winner')}")

            # Brief pause
            time.sleep(0.3)
        else:
            # Waiting for opponent
            print(f"Round {current_round}: Waiting for opponent... (Score: {your_score}-{opp_score})")
            time.sleep(1.0)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        match_id = sys.argv[1]
    else:
        # Check for active match
        print("Usage: python play_active_match.py <match_id>")
        print("\nChecking for active matches...")

        session = requests.Session()
        session.headers.update({
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        })

        # Queue for new match
        resp = session.post(f"{API_BASE}/matches/queue", json={"mode": "realtime"})
        if resp.status_code == 200:
            data = resp.json()
            if data.get("matched") or data.get("match_id"):
                match_id = data.get("match_id")
                print(f"Match found: {match_id}")
            else:
                print("Queued for match, waiting for opponent...")
                # Poll for match
                for _ in range(60):  # Wait up to 60 seconds
                    time.sleep(2)
                    resp = session.post(f"{API_BASE}/matches/queue", json={"mode": "realtime"})
                    if resp.status_code == 200:
                        data = resp.json()
                        if data.get("matched") or data.get("match_id"):
                            match_id = data.get("match_id")
                            print(f"Match found: {match_id}")
                            break
                else:
                    print("No match found after 2 minutes")
                    sys.exit(0)
        else:
            print(f"Error: {resp.text}")
            sys.exit(1)

    play_match(match_id)
