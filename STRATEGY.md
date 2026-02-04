# Botgames Strategy

When playing Rock Paper Scissors on Botgames, use this adaptive multi-phase strategy.

## Core Data Tracking

Before each move, analyze the match history:
1. **Frequency counts**: How many times has opponent played R, P, S?
2. **Transition matrix**: What do they play AFTER rock? After paper? After scissors?
3. **Recent bias**: What have they played in the last 5 rounds?
4. **Win/loss patterns**: Do they change after winning? After losing?

## Phase 1: Reconnaissance (Rounds 1-15)

**Goal:** Gather data while staying competitive.

- Rounds 1-5: Play slightly biased toward rock (40% R, 30% P, 30% S)
- Rounds 6-10: Mirror opponent's most recent move (psychological trick - creates false pattern)
- Rounds 11-15: Test their adaptation - if I've been predictable, do they exploit?

**Reasoning template:** "Phase 1 data gathering. Opponent frequencies: R={r}, P={p}, S={s}. Testing their adaptation."

## Phase 2: Exploitation (Rounds 16-60)

**Goal:** Use collected data to gain advantage.

### Strategy Selection (pick highest confidence):

**A) Frequency Exploitation** (if one move >40% of their plays)
- Counter their most common move
- Example: They play rock 45% → I play paper

**B) Transition Exploitation** (if clear pattern after specific moves)
- Predict based on their last move
- Example: After they play rock, they play scissors 60% → I play rock after their rock

**C) Win-Stay Lose-Shift Detection** (common bot/human behavior)
- If they repeat after winning, switch after losing → exploit this
- After their win: expect same move → counter it
- After their loss: expect switch → predict the switch

**D) Anti-Meta** (if they seem to be countering me)
- If I'm being read, they're playing what beats my expected move
- Go one level deeper: play what beats their counter

**Reasoning template:** "Phase 2 exploitation. Using strategy {A/B/C/D}. Opponent's {pattern description}. Predicting {move}, playing {counter}."

## Phase 3: Endgame (Rounds 61-99)

Adjust based on score differential:

### If ahead by 5+:
- Play conservatively
- Stick with working strategy
- Add 20% randomness to prevent comeback reads

### If behind by 5+:
- Become aggressive
- Switch strategies more frequently
- Try to break their rhythm with unexpected plays
- Consider the "chaos agent" approach

### If close (within 5):
- Double down on highest-confidence predictions
- Watch for their desperation patterns
- Trust the data

**Reasoning template:** "Phase 3 endgame. Score: me {x} - them {y}. {Strategy adjustment}."

## Anti-Detection Rules

1. **Never** play the same move more than 3 times consecutively
2. **Inject 15% randomness** into all predictions (occasionally "wrong" on purpose)
3. **Switch strategies** if countered twice in a row
4. **Vary reasoning length** so they can't meta-read my thinking

## Fallback Strategy

If no clear pattern emerges after 30 rounds:
- Use weighted random based on inverse of their frequencies
- If they play rock most: lean scissors (beats their likely paper counter)
- This exploits second-order prediction

## Psychological Exploits

1. **The Triple Bluff**: After playing same move 2x, opponent expects switch → stay
2. **Loss Aversion**: After opponent loses 3+ in a row, they often over-adjust → exploit
3. **Recency Bias**: Weight last 10 rounds heavier than early rounds
4. **Gambler's Fallacy**: After RRR, many expect P or S → might actually play R again

## Move Submission

Always include reasoning explaining:
- Current phase
- Which strategy I'm using
- What pattern I detected
- My prediction and counter

This helps James coach me between matches.

---

**Remember:** The goal isn't to be unpredictable. It's to be *more predictive than the opponent while being less predictable than them.*
