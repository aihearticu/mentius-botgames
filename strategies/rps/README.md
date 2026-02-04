# Rock Paper Scissors Strategy 🪨📄✂️

## The Game

Rock Paper Scissors seems trivially random, but against AI opponents, it becomes a battle of:
- Pattern recognition
- Opponent modeling
- Meta-game adaptation
- Psychological warfare (for human opponents)

## Theoretical Foundation

### Nash Equilibrium
- True random (33/33/33) is unexploitable
- But it's also unexciting and can't exploit patterns
- Most bots don't play true random

### Exploitable Patterns

Common bot behaviors:
1. **Win-stay, lose-shift** — Repeat winning move, change losing move
2. **Counter previous** — Play what would beat opponent's last move
3. **Cycle** — Rock → Paper → Scissors → Rock...
4. **Frequency balancing** — Try to maintain 33/33/33 (exploitable via timing)
5. **Anti-pattern** — Try to counter expected patterns (second-level thinking)

## Mentius Strategy

### Phase 1: Cold Start (First 5-10 moves)
- Play near-random but slightly biased
- Gather data on opponent tendencies
- Look for immediate patterns

### Phase 2: Pattern Detection (Ongoing)
- Track opponent's:
  - Move frequencies (R/P/S distribution)
  - Transition probabilities (what do they play after R? after P? after S?)
  - Win/loss response (do they repeat winners? change after loss?)
  - Sequence patterns (any repeating sequences?)

### Phase 3: Exploitation
- Once pattern detected, shift to counter-strategy
- But don't be too predictable in exploitation
- Mix exploitation with occasional randomness

### Phase 4: Adaptation Detection
- If opponent adapts to our exploitation, detect it
- Re-enter pattern detection mode
- Stay one level ahead

## Key Metrics to Track

For each opponent:
```python
{
    "move_counts": {"R": 0, "P": 0, "S": 0},
    "transitions": {
        "R": {"R": 0, "P": 0, "S": 0},  # After playing R, what do they play?
        "P": {"R": 0, "P": 0, "S": 0},
        "S": {"R": 0, "P": 0, "S": 0}
    },
    "after_win": {"R": 0, "P": 0, "S": 0},   # What do they play after winning?
    "after_loss": {"R": 0, "P": 0, "S": 0},  # After losing?
    "after_tie": {"R": 0, "P": 0, "S": 0},   # After tie?
    "last_5_moves": [],
    "our_last_5": []
}
```

## Counter Strategies

| If opponent tends to... | We should... |
|-------------------------|--------------|
| Repeat winning move | Counter their last move |
| Play what beats our last | Play what beats our last (same as them!) |
| Cycle R→P→S | Play one step ahead |
| Play high Rock% | Play high Paper% |
| Be truly random | Play random (accept ~50% win rate) |

## Match Format Questions

- Best of N? Or total points?
- Do we see opponent moves in real-time?
- Is there a move time limit?
- Can we access opponent's match history?

## TODO

- [ ] Sign up for BotGames.ai
- [ ] Understand API/integration format
- [ ] Build pattern detection algorithm
- [ ] Test against simple bots locally
- [ ] Play first ranked match
- [ ] Analyze and iterate

---

*"In the midst of chaos, there is also opportunity."* — Sun Tzu
