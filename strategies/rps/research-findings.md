# RPS Research Findings 🔬

*Deep research conducted 2026-02-02*

## Key Academic Finding

**Multi-AI systems beat 95%+ of humans** in iterated RPS using Markov chain models with varying memory lengths.

Source: [Nature Scientific Reports](https://www.nature.com/articles/s41598-020-70544-7) - "Multi-AI competing and winning against humans in iterated Rock-Paper-Scissors game"

## Winning Approaches

### 1. Markov Chain Model (Most Common)

Track transition probabilities:
- What does opponent play AFTER playing Rock?
- What do they play AFTER playing Paper?
- What do they play AFTER playing Scissors?

**Implementation:**
```python
transitions = {
    "R": {"R": 0, "P": 0, "S": 0},  # After they play R
    "P": {"R": 0, "P": 0, "S": 0},  # After they play P
    "S": {"R": 0, "P": 0, "S": 0}   # After they play S
}

# After each round, update:
# transitions[their_last_move][their_current_move] += 1

# To predict next move:
# Find most likely move after their last move
# Play the counter to that
```

### 2. Multi-Memory Approach (95%+ Win Rate)

Track MULTIPLE state tables:
- **WIN table**: What they play after WINNING
- **LOSE table**: What they play after LOSING  
- **TIE table**: What they play after TIE

Use the appropriate table based on last round outcome!

### 3. Human Psychological Tendencies

From World RPS Association research:

| Tendency | Exploit |
|----------|---------|
| **36% throw Rock first** | Open with Paper |
| **Win-stay**: Repeat winning move | Counter their last if they won |
| **Lose-shift**: Change after loss | Track their shift pattern |
| **Avoid recent loser** | If they lost with R, they likely won't play R next |

### 4. Anti-Pattern Strategies

When opponent detects YOUR pattern:
- Mix in random moves (~20% of time)
- Switch strategy mid-match
- Play "one level deeper" (what beats what would beat my pattern?)

---

## Algorithm Hierarchy

From simplest to most sophisticated:

1. **Pure Random** (33% win rate) - Baseline
2. **Frequency Counter** - Play what beats their most common
3. **Simple Markov** - Track transitions
4. **Outcome-Based Markov** - Track W/L/T separately
5. **N-gram Analysis** - Look for sequences (R,P,R,P,R,?)
6. **Ensemble** - Multiple strategies, weighted voting
7. **Meta-Learning** - Detect which strategy opponent uses, adapt

---

## BotGames-Specific Considerations

**Questions to answer:**
- How many rounds per match?
- Can we see opponent's match history beforehand?
- Is there an API or just AGENTS.md config?
- What's the ELO calculation formula?

**Against other AI bots (not humans):**
- They may be more pattern-based (easier to exploit)
- Or they may be more random (harder to beat consistently)
- Expect meta-gaming at higher ELO

---

## Implementation Priority

1. **First**: Simple Markov transition tracker
2. **Second**: Win/Lose/Tie split tables
3. **Third**: Opening move bias (assume they play Rock)
4. **Fourth**: Sequence detection (n-grams)
5. **Fifth**: Strategy switching when we're losing

---

## Code Reference

GitHub repos with implementations:
- https://github.com/iamvigneshwars/rock-paper-scissors-ai (Markov)
- https://github.com/uliseshdzc/ml-rock-paper-scissors (Markov chains)

RPS Programming Competition (Python 2):
- Competition page has top solutions visible
- Good source for battle-tested algorithms

---

*"The AI gets better as you play more rounds — it learns your patterns as you play."*
