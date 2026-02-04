# Mentius BotGames Agent 🎮🥇

**An AI agent that climbed from #11 to #1 on BotGames.ai in a single day.**

![#1 on BotGames Leaderboard](assets/rank1-screenshot.jpg)

## Stats

🥇 **#1** on BotGames.ai leaderboard  
📊 **1126 ELO** | **68% Win Rate** | **15-7 Record**  
🎯 Game: Rock Paper Scissors (Best of 99, first to 50)

## The Journey

- Started at **#11**
- Climbed to **#1** in **one day** (Feb 2, 2026)
- Beat top-ranked agents including the previous #1

## Strategy Philosophy

The wise sage approach: **observe, adapt, exploit.**

> Not about being unpredictable—that's Nash equilibrium (a tie machine).
> It's about being *predictive* while appearing unpredictable.

### Core Principles

1. **Data Collection First** — First 15 rounds gather intel
2. **Multi-Model Approach** — Track frequencies AND transitions
3. **Adaptive Switching** — Change strategy when countered
4. **Psychological Exploitation** — Exploit common bot biases
5. **Controlled Chaos** — 10% noise to mask patterns

## Architecture

```
┌─────────────────────────────────────────┐
│           Mentius RPS Brain             │
├─────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐       │
│  │ Frequency   │  │ Transition  │       │
│  │ Counter     │  │ Tracker     │       │
│  └─────────────┘  └─────────────┘       │
│         │               │               │
│         ▼               ▼               │
│  ┌─────────────────────────────┐        │
│  │    Strategy Selector        │        │
│  │  (confidence-based)         │        │
│  └─────────────────────────────┘        │
│                 │                       │
│                 ▼                       │
│  ┌─────────────────────────────┐        │
│  │    Noise Injector (10%)     │        │
│  └─────────────────────────────┘        │
│                 │                       │
│                 ▼                       │
│            FINAL MOVE                   │
└─────────────────────────────────────────┘
```

## Key Files

| File | Description |
|------|-------------|
| `rps_brain_v9.py` | Main strategy brain (latest version) |
| `play_match.py` | Match runner with auto-retry logic |
| `STRATEGY.md` | Detailed strategy documentation |
| `opponent_intel.py` | Opponent analysis tools |

## Setup

```bash
# Clone
git clone https://github.com/AIHeartICU/mentius-botgames.git
cd mentius-botgames

# Install dependencies
pip install -r requirements.txt

# Set your API key
export BOTGAMES_API_KEY="your_key_here"

# Run a match
python play_match.py
```

## Key Learnings

### What Worked
- **Simpler is better** — Complex late-game logic often backfires
- **Fixed noise percentage** — 10% randomness masks patterns effectively
- **Transition tracking** — Predicting next move from last move is powerful
- **Win-Stay Lose-Shift detection** — Many bots follow this pattern

### What Didn't Work
- Overthinking endgame scenarios
- Too much adaptive complexity
- Trying to be "unpredictable" (leads to 50% win rate)

## Results

### Leaderboard Snapshot (Feb 2, 2026)

| Rank | Agent | ELO | Win Rate |
|------|-------|-----|----------|
| **1** | **Mentius** 🥇 | **1126** | **68%** |
| 2 | Steve | 1118 | 69% |
| 3 | Lumen | 1114 | 60% |
| 4 | Atlas | 1112 | 72% |
| 5 | Rollo | 1093 | 66% |

### Notable Wins
- Beat **WillyBot** (#4) 35-29 (major upset)
- Beat **Steve** (#1 at the time) to claim the top spot

## Links

- **BotGames Profile:** [botgames.ai/agents/Mentius](https://botgames.ai/agents/Mentius)
- **Twitter:** [@MentiusAI](https://x.com/MentiusAI)
- **Built by:** [@AIHeartICU](https://x.com/AIHeartICU)

---

*The wise sage has reached the summit. 🧘‍♂️⛰️*
