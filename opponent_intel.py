#!/usr/bin/env python3
"""
Opponent Intelligence Scraper

Scrapes BotGames.ai match replays to extract:
- Opponent move history
- Exposed reasoning/strategy
- Per-opponent patterns

Uses Playwright for browser automation since the API is unavailable.
"""

import asyncio
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter
from datetime import datetime

# Try to import playwright
try:
    from playwright.async_api import async_playwright, Browser, Page
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    print("Warning: Playwright not installed. Run: pip install playwright && playwright install chromium")

# Local imports for analysis
MOVES = ['rock', 'paper', 'scissors']
BEATS = {'rock': 'paper', 'paper': 'scissors', 'scissors': 'rock'}

# Config
DATA_DIR = Path(__file__).parent / "opponent_data"
BOTGAMES_URL = "https://botgames.ai"
API_KEY = os.environ.get("BOTGAMES_API_KEY", "")


class OpponentProfile:
    """Stores analyzed patterns for an opponent."""

    def __init__(self, name: str):
        self.name = name
        self.matches: List[Dict] = []
        self.total_moves: Dict[str, int] = {'rock': 0, 'paper': 0, 'scissors': 0}
        self.transitions: Dict[str, Dict[str, int]] = {
            m: {'rock': 0, 'paper': 0, 'scissors': 0} for m in MOVES
        }
        self.reasoning_patterns: List[str] = []
        self.strategy_type: Optional[str] = None
        self.win_rate_against_us: float = 0.0

    def analyze_match(self, match_data: Dict):
        """Analyze a single match and update patterns."""
        history = match_data.get('history', [])

        last_opp_move = None
        for round_data in history:
            opp_move = round_data.get('opponent_move')
            if opp_move:
                self.total_moves[opp_move] += 1

                if last_opp_move:
                    self.transitions[last_opp_move][opp_move] += 1

                last_opp_move = opp_move

            # Extract reasoning if available
            opp_reasoning = round_data.get('opponent_reasoning', '')
            if opp_reasoning and len(opp_reasoning) > 10:
                self.reasoning_patterns.append(opp_reasoning)

        self.matches.append(match_data)

    def classify_strategy(self) -> str:
        """Classify opponent's strategy type based on patterns."""
        total = sum(self.total_moves.values())
        if total < 20:
            return 'unknown'

        # Check for spam (>50% one move)
        for move, count in self.total_moves.items():
            if count / total > 0.50:
                return f'spam_{move}'

        # Check for strong transitions
        for from_move, to_moves in self.transitions.items():
            total_trans = sum(to_moves.values())
            if total_trans >= 10:
                for to_move, count in to_moves.items():
                    if count / total_trans > 0.60:
                        return f'transition_{from_move}_to_{to_move}'

        # Check reasoning patterns for strategy hints
        reasoning_text = ' '.join(self.reasoning_patterns).lower()
        if 'meta' in reasoning_text or 'level' in reasoning_text:
            return 'meta_predictor'
        if 'frequency' in reasoning_text or 'freq' in reasoning_text:
            return 'frequency_based'
        if 'transition' in reasoning_text or 'trans' in reasoning_text:
            return 'transition_based'
        if 'wsls' in reasoning_text or 'stay' in reasoning_text:
            return 'wsls_based'
        if 'random' in reasoning_text:
            return 'random_injector'

        # Check for balanced distribution (likely adaptive)
        max_pct = max(self.total_moves.values()) / total
        if max_pct < 0.40:
            return 'balanced_adaptive'

        return 'pattern_based'

    def get_counter_strategy(self) -> Dict:
        """Get recommended counter-strategy."""
        strategy = self.classify_strategy()
        self.strategy_type = strategy

        counters = {
            'spam_rock': {'play': 'paper', 'confidence': 0.85},
            'spam_paper': {'play': 'scissors', 'confidence': 0.85},
            'spam_scissors': {'play': 'rock', 'confidence': 0.85},
            'meta_predictor': {'play': 'meta_p2', 'confidence': 0.60},
            'frequency_based': {'play': 'meta_p1', 'confidence': 0.65},
            'transition_based': {'play': 'break_transitions', 'confidence': 0.60},
            'wsls_based': {'play': 'exploit_wsls', 'confidence': 0.70},
            'balanced_adaptive': {'play': 'high_noise', 'confidence': 0.50},
            'pattern_based': {'play': 'meta_p1', 'confidence': 0.60},
            'unknown': {'play': 'standard', 'confidence': 0.50}
        }

        # Handle transition patterns
        if strategy.startswith('transition_'):
            parts = strategy.split('_')
            if len(parts) >= 4:
                from_move = parts[1]
                to_move = parts[3]
                counter = BEATS[to_move]
                return {
                    'play': f'counter_transition_{from_move}_{counter}',
                    'confidence': 0.70,
                    'note': f'When they play {from_move}, expect {to_move}, play {counter}'
                }

        return counters.get(strategy, counters['unknown'])

    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        return {
            'name': self.name,
            'total_moves': self.total_moves,
            'transitions': self.transitions,
            'strategy_type': self.classify_strategy(),
            'counter_strategy': self.get_counter_strategy(),
            'match_count': len(self.matches),
            'sample_reasoning': self.reasoning_patterns[:5],
            'updated_at': datetime.now().isoformat()
        }


class OpponentIntelScraper:
    """Scrapes BotGames.ai for opponent intelligence."""

    def __init__(self):
        self.browser: Optional[Browser] = None
        self.profiles: Dict[str, OpponentProfile] = {}
        DATA_DIR.mkdir(exist_ok=True)

    async def init_browser(self):
        """Initialize Playwright browser."""
        if not PLAYWRIGHT_AVAILABLE:
            raise RuntimeError("Playwright not available")

        playwright = await async_playwright().start()
        self.browser = await playwright.chromium.launch(headless=True)

    async def close(self):
        """Close browser."""
        if self.browser:
            await self.browser.close()

    async def scrape_match_replay(self, match_id: str) -> Optional[Dict]:
        """Scrape a single match replay."""
        if not self.browser:
            await self.init_browser()

        page = await self.browser.new_page()
        match_data = None

        try:
            url = f"{BOTGAMES_URL}/matches/{match_id}"
            await page.goto(url, wait_until='networkidle', timeout=30000)

            # Wait for match data to load
            await page.wait_for_selector('.match-history, .round-history, [class*="match"]',
                                        timeout=10000)

            # Extract match data from page
            match_data = await page.evaluate('''() => {
                // Try to find match data in various formats
                const data = {
                    id: window.location.pathname.split('/').pop(),
                    history: [],
                    opponent: null,
                    result: null
                };

                // Look for round elements
                const rounds = document.querySelectorAll('[class*="round"], .history-item, tr');
                rounds.forEach((round, idx) => {
                    const text = round.textContent;
                    const moves = text.match(/(rock|paper|scissors)/gi);
                    if (moves && moves.length >= 2) {
                        data.history.push({
                            round: idx + 1,
                            your_move: moves[0].toLowerCase(),
                            opponent_move: moves[1].toLowerCase()
                        });
                    }
                });

                // Try to find opponent name
                const oppElement = document.querySelector('[class*="opponent"], .vs-opponent');
                if (oppElement) {
                    data.opponent = oppElement.textContent.trim();
                }

                return data;
            }''')

        except Exception as e:
            print(f"Error scraping match {match_id}: {e}")
        finally:
            await page.close()

        return match_data

    async def scrape_agent_matches(self, agent_name: str = "Mentius", limit: int = 20) -> List[Dict]:
        """Scrape matches for an agent."""
        if not self.browser:
            await self.init_browser()

        page = await self.browser.new_page()
        matches = []

        try:
            # Go to agent profile or matches page
            url = f"{BOTGAMES_URL}/agents/{agent_name}"
            await page.goto(url, wait_until='networkidle', timeout=30000)

            # Look for match links
            match_links = await page.evaluate('''() => {
                const links = document.querySelectorAll('a[href*="/matches/"]');
                return Array.from(links).map(a => a.href).slice(0, 20);
            }''')

            for link in match_links[:limit]:
                match_id = link.split('/')[-1]
                match_data = await self.scrape_match_replay(match_id)
                if match_data and match_data.get('history'):
                    matches.append(match_data)
                    print(f"  Scraped match {match_id}: {len(match_data['history'])} rounds")

        except Exception as e:
            print(f"Error scraping agent matches: {e}")
        finally:
            await page.close()

        return matches

    def analyze_opponent(self, opponent_name: str, matches: List[Dict]) -> OpponentProfile:
        """Analyze opponent from match data."""
        profile = OpponentProfile(opponent_name)

        for match in matches:
            if match.get('opponent') == opponent_name or not match.get('opponent'):
                profile.analyze_match(match)

        self.profiles[opponent_name] = profile
        return profile

    def save_profiles(self):
        """Save all profiles to disk."""
        for name, profile in self.profiles.items():
            filepath = DATA_DIR / f"{name.lower().replace(' ', '_')}.json"
            with open(filepath, 'w') as f:
                json.dump(profile.to_dict(), f, indent=2)
            print(f"Saved profile: {filepath}")

    def load_profiles(self) -> Dict[str, OpponentProfile]:
        """Load profiles from disk."""
        for filepath in DATA_DIR.glob("*.json"):
            try:
                with open(filepath) as f:
                    data = json.load(f)
                name = data.get('name', filepath.stem)
                profile = OpponentProfile(name)
                profile.total_moves = data.get('total_moves', {})
                profile.transitions = data.get('transitions', {})
                profile.strategy_type = data.get('strategy_type')
                self.profiles[name] = profile
            except Exception as e:
                print(f"Error loading {filepath}: {e}")

        return self.profiles


def analyze_local_history(history: List[Tuple[str, str]]) -> Dict:
    """Analyze opponent patterns from local match history.

    Args:
        history: List of (my_move, opp_move) tuples

    Returns:
        Analysis dict with patterns
    """
    if len(history) < 10:
        return {'status': 'insufficient_data'}

    opp_moves = [h[1] for h in history]

    # Frequency analysis
    freq = Counter(opp_moves)
    total = len(opp_moves)

    # Transition analysis
    transitions = {m: Counter() for m in MOVES}
    for i in range(1, len(opp_moves)):
        prev = opp_moves[i-1]
        curr = opp_moves[i]
        transitions[prev][curr] += 1

    # Find strongest patterns
    patterns = []

    # Check for spam
    for move, count in freq.items():
        pct = count / total
        if pct > 0.45:
            patterns.append(f"spam_{move}_{pct:.0%}")

    # Check for transitions
    for from_move, to_counts in transitions.items():
        total_trans = sum(to_counts.values())
        if total_trans >= 5:
            for to_move, count in to_counts.items():
                pct = count / total_trans
                if pct > 0.55:
                    patterns.append(f"transition_{from_move}>{to_move}_{pct:.0%}")

    return {
        'frequency': dict(freq),
        'frequency_pct': {m: c/total for m, c in freq.items()},
        'transitions': {m: dict(c) for m, c in transitions.items()},
        'patterns': patterns,
        'recommended_counter': _get_counter_from_patterns(patterns, freq)
    }


def _get_counter_from_patterns(patterns: List[str], freq: Counter) -> str:
    """Get counter move recommendation from patterns."""
    # Priority: spam > transition > frequency

    for p in patterns:
        if p.startswith('spam_'):
            move = p.split('_')[1]
            return BEATS[move]

    # Else counter most common
    if freq:
        most_common = freq.most_common(1)[0][0]
        return BEATS[most_common]

    return 'random'


# ============================================================================
# CLI Interface
# ============================================================================

async def main():
    """Main CLI for opponent intelligence."""
    import argparse

    parser = argparse.ArgumentParser(description='Opponent Intelligence Scraper')
    parser.add_argument('--scrape', action='store_true', help='Scrape matches from BotGames')
    parser.add_argument('--agent', default='Mentius', help='Agent name to scrape')
    parser.add_argument('--limit', type=int, default=20, help='Max matches to scrape')
    parser.add_argument('--analyze', help='Analyze specific opponent')
    parser.add_argument('--list', action='store_true', help='List saved profiles')

    args = parser.parse_args()

    scraper = OpponentIntelScraper()

    if args.list:
        profiles = scraper.load_profiles()
        print(f"\nLoaded {len(profiles)} opponent profiles:")
        for name, profile in profiles.items():
            print(f"  - {name}: {profile.strategy_type or 'unclassified'}")
        return

    if args.scrape:
        if not PLAYWRIGHT_AVAILABLE:
            print("Error: Playwright required for scraping")
            print("Install with: pip install playwright && playwright install chromium")
            return

        print(f"Scraping matches for {args.agent}...")
        await scraper.init_browser()

        try:
            matches = await scraper.scrape_agent_matches(args.agent, args.limit)
            print(f"\nScraped {len(matches)} matches")

            # Group by opponent and analyze
            opponents = {}
            for match in matches:
                opp = match.get('opponent', 'Unknown')
                if opp not in opponents:
                    opponents[opp] = []
                opponents[opp].append(match)

            for opp_name, opp_matches in opponents.items():
                profile = scraper.analyze_opponent(opp_name, opp_matches)
                print(f"\n{opp_name}:")
                print(f"  Strategy: {profile.classify_strategy()}")
                print(f"  Counter: {profile.get_counter_strategy()}")

            scraper.save_profiles()

        finally:
            await scraper.close()

    elif args.analyze:
        profiles = scraper.load_profiles()
        if args.analyze in profiles:
            profile = profiles[args.analyze]
            print(f"\nProfile: {profile.name}")
            print(f"  Moves: {profile.total_moves}")
            print(f"  Strategy: {profile.classify_strategy()}")
            print(f"  Counter: {profile.get_counter_strategy()}")
        else:
            print(f"No profile found for: {args.analyze}")
            print(f"Available: {list(profiles.keys())}")

    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())
