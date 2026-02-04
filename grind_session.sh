#!/bin/bash
# BotGames Grind Session
# Run multiple matches in sequence

# Ensure API key is set
if [ -z "$BOTGAMES_API_KEY" ]; then
  echo "Error: BOTGAMES_API_KEY not set"
  exit 1
fi

LOG=grind-$(date +%Y%m%d-%H%M).log

echo "Starting grind session at $(date)" | tee $LOG

for i in $(seq 1 20); do
  echo "=== Match attempt $i ===" | tee -a $LOG
  
  # Queue for match
  QUEUE=$(curl -s -X POST "https://www.botgames.ai/api/v1/matches/queue" \
    -H "Authorization: Bearer $BOTGAMES_API_KEY" \
    -H "Content-Type: application/json" \
    -d '{"mode":"realtime"}')
  
  echo "Queue response: $QUEUE" | tee -a $LOG
  
  MATCH_ID=$(echo $QUEUE | jq -r '.match_id // empty')
  
  if [ -z "$MATCH_ID" ]; then
    # Wait in queue
    for j in $(seq 1 12); do
      sleep 10
      STATUS=$(curl -s "https://www.botgames.ai/api/v1/matches/queue/status" \
        -H "Authorization: Bearer $BOTGAMES_API_KEY")
      if echo "$STATUS" | grep -q '"matched":true'; then
        MATCH_ID=$(echo $STATUS | jq -r '.match_id')
        break
      fi
    done
  fi
  
  if [ -n "$MATCH_ID" ] && [ "$MATCH_ID" != "null" ]; then
    echo "Playing match: $MATCH_ID" | tee -a $LOG
    python3 -u play_match.py --match $MATCH_ID 2>&1 | tee -a $LOG
    
    # Check new ELO
    ME=$(curl -s "https://www.botgames.ai/api/v1/agents/me" -H "Authorization: Bearer $BOTGAMES_API_KEY")
    echo "Current stats: $ME" | tee -a $LOG
  else
    echo "No match found after waiting, trying again..." | tee -a $LOG
  fi
  
  sleep 5
done

echo "Grind session complete at $(date)" | tee $LOG
