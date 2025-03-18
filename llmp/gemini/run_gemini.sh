#!/bin/bash

# Dataset name
DATASET="purchasing"

# List of hyperparameter values
HYPERPARAM_VALUES=(2 10 50 100 500)

# File to store PIDs
PID_FILE="gemini_processes.txt"

# Clear the PID file if it exists
> "$PID_FILE"

echo "=== Starting Gemini Processes ==="
echo "Process IDs will be saved to: $PID_FILE"
echo ""

# Run standard mode (without --mode seq --k2)
for val in "${HYPERPARAM_VALUES[@]}"; do
  echo "Starting: python -u gemini.py $val $DATASET"
  nohup python -u gemini.py "$val" "$DATASET" &
  PID=$!
  echo "PID: $PID - python -u gemini.py $val $DATASET" >> "$PID_FILE"
  echo "Started process with PID: $PID"
  sleep 5  # Small delay between launches to avoid potential issues
done

# Run sequential mode (with --mode seq --k2)
for val in "${HYPERPARAM_VALUES[@]}"; do
  echo "Starting: python -u gemini.py $val $DATASET --mode seq --k2"
  nohup python -u gemini.py "$val" "$DATASET" --mode seq --k2 &
  PID=$!
  echo "PID: $PID - python -u gemini.py $val $DATASET --mode seq --k2" >> "$PID_FILE"
  echo "Started process with PID: $PID"
  sleep 5  # Small delay between launches to avoid potential issues
done

echo ""
echo "=== All jobs launched ==="
echo "To check running processes, use:"
echo "ps -f \`cat $PID_FILE | cut -d' ' -f2 | tr '\n' ' '\`"
echo ""
echo "To kill all these processes if needed:"
echo "kill \`cat $PID_FILE | cut -d' ' -f2\`"
echo "To kill all gemini processes at once:"
echo "  pkill -f gemini.py"
echo "To see all running processes:"
echo " ps -ef | grep "[p]ython""