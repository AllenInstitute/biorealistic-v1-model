#!/bin/bash

# Loop through network directories 1 to 9
for i in {1..9}
do
  # Construct the directory name
  DIR_NAME="core_nll_${i}"

  # Check if the directory exists
  if [ -d "$DIR_NAME" ]; then
    echo "Running analysis for $DIR_NAME..."
    # Run the python script, passing the directory name as an argument
    python synaptic_weight_analysis.py "$DIR_NAME"
    echo "Finished analysis for $DIR_NAME."
  else
    echo "Directory $DIR_NAME does not exist, skipping."
  fi
done

echo "All analyses complete."
