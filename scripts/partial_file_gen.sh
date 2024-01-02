#!/bin/bash
# Input and output files
INPUT_FILE="dpfn/data/val_app_users/all_35_0.6.jl"
OUTPUT_FILE="dpfn/data/val_app_users/all_35_0.6_exp_partial.jl"

# Check if the input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Input file not found: $INPUT_FILE"
    exit 1
fi

# Calculate 33% of the number of lines in the input file
TOTAL_LINES=$(wc -l < "$INPUT_FILE")
LINES_TO_SAVE=$((TOTAL_LINES * 1 / 100))

# Save the first 33% of lines of the input file to the output file
head -n $LINES_TO_SAVE "$INPUT_FILE" > "$OUTPUT_FILE"
