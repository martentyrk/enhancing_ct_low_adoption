#!/bin/bash
#SBATCH --output=merge_files%A.out

merged_file_name="train_merge_06.jl"

# Check if merged_file.txt already exists and delete if it does
if [ -f "$merged_file_name" ]; then
    rm "$merged_file_name"
fi

# dirname='dpfn/data/data_all_users/merge0.6_0.8'
dirname='../../../../scratch-shared/mturk/datadump_06_include_all15seeds_covasim/train'
dirname_out=${dirname}_out

cd "$dirname"

# Check if the directory change was successful
if [ $? -ne 0 ]; then
    echo "Error: Directory not found or inaccessible."
    exit 1
fi

# Loop through all files in the specified directory
for file in *; do
    # Check if the file is a regular file and not the script itself
    if [ "$file" != "$merged_file_name" ]; then
        # Append the content of the file to merged_file.txt
        cat "$file" >> "$merged_file_name"
    fi
done

echo "All files in $DIRECTORY have been merged into $merged_file_name."