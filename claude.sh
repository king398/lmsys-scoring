#!/bin/bash

# Check if correct number of arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <directory_path> <output_file_path>"
    exit 1
fi

directory_path="$1"
output_file_path="$2"

# Check if the directory exists
if [ ! -d "$directory_path" ]; then
    echo "Error: The directory $directory_path does not exist."
    exit 1
fi

# Clear the output file if it already exists
> "$output_file_path"

# Loop through all files in the directory
for file in "$directory_path"/*; do
    # Check if it's a regular file
    if [ -f "$file" ]; then
        # Write the filename to the output file
        echo "--- Contents of $(basename "$file") ---" >> "$output_file_path"
        echo "" >> "$output_file_path"

        # Try to read the file contents
        if cat "$file" >> "$output_file_path" 2>/dev/null; then
            echo "" >> "$output_file_path"
            echo "" >> "$output_file_path"
        else
            echo "Error reading file $(basename "$file")" >> "$output_file_path"
            echo "" >> "$output_file_path"
            echo "" >> "$output_file_path"
        fi
    fi
done

echo "Contents of all files have been written to $output_file_path"
