#!/bin/bash

# Create the main output directory


# Loop through all bird categories (using Train directory as reference)
for category in /Users/chiragtagadiya/MyProjects/EMLO_V4_projects/deployment_projects/DVC-pytorch-lightning-MLOps/data/Bird200/Train/*; do
    # Get the base category name
    category_name=$(basename "$category")
    
    # Create category directory in merged folder
    mkdir -p "data/merged_birds/$category_name"
    
    # Copy images from Train directory
    cp -r "data/Bird200/Train/$category_name"/* "data/merged_birds/$category_name/"
    
    # Copy images from Test directory
    cp -r "data/Bird200/Test/$category_name"/* "data/merged_birds/$category_name/"
done

echo "Bird data merging completed!"
