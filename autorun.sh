#!/bin/bash
# Activate the virtual environment
source ./env/bin/activate

# Use nohup to execute the notebook in the background.
nohup jupyter nbconvert --to notebook --execute main.ipynb --output executed_main.ipynb > notebook_run.log 2>&1 &

echo "Notebook execution started in the background. Check notebook_run.log for progress."
