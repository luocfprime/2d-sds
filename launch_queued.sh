#!/bin/bash

# Check if GPU ID is provided as an argument
if [ -z "$1" ]; then
    echo "Usage: $0 <GPU_ID>"
    exit 1
fi

# Assign the GPU ID to a variable
GPU_ID=$1

# Ensure locks and logs directories exist
mkdir -p locks
mkdir -p logs

# Flag to track if a job was executed
job_executed=false

# Loop over all .sh files in the benchmark_scripts directory
for script in benchmark_scripts/*.sh; do
    # Define the lock file path
    lock_file="locks/$(basename "$script" .sh).lock"

    # Check if the lock file exists
    if [ -e "$lock_file" ]; then
        echo "Lock file $lock_file exists. Skipping $script."
        continue  # Skip this script file
    elif [ -e "$lock_file.crashed" ]; then
        echo "Crash lock file $lock_file.crashed exists. Skipping $script."
        continue  # Skip this script file
    else
        # Lock the file to prevent re-running this script
        touch "$lock_file"

        # Define the log file for the current script
        log_file="logs/$(basename "$script" .sh).log"

        # Run the script with the GPU ID as an argument and capture output to log file
        echo "Running $script with GPU_ID=$GPU_ID"
        bash -x "$script" "$GPU_ID" 2>&1 | tee "$log_file"  # Pass GPU_ID as an argument and log output

        # Check if the script ran successfully
        if [ $? -ne 0 ]; then
            echo "Script $script failed. Marking as crashed."
            rm "$lock_file"  # Remove the original lock
            touch "$lock_file.crashed"  # Mark the lock file as crashed
        else
            echo "Script $script completed successfully."
        fi

        echo "End running $script"

        # Mark that a job was executed and break the loop
        job_executed=true
        break
    fi
done

# If no job was executed, print a message
if ! $job_executed; then
    echo "No jobs left to execute or all jobs are locked."
fi