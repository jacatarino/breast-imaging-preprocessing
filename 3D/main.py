# ----------------- Imports -----------------
import os
import json
import argparse
from time import sleep
from transform_utils import transform_lib

# ----------------- Configuration -----------------
# Specify which GPUs are allowed to run jobs; currently allowing only GPU "1"
allow_gpu = ["1"]  # Available GPUs are "0" and "1"; modify this list to select active GPUs

# ----------------- Job Scheduling Loop -----------------
# Run indefinitely to check for available GPU and launch training jobs
while(True):
    # Load current GPU availability status from shared JSON file
    cuda_file = json.load(open("/workspace/code/cuda.json", "r"))

    # Iterate over GPUs and check if any are free and allowed
    for gpu, is_free in cuda_file.items():
        if is_free and gpu in allow_gpu:
            # Load transformation configuration file for UNet runs
            transf_file = json.load(open(f"/workspace/code/config_unet_duke.json", "r"))
            
            # Iterate over each training run in the configuration
            for run_name, run_params in transf_file.items():
                # Only consider runs not already marked with a status
                if "status" not in run_params:
                    # Construct command to run the training script asynchronously (nohup + background)
                    command = f"nohup python /workspace/code/unet_dukemri_5folds.py -t {run_name} -gpu {gpu} > unet_duke_{run_name}.txt &"
                    os.system(command)  # Launch the training job

                    # Update config to indicate this run is now in progress
                    transf_file[run_name]["status"] = "running"
                    json.dump(transf_file, open("/workspace/code/config_unet_duke.json", "w"), indent=2)

                    # Mark the GPU as busy
                    cuda_file[gpu] = False
                    json.dump(cuda_file, open("/workspace/code/cuda.json", "w"), indent=2)
                    break  # Exit inner loop after assigning one job

    # Sleep for 15 minutes before checking again
    sleep(60 * 15)
