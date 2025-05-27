import os
import json
import argparse
from time import sleep
from transform_utils import transform_lib

# List of allowed GPUs by their IDs as strings (adjust based on your available GPUs)
allow_gpu = ["0", "1"]  # we have "0" and "1" (two GPUs). Modify to the number of GPUs you have 

# Infinite loop to continuously check for available GPUs and pending runs
while(True):
    # Load the GPU availability status from a shared JSON file
    cuda_file = json.load(open("/workspace/code/cuda.json", "r"))
    
    # Iterate over all GPUs and check if any allowed GPU is free
    for gpu, is_free in cuda_file.items():
        if is_free and gpu in allow_gpu:
            # Load the transformation configuration file
            transf_file = json.load(open(f"/workspace/code/config_unet_inbreast.json", "r"))
            
            # Iterate over each training run configuration
            for run_name, run_params in transf_file.items():
                # Only proceed if the run has not already been marked as running/completed
                if "status" not in run_params:
                    # Construct and launch the training command in the background using nohup
                    command = f"nohup python /workspace/code/unet_inbreast_5folds.py -t {run_name} -gpu {gpu} > unet_inbreast_{run_name}.txt &"
                    os.system(command)
                    
                    # Mark the run as "running" in the config
                    transf_file[run_name]["status"] = "running"
                    json.dump(transf_file, open("/workspace/code/config_unet_inbreast.json", "w"), indent=2)

                    # Mark the GPU as now busy
                    cuda_file[gpu] = False
                    json.dump(cuda_file, open("/workspace/code/cuda.json", "w"), indent=2)
                    break  # Exit the loop after starting one run

    # Sleep for 15 minutes before checking again
    sleep(60 * 15)
