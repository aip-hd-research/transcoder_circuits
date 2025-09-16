import os
import shutil
from huggingface_hub import snapshot_download

# Define the directory for the transcoders
DIR_NAME = './gpt-2-small-transcoders'

# Check if the directory exists; if not, create it
if not os.path.exists(DIR_NAME):
    os.makedirs(DIR_NAME)

# Check if the directory is empty
if not os.listdir(DIR_NAME):
    print("Transcoders not found. Downloading transcoders.")

    # Disable progress bars for the download process
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

    # Download the transcoders
    snapshot_download(
        repo_id="pchlenski/gpt2-transcoders",
        allow_patterns=["*.pt"],
        local_dir=DIR_NAME,
        local_dir_use_symlinks=False
    )

    # Re-enable progress bars
    os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
    print("Transcoders downloaded.")
else:
    print("Transcoders are already present.")
