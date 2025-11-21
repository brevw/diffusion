import argparse
import diffusion.logger as logging
import torch

# Parse command line arguments
parser = argparse.ArgumentParser(description="Setup for Diffusion Model Training and Inference")
parser.add_argument("--train", action="store_true", help="Flag to indicate training mode")
parser.add_argument("--inference", action="store_true", help="Flag to indicate inference mode")
parser.add_argument("--progressive_output", action="store_true", help="Flag to enable progressive output during inference, mainly for debugging")
args = parser.parse_args()

# Set up a logger
logging.Logger.set_current(
    logging.Logger(
        backend=logging.GroupedBackends([logging.StdoutBackend(),logging.JSONBackend("logs.json")]),
        level=logging.DEBUG,
        )
)

# check for accelerators
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")


