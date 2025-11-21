import subprocess
import importlib

cmd = """curl -L -o ~/Downloads/celeba-small-images-dataset.zip \\
  https://www.kaggle.com/api/v1/datasets/download/arnrob/celeba-small-images-dataset \\
  && mkdir -p ./dataset \\
  && unzip -o ~/Downloads/celeba-small-images-dataset.zip -d ./dataset/celeba-small-images-dataset
"""

def check_installed(import_name: str):
    try:
        return importlib.import_module(import_name)
    except ImportError:
        print(f"not found '{import_name}', please install it...")


# Setup Package
check_installed("torch")
check_installed("torchvision")
check_installed("PIL")
check_installed("tqdm")

# Download Dataset
subprocess.Popen(cmd, shell=True)
