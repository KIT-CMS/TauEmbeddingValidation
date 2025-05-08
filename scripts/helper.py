from pathlib import Path
import shutil

def initialize_dir(base_path: str, subfolders: list[str]=None):

    base_dir = Path(base_path)

    # Delete the folder if it exists
    if base_dir.exists():
        shutil.rmtree(base_dir)

    # Recreate the base directory
    base_dir.mkdir(parents=True, exist_ok=True)

    if type(subfolders)!=type(None):
        # Create subfolders 
        for sub in subfolders:
            (base_dir / sub).mkdir(parents=True, exist_ok=True)