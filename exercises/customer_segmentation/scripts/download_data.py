from pathlib import Path
import shutil
import kagglehub

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data' / 'raw'
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Download to kagglehub cache, then copy CSV into project's data/raw/
cache_path = Path(kagglehub.dataset_download("vjchoudhary7/customer-segmentation-tutorial-in-python"))
csv_src = cache_path / 'Mall_Customers.csv'
csv_dest = DATA_DIR / 'Mall_Customers.csv'
shutil.copy(csv_src, csv_dest)

print(f"Dataset saved to: {csv_dest}")
