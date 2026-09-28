"""
WorldFloods v2 Dataset Downloader
Downloads test + val (full) + train subset from Hugging Face.
Uses huggingface_hub with controlled concurrency to avoid rate limiting.
"""
import os
import time
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import HfHubHTTPError

# Configuration
REPO_ID = "isp-uv-es/WorldFloodsv2"
LOCAL_DIR = Path(r"d:\superiorparab\final year DL project\data\raw\flood\worldfloods_v2")
TRAIN_SUBSET_COUNT = 150  # Number of train image sets to download

def get_file_list():
    """Get the complete file listing from the HF API."""
    api = HfApi()
    print("Fetching file listing from Hugging Face...")
    files = list(api.list_repo_tree(REPO_ID, repo_type="dataset", recursive=True))
    print(f"Found {len(files)} items in repository")
    return files

def categorize_files(files):
    """Categorize files by split and subdirectory."""
    categories = {
        'test': [], 'val': [], 'train': [], 'root': []
    }
    
    for f in files:
        if hasattr(f, 'rfilename'):
            path = f.rfilename
        elif hasattr(f, 'path'):
            path = f.path
        else:
            continue
            
        # Skip directories
        if not hasattr(f, 'size') and not hasattr(f, 'blob_id'):
            continue
            
        parts = path.split('/')
        if len(parts) >= 2:
            split = parts[0]
            if split in categories:
                categories[split].append(path)
        else:
            categories['root'].append(path)
    
    return categories

def select_train_subset(train_files, count=150):
    """Select a deterministic subset of train files.
    
    Strategy: Get unique base names (without subdirectory), sort alphabetically,
    take first `count`, then include all associated files for those base names.
    """
    # Extract unique base names (the image/event identifier)
    # Files are like: train/S2/EMSR249_03GALWAY_DEL_v2.tif
    # Base name: EMSR249_03GALWAY_DEL_v2
    base_names = set()
    for f in train_files:
        parts = f.split('/')
        if len(parts) >= 3:
            filename = parts[2]
            # Remove extension
            base = os.path.splitext(filename)[0]
            base_names.add(base)
    
    # Sort and take first `count`
    sorted_bases = sorted(base_names)[:count]
    selected_bases = set(sorted_bases)
    
    # Include all files matching selected base names
    selected_files = []
    for f in train_files:
        parts = f.split('/')
        if len(parts) >= 3:
            base = os.path.splitext(parts[2])[0]
            if base in selected_bases:
                selected_files.append(f)
    
    return selected_files, sorted_bases

def download_file_with_retry(repo_id, filename, local_dir, max_retries=5):
    """Download a single file with retry logic and backoff."""
    for attempt in range(max_retries):
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="dataset",
                local_dir=str(local_dir),
            )
            return True
        except HfHubHTTPError as e:
            if "429" in str(e):
                wait_time = min(30 * (2 ** attempt), 300)  # exponential backoff, max 5 min
                print(f"  Rate limited. Waiting {wait_time}s... (attempt {attempt+1}/{max_retries})")
                time.sleep(wait_time)
            else:
                print(f"  HTTP error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                else:
                    return False
        except Exception as e:
            print(f"  Error downloading {filename}: {e}")
            if attempt < max_retries - 1:
                time.sleep(5)
            else:
                return False
    return False

def main():
    # Get file listing
    files = get_file_list()
    categories = categorize_files(files)
    
    print(f"\nRepository structure:")
    print(f"  Root files: {len(categories['root'])}")
    print(f"  Train files: {len(categories['train'])}")
    print(f"  Val files: {len(categories['val'])}")
    print(f"  Test files: {len(categories['test'])}")
    
    # Select files to download
    files_to_download = []
    
    # All root files
    files_to_download.extend(categories['root'])
    
    # All test files
    files_to_download.extend(categories['test'])
    
    # All val files
    files_to_download.extend(categories['val'])
    
    # Train subset
    train_selected, train_bases = select_train_subset(
        categories['train'], count=TRAIN_SUBSET_COUNT
    )
    files_to_download.extend(train_selected)
    
    print(f"\nDownload plan:")
    print(f"  Root: {len(categories['root'])} files")
    print(f"  Test: {len(categories['test'])} files (ALL)")
    print(f"  Val: {len(categories['val'])} files (ALL)")
    print(f"  Train: {len(train_selected)} files ({len(train_bases)} image sets of {TRAIN_SUBSET_COUNT} requested)")
    print(f"  TOTAL: {len(files_to_download)} files")
    
    # Check what's already downloaded
    already_downloaded = 0
    remaining = []
    for f in files_to_download:
        local_path = LOCAL_DIR / f
        if local_path.exists():
            already_downloaded += 1
        else:
            remaining.append(f)
    
    if already_downloaded > 0:
        print(f"\nAlready downloaded: {already_downloaded} files")
    print(f"Remaining to download: {len(remaining)} files")
    
    if not remaining:
        print("\nAll files already downloaded!")
        return
    
    # Download with rate limiting
    print(f"\nStarting download to: {LOCAL_DIR}")
    print("-" * 60)
    
    success = 0
    failed = []
    
    for i, filename in enumerate(remaining):
        print(f"[{i+1}/{len(remaining)}] {filename}")
        
        if download_file_with_retry(REPO_ID, filename, LOCAL_DIR):
            success += 1
        else:
            failed.append(filename)
            print(f"  FAILED: {filename}")
        
        # Small delay between files to avoid rate limiting
        if i < len(remaining) - 1:
            time.sleep(0.5)
    
    print(f"\n{'='*60}")
    print(f"Download complete!")
    print(f"  Success: {success}")
    print(f"  Failed: {len(failed)}")
    if failed:
        print(f"  Failed files:")
        for f in failed:
            print(f"    - {f}")
    print(f"{'='*60}")
    
    # Save the selected train bases for reproducibility
    manifest_path = Path(r"d:\superiorparab\final year DL project\data\metadata")
    manifest_path.mkdir(parents=True, exist_ok=True)
    with open(manifest_path / "train_subset_selection.txt", 'w') as f:
        f.write(f"# WorldFloods v2 Train Subset Selection\n")
        f.write(f"# Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Strategy: First {TRAIN_SUBSET_COUNT} base names alphabetically\n")
        f.write(f"# Total selected: {len(train_bases)}\n\n")
        for base in train_bases:
            f.write(f"{base}\n")
    print(f"\nTrain subset selection saved to: {manifest_path / 'train_subset_selection.txt'}")

if __name__ == "__main__":
    main()
