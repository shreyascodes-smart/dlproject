"""
WorldFloods v2 Dataset Downloader (Direct URL approach)
Builds file list from cached HF API data and downloads using direct URLs
to avoid the rate-limited HF API calls for file listing.
"""
import json
import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from collections import defaultdict

REPO_ID = "isp-uv-es/WorldFloodsv2"
REVISION = "1f3faa2989e69930ac31d5c0a79fd224461123f8"
BASE_URL = f"https://huggingface.co/datasets/{REPO_ID}/resolve/{REVISION}"
LOCAL_DIR = Path(r"d:\superiorparab\final year DL project\data\raw\flood\worldfloods_v2")
TRAIN_SUBSET_COUNT = 150

API_CACHE = r"C:\Users\SHRAVAN\.gemini\antigravity-ide\brain\5125c3d8-8eef-4452-b905-76c530ad4f6b\.system_generated\steps\43\content.md"


def load_file_list_from_api_cache():
    """Load file list from the cached HF API response (siblings list)."""
    with open(API_CACHE, 'r', encoding='utf-8') as f:
        content = f.read()
    
    json_start = content.find('{')
    json_str = content[json_start:]
    data = json.loads(json_str)
    siblings = data.get('siblings', [])
    
    files = []
    for s in siblings:
        fname = s['rfilename']
        if fname != '.gitattributes':  # Skip git internal files
            files.append(fname)
    
    return files


def categorize_and_select(all_files):
    """Categorize files and select the download subset."""
    categories = defaultdict(list)
    
    for f in all_files:
        parts = f.split('/')
        if len(parts) >= 2:
            categories[parts[0]].append(f)
        else:
            categories['root'].append(f)
    
    # Select train subset: get unique base names, sort, take first N
    train_bases = set()
    for f in categories.get('train', []):
        parts = f.split('/')
        if len(parts) >= 3:
            base = os.path.splitext(parts[2])[0]
            train_bases.add(base)
    
    selected_bases = sorted(train_bases)[:TRAIN_SUBSET_COUNT]
    selected_base_set = set(selected_bases)
    
    train_selected = []
    for f in categories.get('train', []):
        parts = f.split('/')
        if len(parts) >= 3:
            base = os.path.splitext(parts[2])[0]
            if base in selected_base_set:
                train_selected.append(f)
    
    # Build final download list
    download_list = []
    download_list.extend(categories.get('root', []))
    download_list.extend(categories.get('test', []))
    download_list.extend(categories.get('val', []))
    download_list.extend(train_selected)
    
    return download_list, selected_bases


def download_file(url, local_path, max_retries=5):
    """Download a single file with retry and exponential backoff."""
    local_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Skip if already exists
    if local_path.exists() and local_path.stat().st_size > 0:
        return True, "already exists"
    
    for attempt in range(max_retries):
        try:
            urllib.request.urlretrieve(url, str(local_path))
            return True, "downloaded"
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = min(60 * (2 ** attempt), 600)
                print(f"    Rate limited. Waiting {wait}s (attempt {attempt+1}/{max_retries})...")
                time.sleep(wait)
            elif e.code == 302:
                # Follow redirect
                redirect_url = e.headers.get('Location', url)
                try:
                    urllib.request.urlretrieve(redirect_url, str(local_path))
                    return True, "downloaded (redirect)"
                except Exception as e2:
                    if attempt < max_retries - 1:
                        time.sleep(2)
            else:
                print(f"    HTTP {e.code}: {e.reason}")
                if attempt < max_retries - 1:
                    time.sleep(5)
        except Exception as e:
            print(f"    Error: {e}")
            if attempt < max_retries - 1:
                time.sleep(5)
    
    return False, "failed"


def main():
    print("=" * 70)
    print("WorldFloods v2 Dataset Download")
    print("=" * 70)
    
    # Load file list from cache
    print("\nLoading file list from cached API data...")
    all_files = load_file_list_from_api_cache()
    print(f"Total files in repository: {len(all_files)}")
    
    # Select subset
    download_list, train_bases = categorize_and_select(all_files)
    
    # Count by category
    root_count = len([f for f in download_list if '/' not in f])
    test_count = len([f for f in download_list if f.startswith('test/')])
    val_count = len([f for f in download_list if f.startswith('val/')])
    train_count = len([f for f in download_list if f.startswith('train/')])
    
    print(f"\nDownload plan:")
    print(f"  Root files: {root_count}")
    print(f"  Test: {test_count} files (ALL)")
    print(f"  Val: {val_count} files (ALL)")
    print(f"  Train: {train_count} files ({len(train_bases)} image sets)")
    print(f"  TOTAL: {len(download_list)} files")
    
    # Check existing
    already = sum(1 for f in download_list if (LOCAL_DIR / f).exists() and (LOCAL_DIR / f).stat().st_size > 0)
    remaining = [f for f in download_list if not ((LOCAL_DIR / f).exists() and (LOCAL_DIR / f).stat().st_size > 0)]
    
    print(f"\nAlready downloaded: {already}")
    print(f"Remaining: {len(remaining)}")
    
    if not remaining:
        print("\nAll files already downloaded!")
        return
    
    # Download
    print(f"\nDownloading to: {LOCAL_DIR}")
    print("-" * 70)
    
    success = 0
    failed_files = []
    total_bytes = 0
    start_time = time.time()
    
    for i, filename in enumerate(remaining):
        url = f"{BASE_URL}/{urllib.request.quote(filename)}"
        local_path = LOCAL_DIR / filename
        
        status_prefix = f"[{i+1}/{len(remaining)}]"
        print(f"{status_prefix} {filename}", end="", flush=True)
        
        ok, reason = download_file(url, local_path)
        
        if ok:
            success += 1
            if local_path.exists():
                size = local_path.stat().st_size
                total_bytes += size
                print(f" - {reason} ({size / 1024 / 1024:.1f} MB)")
            else:
                print(f" - {reason}")
        else:
            failed_files.append(filename)
            print(f" - FAILED")
        
        # Rate limit protection: small delay between downloads
        if i < len(remaining) - 1:
            time.sleep(1.0)  # 1 second between files
    
    elapsed = time.time() - start_time
    
    print(f"\n{'='*70}")
    print(f"Download Summary")
    print(f"  Time: {elapsed/60:.1f} minutes")
    print(f"  Success: {success}/{len(remaining)}")
    print(f"  Downloaded: {total_bytes / 1024**3:.2f} GB")
    print(f"  Failed: {len(failed_files)}")
    if failed_files:
        print(f"  Failed files:")
        for f in failed_files[:10]:
            print(f"    - {f}")
        if len(failed_files) > 10:
            print(f"    ... and {len(failed_files) - 10} more")
    print(f"{'='*70}")
    
    # Save train subset selection
    meta_dir = Path(r"d:\superiorparab\final year DL project\data\metadata")
    meta_dir.mkdir(parents=True, exist_ok=True)
    with open(meta_dir / "train_subset_selection.txt", 'w') as f:
        f.write(f"# WorldFloods v2 Train Subset Selection\n")
        f.write(f"# Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Strategy: First {TRAIN_SUBSET_COUNT} base names alphabetically\n")
        f.write(f"# Total unique image sets: {len(train_bases)}\n\n")
        for base in train_bases:
            f.write(f"{base}\n")
    print(f"\nTrain subset saved to: {meta_dir / 'train_subset_selection.txt'}")


if __name__ == "__main__":
    main()
