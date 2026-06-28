# Infrastructure & Technical Findings

**Team Yantra | Wilfred Doré | June 2026**

## Kaggle Competition Setup

### Data Mount Path (CRITICAL)
Kaggle mounts competition data at:
```
/kaggle/input/competitions/ehl-paris-medical-image-retrieval/
```
NOT at `/kaggle/input/ehl-paris-medical-image-retrieval/` (the usual convention).
This extra `competitions/` directory caused all initial kernel failures.

### Image Extension Mismatch
- CSVs reference images as `.nii.gz`
- Actual files on Kaggle are `.nii` (uncompressed)
- Solution: path fallback function that tries `.nii.gz`, then `.nii`

### CSV Locations
```
dataset1/train_pairs.csv    # 350 labeled pairs
dataset1/val_queries.csv    # 40 queries
dataset1/val_gallery.csv    # 40 targets
dataset1/test_queries.csv   # 100 queries
dataset1/test_gallery.csv   # 100 targets
dataset2/val_queries.csv    # 40 queries
dataset2/test_gallery.csv   # 100 targets
dataset3/val_queries.csv    # 20 queries
dataset3/test_gallery.csv   # 77 targets
```
No `sample_submission.csv` on Kaggle (only available via API download).

### Kaggle Kernel Limitations
- 4 CPU cores (no GPU by default)
- ~1h runtime for full feature extraction (1174 volumes)
- 100 submissions/day
- No `unzip` command, no `apt-get install` (sandboxed)

## AMD MI300X Server

### Specs
- GPU: AMD Instinct MI300X VF (205.8 GB VRAM)
- CPU: 20 cores
- RAM: 235 GB
- Disk: 697 GB (595 GB free)
- ROCm: installed at `/opt/rocm/`
- PyTorch: 2.10.0+git8514f05 with CUDA (ROCm backend)
- MONAI: 1.6.0
- nibabel: 5.4.2, numpy: 2.1.3, scipy: 1.17.1, cv2: 4.13.0

### Access Methods
- **SSH (port 22):** Unstable — kernel panics with 3D operations, connection drops
- **JupyterLab (port 80):** Reliable — use API for code execution
- **Jupyter API token:** `aS7z0lLJS5A6fTOA35tTeQK2xfBoiHA76xkdKv%2BXFNa%2F6RuoP` (URL-encoded)
- **Root password:** `Ember7318`

### Jupyter API Execution (from local machine)
```python
# WebSocket-based code execution on the AMD server
import websocket, json, uuid, time

TOKEN = "aS7z0lLJS5A6fTOA35tTeQK2xfBoiHA76xkdKv%2BXFNa%2F6RuoP"
KERNEL_ID = "90bcccac-2ab5-4045-8965-446dbf8efae4"
HOST = "165.245.138.235"

# Upload files via REST API:
# curl -X PUT "http://HOST/api/contents/filename.py?token=TOKEN" \
#   -H "Content-Type: application/json" \
#   -d '{"type":"file","format":"base64","content":"BASE64_CONTENT"}'

# Files appear at /shared-docker/ on the server
```

### Parallelization Strategy
- 20-core multiprocessing for feature extraction (`multiprocessing.Pool`)
- Reduces 700-volume extraction from ~500s (Kaggle 4-core) to ~30s
- MI computation is sequential (350×350 pairs) — bottleneck
- MI optimization: subsample voxels to 5000, reduce bins to 16

### Data Download on AMD
```bash
pip3 install kaggle
mkdir -p ~/.kaggle
echo "KGAT_..." > ~/.kaggle/access_token
chmod 600 ~/.kaggle/access_token
kaggle competitions download ehl-paris-medical-image-retrieval -p /root/data
# No unzip command → use Python zipfile
python3 -c "import zipfile; zipfile.ZipFile('file.zip').extractall('/root/data')"
```

### Direct Submission from AMD
```bash
kaggle competitions submit ehl-paris-medical-image-retrieval \
  -f /root/output/v8_submission.csv \
  -m "description"
```

## entire.io (Mandatory for Judging)
- Must be installed in the GitHub repo for automated codebase evaluation
- Install: `curl -fsSL https://entire.io/install.sh | bash`
- Detects cheating and judges codebase automatically
