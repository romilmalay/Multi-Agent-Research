# Multi-Agent Research System

A 7-agent / 8-node LangGraph pipeline that turns a research question into a cited,
peer-reviewed report, served as a job by a FastAPI + Postgres backend with a React UI.


## Layout

```
backend/    research_system/ (the agent, no FastAPI/DB) + app/ (the service)
frontend/   React + Vite + TypeScript
```

`app` may import `research_system`. Never the reverse — enforced by import-linter.

## Develop

```bash
make install          # uv sync + pre-commit hooks
make lint type test   # must be green before a phase is done
```

Copy `backend/.env.example` to `backend/.env` and fill in the API keys.

## Status

Phase 0 (toolchain and scaffold) complete. See the progress list in `plan.md`.


Road Detection — Data Collection (3 Days, 3 People)

Scope: acquisition only. Get data onto disk, verified and catalogued. Not in scope: auditing, cleaning, splitting, training. That's AFTER_COLLECTION.md. Done means: every dataset downloaded, checksummed, extracted, catalogued, and a one-page inventory exists.

Roles
	Owner	Owns
A	You (lead)	Primary dataset (SpaceNet 3) + storage + coordination
B	Engineer 2	Secondary labelled datasets (cross-dataset generalisation)
C	Engineer 3	Unlabelled imagery + OSM weak labels + inventory tooling

Downloads are bandwidth-bound and mostly waiting. Parallelise across three machines/accounts, don't queue them.

Prerequisites — do this in the first hour, all three of you
Item	Command / link
AWS account (free; SpaceNet is Open Data)	https://aws.amazon.com
IAM user + AmazonS3ReadOnlyAccess + access key	Console → IAM
AWS CLI	curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o a.zip && unzip a.zip && sudo ./aws/install
Configure	aws configure (region us-east-1)
Kaggle account + API token	https://www.kaggle.com → Account → Create API Token → ~/.kaggle/kaggle.json, chmod 600
Disk	500 GB free minimum per machine handling SpaceNet
Shared storage	One Azure Blob container, all three have write access

Blocker check — run before anything else:

bash
aws s3 ls s3://spacenet-dataset/spacenet/SN3_roads/tarballs/

If this hangs or errors, S3 is blocked on your network. Escalate immediately — it's a 1-day fix if raised now and a 3-day loss if discovered tomorrow. Fallback path is in §4.

1. Person A — SpaceNet 3 (primary)

Source: https://spacenet.ai/spacenet-roads-dataset/ · https://registry.opendata.aws/spacenet/ Content: 4 cities (Las Vegas, Paris, Shanghai, Khartoum), 0.3 m WorldView-3, ~8,000 km hand-traced road centerlines, GeoJSON labels Licence: CC BY-SA 4.0 (attribution + share-alike — flag to whoever handles legal) Size: ~80 GB tarballs, ~250 GB extracted

Step 1 — list and copy exact filenames
bash
aws s3 ls s3://spacenet-dataset/spacenet/SN3_roads/tarballs/

Copy filenames from this output. Do not retype them from any document, including this one.

Step 2 — sample first (728 MB)
bash
mkdir -p ~/data/raw/spacenet3 && cd ~/data/raw/spacenet3
aws s3 cp s3://spacenet-dataset/spacenet/SN3_roads/tarballs/SN3_roads_sample.tar.gz .
tar -xzf SN3_roads_sample.tar.gz
find . -maxdepth 3 -type d

Confirm you see imagery folders (PS-RGB, PS-MS, MS, PAN) and a GeoJSON labels folder before pulling 80 GB.

Step 3 — full download, in a detached session
bash
screen -S sn3     # or: tmux new -s sn3
cd ~/data/raw/spacenet3

for city in \
  SN3_roads_train_AOI_2_Vegas \
  SN3_roads_train_AOI_3_Paris \
  SN3_roads_train_AOI_4_Shanghai \
  SN3_roads_train_AOI_5_Khartoum
do
  aws s3 cp s3://spacenet-dataset/spacenet/SN3_roads/tarballs/${city}.tar.gz .
  aws s3 cp s3://spacenet-dataset/spacenet/SN3_roads/tarballs/${city}_geojson_roads_speed.tar.gz .
done

⚠️ Two tarballs per city — imagery AND _geojson_roads_speed labels. Missing the second one is the most common mistake here. ⚠️ Verify the Shanghai/Khartoum names against Step 1 output; if one is wrong the loop skips that city quietly.

Detach: Ctrl+A then D. Reattach: screen -r sn3.

Step 4 — extract and verify
bash
for f in *.tar.gz; do tar -xzf "$f"; done

find . -name "*.tif"     | wc -l     # must be > 0
find . -name "*.geojson" | wc -l     # must be > 0
du -sh */

If GeoJSON count is 0, you downloaded imagery without labels.

Step 5 — optional, if time allows: SpaceNet 5

Adds Moscow + Mumbai and road-speed attributes. Mumbai is directly relevant if this ever targets Indian geography.

bash
aws s3 ls s3://spacenet-dataset/spacenet/SN5_roads/tarballs/

Reference: https://spacenet.ai/sn5-challenge/

2. Person B — Secondary labelled datasets

Purpose: a second dataset from a different source is what lets you test whether a model generalises beyond one benchmark. Also your fallback if SpaceNet access fails.

2.1 DeepGlobe Road Extraction — priority

RGB, 50 cm, ~8,570 images, Thailand/Indonesia/India — the closest public road dataset to Indian imagery.

bash
pip install kaggle
kaggle datasets download -d balraj98/deepglobe-road-extraction-dataset
unzip deepglobe-road-extraction-dataset.zip -d ~/data/raw/deepglobe

~4 GB. Masks are already rasterised — no GeoJSON conversion needed, so it's the fastest path to a first training run.

2.2 Massachusetts Roads

Small, aerial, ~1 m. Ideal for smoke-testing the pipeline in minutes rather than hours.

Kaggle: search massachusetts roads segmentation
Original: https://www.cs.toronto.edu/~vmnih/data/
2.3 If time allows
Dataset	Why	Where
CHN6-CUG	Chinese cities, 50 cm, road-specific	Search GitHub — verify current mirror
CityScale / SpaceNet graph splits	The splits used by modern graph-extraction papers — makes your numbers comparable	https://github.com/htcr/sam_road
SpaceNet 8	Roads + flood context	aws s3 ls s3://spacenet-dataset/spacenet/SN8_floods/

Don't chase all of these. DeepGlobe + Massachusetts is enough.

3. Person C — Unlabelled imagery, weak labels, and tooling
3.1 OSM road vectors (weak labels, global, free)

Country/region extracts, updated regularly:

bash
# https://download.geofabrik.de
wget https://download.geofabrik.de/asia/india-latest.osm.pbf

pip install osmium-tool  # or: sudo apt install osmium-tool
osmium tags-filter india-latest.osm.pbf w/highway -o roads-india.osm.pbf

Also useful: https://overpass-turbo.eu for interactive extraction of small areas.

⚠️ Licence: ODbL — share-alike on derived databases. Same legal flag as SpaceNet.

3.2 Unlabelled imagery to pair with OSM
Source	Resolution	Access
Sentinel-2	10 m	https://dataspace.copernicus.eu — free, global, needs registration
NAIP (US)	0.6–1 m	https://registry.opendata.aws/naip
Maxar Open Data	VHR, disaster AOIs	https://registry.opendata.aws/maxar-open-data
Bhuvan (ISRO, India)	varies	https://bhuvan.nrsc.gov.in

⚠️ Sentinel-2 at 10 m is too coarse for road centerlines — useful for context and pretraining, not for this task's labels. Don't over-invest here.

⚠️ Do not scrape Google/Bing/Esri tiles. Their terms prohibit it for training data. If someone suggests it, that's a legal problem, not a shortcut.

3.3 Label-prep tooling — set this up now, use it next week

SpaceNet ships 16-bit GeoTIFFs and vector labels. You need 8-bit imagery and raster masks. Don't write this yourself.

bash
git clone https://github.com/CosmiQ/cresi ~/tools/cresi

The scripts you'll need: create_8bit_masks.py and speed_masks.py (see the repo's "SpaceNet 5 Baseline Part 1 — Data Prep" walkthrough). Also worth cloning as a reference for the SpaceNet directory conventions: https://github.com/aavek/Satellite-Image-Road-Segmentation

Just clone and confirm they run on the sample tarball. Don't process the full dataset yet.

3.4 Inventory script — Person C's main deliverable
python
# tools/inventory.py — run over every raw directory
import hashlib, pathlib, rasterio, pandas as pd

rows = []
for p in pathlib.Path("~/data/raw").expanduser().rglob("*"):
    if not p.is_file():
        continue
    rec = {"path": str(p), "bytes": p.stat().st_size, "suffix": p.suffix}
    rec["sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
    if p.suffix.lower() in {".tif", ".tiff"}:
        try:
            with rasterio.open(p) as src:
                rec |= {"w": src.width, "h": src.height, "bands": src.count,
                        "dtype": src.dtypes[0], "crs": str(src.crs),
                        "gsd": abs(src.transform.a)}
        except Exception as e:
            rec["error"] = str(e)
    rows.append(rec)

pd.DataFrame(rows).to_parquet("data/inventory.parquet")

Install: pip install rasterio pandas pyarrow (use conda-forge or a prebuilt image if GDAL fights you — budget half a day for this, it's the classic blocker).

4. If S3 is blocked

Don't lose the three days. Reorder:

DeepGlobe becomes the primary dataset. Kaggle only, no AWS. Same task, masks already rasterised — you can be training in days rather than weeks.
Escalate the S3 firewall request in parallel.
Alternative SpaceNet routes to check: Azure Open Datasets, Radiant MLHub, or a mirror on HuggingFace Datasets.
Everything in AFTER_COLLECTION.md works unchanged with DeepGlobe.
5. Three-day schedule
Day 1 — Access and start the pipes
Who	Task
All	Accounts, CLI, aws s3 ls blocker check
A	Blob container created, access granted to B and C
A	Sample tarball → verify structure → start the full 80 GB download
B	DeepGlobe + Massachusetts downloaded and extracted
C	rasterio/GDAL environment working; clone CRESI; write inventory.py

End of day: SpaceNet download running unattended. Everyone's tooling works.

Day 2 — Finish downloads, catalogue
Who	Task
A	Downloads complete → extract → verify TIF and GeoJSON counts per city
A	Optional: SpaceNet 5 (Mumbai/Moscow)
B	Structure both secondary datasets into a consistent folder layout; count images/masks
B	Open 10 samples from each — confirm imagery and masks actually correspond
C	OSM extract for one target region + one unlabelled imagery source
C	Run inventory.py across everything downloaded so far

End of day: all bulk data on disk.

Day 3 — Verify, upload, report
Who	Task
All	Re-run inventory across everything; resolve errors and orphans
A	Upload all raw data to blob (az storage blob upload-batch)
A	Record every licence: SpaceNet CC BY-SA, DeepGlobe, OSM ODbL
B	Open ~20 tiles across all datasets, eyeball imagery + labels together
C	Finalise inventory.parquet; generate the summary tables
All	Write D1_COLLECTION_REPORT.md
6. Exit criteria — all must be true
 SpaceNet 3: all 4 cities, imagery and _geojson_roads_speed labels, extracted
 find . -name "*.tif" | wc -l and *.geojson both non-zero per city
 DeepGlobe extracted, images and masks paired
 Massachusetts Roads present (smoke-test set)
 OSM extract for at least one target region
 inventory.parquet covers every file: path, sha256, size, dimensions, bands, dtype, CRS, GSD
 Zero unreadable/corrupt files, or every one listed as a known exception
 Zero orphans — no image without a label, no label without an image
 Everything uploaded to shared blob storage
 Licences recorded per dataset
 D1_COLLECTION_REPORT.md written
7. D1_COLLECTION_REPORT.md template
markdown
# Data Collection Report

## Datasets acquired
| Dataset | Source | Size | Files | Labelled | Licence |
|---|---|---|---|---|---|
| SpaceNet 3 | AWS s3://spacenet-dataset | | | Yes (GeoJSON centerlines) | CC BY-SA 4.0 |
| DeepGlobe Roads | Kaggle | | | Yes (raster masks) | (record) |
| Massachusetts Roads | Kaggle / UofT | | | Yes (raster masks) | (record) |
| OSM extract — <region> | Geofabrik | | | Weak/proxy | ODbL |

## SpaceNet 3 per city
| City | Tiles | GSD | Bands avail. | Label files | Notes |
|---|---|---|---|---|---|
| Vegas | | | PS-RGB, PS-MS, MS, PAN | | |
| Paris | | | | | |
| Shanghai | | | | | |
| Khartoum | | | | | |

## File health
- Total files / total size:
- Unreadable:
- Orphan images (no label):
- Orphan labels (no image):
- CRS values observed:
- GSD values observed:

## Issues and blockers

## Storage
- Blob container:
- inventory.parquet:

## Licence flags
- SpaceNet CC BY-SA 4.0 — share-alike on derivatives
- OSM ODbL — share-alike on derived databases
- Both need a legal check before anything derived ships externally

## Next
→ AFTER_COLLECTION.md
8. Things that will cost you a day if you skip them
Test S3 access in the first hour. Not on day 2.
Download the sample before the full set. 728 MB tells you whether your assumptions hold.
Two tarballs per city. Imagery and labels are separate downloads.
Use screen/tmux. An SSH drop mid-download costs hours.
Checksum at ingest. A silently truncated tarball surfaces as a bizarre training bug three weeks later.
Fight GDAL on day 1, not day 3. Use conda-forge or a prebuilt image.
Actually look at 20 tiles. Imagery and labels overlaid. Ten minutes, and it catches problems no script will.
