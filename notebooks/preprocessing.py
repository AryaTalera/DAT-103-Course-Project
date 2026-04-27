import os
import io
import json
import tarfile
import random
import requests
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds, reproject
from rasterio.enums import Resampling
from rasterio.transform import array_bounds

import pystac_client
import planetary_computer

np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

# USGS credentials and project settings
USERNAME = "YOUR_USERNAME"
APPLICATION_TOKEN ="YOUR_APPLICATION_TOKEN"

# Assigned AOI
bbox = {
    "min_lon": MIN_LON,
    "min_lat": MIN_LAT,
    "max_lon": MAX_LON,
    "max_lat": MAX_LAT
}

start_date = "start_date"
end_date = "end_date "
max_cloud = MAX_CLOUD

# USGS collection for Landsat 8/9 Collection 2 Level-2
dataset_name = "landsat_ot_c2_l2"
service_url = "https://m2m.cr.usgs.gov/api/api/json/stable/"

# USGS helper

def usgs_post(endpoint, payload=None, api_key=None):
    url = service_url + endpoint
    headers = {"Content-Type": "application/json"}
    if api_key is not None:
        headers["X-Auth-Token"] = api_key

    response = requests.post(url, headers=headers, json=payload or {})
    response.raise_for_status()

    out = response.json()
    if out.get("errorCode"):
        raise Exception(f"{out.get('errorCode')}: {out.get('errorMessage')}")
    return out.get("data")

# login with username + application token

login_payload = {
    "username": USERNAME,
    "token": APPLICATION_TOKEN
}

api_key = usgs_post("login-token", login_payload)
print("Logged in successfully.")

# searching Landsat scenes

search_payload = {
    "datasetName": dataset_name,
    "maxResults": 10,
    "startingNumber": 1,
    "sortDirection": "DESC",
    "sortField": "acquisitionDate",
    "sceneFilter": {
        "spatialFilter": {
            "filterType": "mbr",
            "lowerLeft": {
                "latitude": bbox["min_lat"],
                "longitude": bbox["min_lon"]
            },
            "upperRight": {
                "latitude": bbox["max_lat"],
                "longitude": bbox["max_lon"]
            }
        },
        "acquisitionFilter": {
            "start": start_date,
            "end": end_date
        },
        "cloudCoverFilter": {
            "min": 0,
            "max": max_cloud,
            "includeUnknown": False
        }
    }
}

search_data = usgs_post("scene-search", search_payload, api_key=api_key)
results = search_data.get("results", [])

print("Scenes found:", len(results))
for i, s in enumerate(results[:5]):
    print(i, s.get("displayId"), "|", s.get("acquisitionDate"), "| cloud:", s.get("cloudCover"))

# Cell 7: choosing first scene

if not results:
    raise Exception("No scenes found for the given AOI/date/cloud settings.")

scene = results[0]
entity_id = scene["entityId"]
display_id = scene["displayId"]

print("Using scene:", display_id)

# getting download options

download_options = usgs_post(
    "download-options",
    {
        "datasetName": dataset_name,
        "entityIds": [entity_id]
    },
    api_key=api_key
)

available_products = [x for x in download_options if x.get("available")]
print("Available products:", len(available_products))
for opt in available_products[:10]:
    print(opt.get("productName"), "| id:", opt.get("id"))

# requesting download URL for first available product

if not available_products:
    raise Exception("No available downloadable products for this scene.")

selected_product = available_products[0]
product_id = selected_product["id"]

download_req = usgs_post(
    "download-request",
    {
        "downloads": [
            {
                "entityId": entity_id,
                "productId": product_id
            }
        ],
        "label": "landsat_job"
    },
    api_key=api_key
)

download_url = None
available_downloads = download_req.get("availableDownloads", [])
if available_downloads:
    download_url = available_downloads[0].get("url")

if download_url is None:
    raise Exception(
        "No immediate download URL returned. Product may still be preparing. "
        "Re-run this cell after a short wait if needed."
    )

print("Download URL found.")

# downloading and extracting Landsat archive

os.makedirs("usgs_download", exist_ok=True)
archive_path = os.path.join("usgs_download", f"{display_id}.tar")

with requests.get(download_url, stream=True) as r:
    r.raise_for_status()
    with open(archive_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

extract_dir = os.path.join("usgs_download", display_id)
os.makedirs(extract_dir, exist_ok=True)

with tarfile.open(archive_path) as tar:
    tar.extractall(extract_dir)

print("Archive:", archive_path)
print("Extracted to:", extract_dir)
print("Example files:", os.listdir(extract_dir)[:15])

# locating required Landsat bands

def find_band_file(folder, suffix):
    for f in os.listdir(folder):
        if f.endswith(suffix):
            return os.path.join(folder, f)
    return None

blue_path = find_band_file(extract_dir, "_SR_B2.TIF")
green_path = find_band_file(extract_dir, "_SR_B3.TIF")
red_path = find_band_file(extract_dir, "_SR_B4.TIF")
nir_path = find_band_file(extract_dir, "_SR_B5.TIF")

print("Blue :", blue_path)
print("Green:", green_path)
print("Red  :", red_path)
print("NIR  :", nir_path)

if not all([blue_path, green_path, red_path, nir_path]):
    raise Exception("One or more required band files were not found.")

# helper to crop Landsat bands to AOI

def read_crop_band_with_meta(path, bbox_lonlat):
    with rasterio.open(path) as src:
        if str(src.crs) != "EPSG:4326":
            left, bottom, right, top = transform_bounds(
                "EPSG:4326",
                src.crs,
                bbox_lonlat["min_lon"],
                bbox_lonlat["min_lat"],
                bbox_lonlat["max_lon"],
                bbox_lonlat["max_lat"]
            )
        else:
            left, bottom, right, top = (
                bbox_lonlat["min_lon"],
                bbox_lonlat["min_lat"],
                bbox_lonlat["max_lon"],
                bbox_lonlat["max_lat"]
            )

        left = max(left, src.bounds.left)
        right = min(right, src.bounds.right)
        bottom = max(bottom, src.bounds.bottom)
        top = min(top, src.bounds.top)

        if left >= right or bottom >= top:
            raise Exception("AOI does not overlap raster after CRS conversion.")

        window = from_bounds(left, bottom, right, top, src.transform)
        window = window.round_offsets().round_lengths()

        data = src.read(1, window=window).astype(np.float32)
        win_transform = src.window_transform(window)

        if data.size == 0:
            raise Exception("Cropped Landsat window is empty.")

        return data, win_transform, src.crs

# reading full cropped Landsat bands

blue_full, landsat_transform_full, red_crs = read_crop_band_with_meta(blue_path, bbox)
green_full, _, _ = read_crop_band_with_meta(green_path, bbox)
red_full, _, _ = read_crop_band_with_meta(red_path, bbox)
nir_full, _, _ = read_crop_band_with_meta(nir_path, bbox)

print("Full cropped shapes:", blue_full.shape, green_full.shape, red_full.shape, nir_full.shape)
print("Landsat CRS:", red_crs)

# computing valid inner window to remove black border later

valid_mask_full = (blue_full > 0) & (green_full > 0) & (red_full > 0)

rows = np.where(np.any(valid_mask_full, axis=1))[0]
cols = np.where(np.any(valid_mask_full, axis=0))[0]

if len(rows) == 0 or len(cols) == 0:
    raise Exception("No valid pixels found in Landsat crop.")

r0, r1 = rows[0], rows[-1]
c0, c1 = cols[0], cols[-1]

print("Trim indices:", r0, r1, c0, c1)

# trimming Landsat bands to clean valid area

blue = blue_full[r0:r1+1, c0:c1+1]
green = green_full[r0:r1+1, c0:c1+1]
red = red_full[r0:r1+1, c0:c1+1]
nir = nir_full[r0:r1+1, c0:c1+1]

landsat_transform_trim = landsat_transform_full * rasterio.Affine.translation(c0, r0)

print("Trimmed shapes:", blue.shape, green.shape, red.shape, nir.shape)

# computing clean bbox from trimmed Landsat area

bottom, left, top, right = array_bounds(red.shape[0], red.shape[1], landsat_transform_trim)

if str(red_crs) != "EPSG:4326":
    min_lon, min_lat, max_lon, max_lat = transform_bounds(red_crs, "EPSG:4326", left, bottom, right, top)
else:
    min_lon, min_lat, max_lon, max_lat = left, bottom, right, top

bbox_clean = {
    "min_lon": min_lon,
    "min_lat": min_lat,
    "max_lon": max_lon,
    "max_lat": max_lat
}

print("bbox_clean:", bbox_clean)

# searching for ESRI LULC item from Planetary Computer

catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace
)

search = catalog.search(
    collections=["io-lulc-annual-v02"],
    intersects={
        "type": "Polygon",
        "coordinates": [[
            [bbox["min_lon"], bbox["min_lat"]],
            [bbox["max_lon"], bbox["min_lat"]],
            [bbox["max_lon"], bbox["max_lat"]],
            [bbox["min_lon"], bbox["max_lat"]],
            [bbox["min_lon"], bbox["min_lat"]]
        ]]
    },
    datetime="2023-01-01/2023-12-31"
)

lulc_items = list(search.items())
print("LULC items found:", len(lulc_items))
for item in lulc_items[:5]:
    print(item.id)

if not lulc_items:
    raise Exception("No ESRI LULC item found for the AOI.")

# choosing first LULC item and inspecting asset

lulc_item = lulc_items[0]
lulc_href = lulc_item.assets["data"].href

print("Selected LULC item:", lulc_item.id)
print("LULC asset:", lulc_href)

# ESRI LULC class names and colors

lulc_classes = {
    1: ("Water", "#419BDF"),
    2: ("Trees", "#397D49"),
    4: ("Flooded Vegetation", "#7A87C6"),
    5: ("Crops", "#E49635"),
    7: ("Built Area", "#C4281B"),
    8: ("Bare Ground", "#A59B8F"),
    9: ("Snow/Ice", "#B39FE1"),
    10: ("Clouds", "#E3E2C3"),
    11: ("Rangeland", "#88B053")
}

# computing NDVI from trimmed Landsat bands

ndvi = (nir - red) / (nir + red + 1e-6)

print("NDVI range:", float(np.nanmin(ndvi)), float(np.nanmax(ndvi)))

# aligning ESRI LULC to FULL Landsat grid using rasterio.warp.reproject

landsat_h, landsat_w = red_full.shape
lulc_full = np.zeros((landsat_h, landsat_w), dtype=np.float32)

reproject(
    source=lulc_raw,
    destination=lulc_full,
    src_transform=lulc_transform,
    src_crs=lulc_crs,
    src_nodata=0,
    dst_transform=landsat_transform_full,
    dst_crs=red_crs,
    dst_nodata=0,
    resampling=Resampling.nearest
)

print("Aligned full LULC shape:", lulc_full.shape)
print("Aligned full LULC unique values:", np.unique(lulc_full)[:20])

# trimming LULC with the SAME trim indices used for Landsat

lulc_arr = lulc_full[r0:r1+1, c0:c1+1]

print("Trimmed LULC shape:", lulc_arr.shape)
print("Trimmed LULC unique values:", np.unique(lulc_arr)[:20])

# Memory-Efficient Data Augmentation
# Generates 500 augmented image+label pairs but extracts patches
# Per-class with a hard cap

import numpy as np
np.random.seed(42)

# Stack base bands (H, W, 5)
X_raw = np.stack([blue, green, red, nir, ndvi], axis=-1).astype(np.float32)
y_raw = lulc_arr.copy()
H, W, C = X_raw.shape
print("Base image shape:", X_raw.shape)

# Per-band min-max normalisation 
band_min = X_raw.reshape(-1, C).min(axis=0)
band_max = X_raw.reshape(-1, C).max(axis=0)
band_rng  = np.where(band_max - band_min > 0, band_max - band_min, 1.0)
def _norm(img):  return (img - band_min) / band_rng

X_norm = _norm(X_raw)

# Augmentation functions

# Mirrors the image left-right
def aug_hflip(img, lbl):                                       
    return np.fliplr(img).copy(), np.fliplr(lbl).copy()

# Mirrors the image top-to-bottom
def aug_vflip(img, lbl):      
    return np.flipud(img).copy(), np.flipud(lbl).copy()

# Rotates the image by 90°, 180°, 270°
def aug_rot90_1(img, lbl):    
    return np.rot90(img,k=1,axes=(0,1)).copy(), np.rot90(lbl,k=1).copy()

def aug_rot90_2(img, lbl):    
    return np.rot90(img,k=2,axes=(0,1)).copy(), np.rot90(lbl,k=2).copy()

def aug_rot90_3(img, lbl):    
    return np.rot90(img,k=3,axes=(0,1)).copy(), np.rot90(lbl,k=3).copy()

# Adds Gaussian noise with a random sigma between 0.005 and 0.03
def aug_noise(img, lbl):
    return np.clip(img + np.random.normal(0, np.random.uniform(0.005,0.03), img.shape).astype(np.float32), 0, 1), lbl.copy()

# Shifts all band values up or down by a constant (±15%)
def aug_brightness(img, lbl): 
    return np.clip(img + np.random.uniform(-0.15,0.15), 0, 1), lbl.copy()

# Stretches or compresses pixel values around the image mean by factor 0.7–1.4
def aug_contrast(img, lbl):
    f = np.random.uniform(0.7,1.4); m = img.mean(axis=(0,1),keepdims=True)
    return np.clip((img-m)*f+m, 0, 1), lbl.copy()

# Applies a non-linear power transform pixel^gamma to each band independently with a random gamma between 0.6 and 1.6 
# The clip(1e-6,1) prevents log(0) errors in the power operation
def aug_gamma(img, lbl):
    out = img.copy()
    for b in range(img.shape[-1]):
        out[...,b] = np.power(np.clip(img[...,b],1e-6,1), np.random.uniform(0.6,1.6))
    return out, lbl.copy()

# Randomly zeros out one entire band
def aug_band_dropout(img, lbl):
    out = img.copy(); out[..., np.random.randint(0, img.shape[-1])] = 0.0
    return out, lbl.copy()

# Horizontal flip followed by Gaussian noise
def aug_flip_noise(img, lbl):
    img2, lbl2 = aug_hflip(img, lbl); return aug_noise(img2, lbl2)

# Random 90/180/270° rotation followed by brightness shift
def aug_rot_brightness(img, lbl):
    k = np.random.choice([1,2,3])
    img2, lbl2 = np.rot90(img,k=k,axes=(0,1)).copy(), np.rot90(lbl,k=k).copy()
    return aug_brightness(img2, lbl2)

TRANSFORMS = [
    aug_hflip, aug_vflip, aug_rot90_1, aug_rot90_2, aug_rot90_3,
    aug_noise, aug_brightness, aug_contrast,
    aug_gamma, aug_band_dropout, aug_flip_noise, aug_rot_brightness,
]

# With TARGET_AUGMENTED = 500 scenes and si % len(TRANSFORMS), each of the 12 transforms is applied roughly equally (~41–42 times each). 
# This ensures no single type of augmentation dominates. 
TARGET_AUGMENTED   = 500   # number of full augmented scenes
PATCH_SIZE_AUG     = 11
pad_a              = PATCH_SIZE_AUG // 2
MAX_PX_PER_CLASS_PER_SCENE = 150   # hard cap 

print(f"Generating {TARGET_AUGMENTED} augmented scenes ...")
print(f"  Patch size: {PATCH_SIZE_AUG}x{PATCH_SIZE_AUG}, cap: {MAX_PX_PER_CLASS_PER_SCENE} px/class/scene")

valid_cls_ids = np.array(list(lulc_classes.keys()))

aug_X_patches, aug_y_labels = [], []

for si in range(TARGET_AUGMENTED):
    fn = TRANSFORMS[si % len(TRANSFORMS)]
    img_a, lbl_a = fn(X_norm, y_raw)

    img_pad  = np.pad(img_a, ((pad_a,pad_a),(pad_a,pad_a),(0,0)), mode="reflect")
    finite_a = np.isfinite(img_a).all(axis=-1)

    for cls_val in valid_cls_ids:
        cls_mask = (lbl_a == cls_val) & finite_a
        r_cls, c_cls = np.where(cls_mask)
        if len(r_cls) == 0:
            continue
        n_pick = min(len(r_cls), MAX_PX_PER_CLASS_PER_SCENE)
        sel = np.random.choice(len(r_cls), n_pick, replace=False)
        for idx in sel:
            r, c = r_cls[idx], c_cls[idx]
            rr, cc = r + pad_a, c + pad_a
            patch = img_pad[rr-pad_a:rr+pad_a+1, cc-pad_a:cc+pad_a+1, :]
            if patch.shape == (PATCH_SIZE_AUG, PATCH_SIZE_AUG, C):
                aug_X_patches.append(patch.astype(np.float32))
                aug_y_labels.append(int(cls_val))

    if (si + 1) % 100 == 0:
        print(f"  ... {si+1}/{TARGET_AUGMENTED} scenes | patches so far: {len(aug_X_patches):,}")

X_aug_patches = np.array(aug_X_patches, dtype=np.float32)
y_aug_patches = np.array(aug_y_labels,  dtype=np.int32)

print(f"\nAugmented patch tensor : {X_aug_patches.shape}")
print(f"Augmented label vector : {y_aug_patches.shape}")
print(f"Memory usage           : {X_aug_patches.nbytes/1024**2:.1f} MB")

# Visualise 20 random augmented thumbnails
sample_scenes = []
for si in range(20):
    fn = TRANSFORMS[si % len(TRANSFORMS)]
    img_a, _ = fn(X_norm, y_raw)
    sample_scenes.append(img_a)

fig, axes = plt.subplots(4, 5, figsize=(14, 11))
for ax, img_a in zip(axes.ravel(), sample_scenes):
    rgb = np.clip(img_a[..., [2,1,0]], 0, 1)
    ax.imshow(rgb)
    ax.axis("off")
plt.suptitle(f"20 Sample Augmented Scenes — {TARGET_AUGMENTED} total generated",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.show()

print("X_aug_patches and y_aug_patches ready.")

# Landsat + NDVI input stack

X_stack = np.dstack([
    blue.astype(np.float32),
    green.astype(np.float32),
    red.astype(np.float32),
    nir.astype(np.float32),
    ndvi.astype(np.float32)
])

print("X_stack shape:", X_stack.shape)

# building valid labelled pixel mask

valid_classes = np.array(list(lulc_classes.keys()))

mask = (
    np.isfinite(X_stack).all(axis=2) &
    (lulc_arr > 0) &
    np.isin(lulc_arr.astype(int), valid_classes)
)

print("Valid labelled pixels:", int(mask.sum()))
print("Present classes:", np.unique(lulc_arr[mask]).astype(int))

# Debug if valid labelled pixels are 0

if int(mask.sum()) == 0:
    print("Finite X_stack pixels:", int(np.isfinite(X_stack).all(axis=2).sum()))
    print("LULC > 0 pixels     :", int((lulc_arr > 0).sum()))
    print("Valid class pixels   :", int(np.isin(lulc_arr.astype(int), valid_classes).sum()))
    print("Unique LULC values   :", np.unique(lulc_arr)[:50])
    raise Exception("No valid labelled pixels. Check AOI overlap and aligned LULC values.")

# mapping original LULC class ids to contiguous model labels

class_ids = sorted(np.unique(lulc_arr[mask]).astype(int))
class_to_idx = {c: i for i, c in enumerate(class_ids)}
idx_to_class = {i: c for c, i in class_to_idx.items()}

y_idx = np.full(lulc_arr.shape, -1, dtype=np.int32)
for c, i in class_to_idx.items():
    y_idx[lulc_arr == c] = i

print("Class mapping:", class_to_idx)

# Normalizing the input stack band-wise (zero-mean, unit-std)

from sklearn.preprocessing import StandardScaler

H, W, C = X_stack.shape
X_flat = X_stack.reshape(-1, C)
scaler = StandardScaler()
X_scaled_flat = scaler.fit_transform(X_flat)
X_scaled = X_scaled_flat.reshape(H, W, C).astype(np.float32)

print("X_scaled shape:", X_scaled.shape)
print("Band means (should be ~0):", X_scaled[np.isfinite(X_scaled).all(axis=2)].mean(axis=0).round(3))
