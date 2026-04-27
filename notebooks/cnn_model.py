from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, f1_score, jaccard_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

# Building valid labelled pixel mask (unchanged)

valid_classes = np.array(list(lulc_classes.keys()))

mask = (
    np.isfinite(X_scaled).all(axis=2) &
    (lulc_arr > 0) &
    np.isin(lulc_arr.astype(int), valid_classes)
)

print("Valid labelled pixels:", int(mask.sum()))
print("Present classes:", np.unique(lulc_arr[mask]).astype(int))

if int(mask.sum()) == 0:
    raise Exception("No valid labelled pixels. Check AOI overlap and aligned LULC values.")

# Mapping LULC class ids -> contiguous model labels

class_ids     = sorted(np.unique(lulc_arr[mask]).astype(int))
class_to_idx  = {c: i for i, c in enumerate(class_ids)}
idx_to_class  = {i: c for c, i in class_to_idx.items()}
num_classes   = len(class_ids)

y_idx = np.full(lulc_arr.shape, -1, dtype=np.int32)
for c, i in class_to_idx.items():
    y_idx[lulc_arr == c] = i

print("Class mapping:", class_to_idx)
print("Number of classes:", num_classes)

# Extracting original patches + Merging with augmented patches

PATCH_SIZE = 11
pad = PATCH_SIZE // 2

X_pad = np.pad(X_scaled, ((pad,pad),(pad,pad),(0,0)), mode="reflect")

patches, labels, coords = [], [], []
rows, cols = np.where(mask)

for r, c in zip(rows, cols):
    rr, cc = r + pad, c + pad
    patch = X_pad[rr-pad:rr+pad+1, cc-pad:cc+pad+1, :]
    if patch.shape == (PATCH_SIZE, PATCH_SIZE, C):
        patches.append(patch)
        labels.append(y_idx[r, c])
        coords.append((r, c))

print(f"Original patches : {len(patches):,}")

# Reprojecting augmented patches from [0,1] min-max space
band_rng_safe   = np.where(band_rng > 0, band_rng, 1.0)
X_aug_raw       = X_aug_patches * band_rng_safe + band_min     # denormalise
aug_flat        = X_aug_raw.reshape(-1, C)
X_aug_scaled    = scaler.transform(aug_flat).reshape(X_aug_patches.shape).astype(np.float32)  # applying the StandardScaler

# Filtering augmented patches to only classes present in class_to_idx
valid_aug_mask  = np.isin(y_aug_patches, list(class_to_idx.keys()))
X_aug_scaled    = X_aug_scaled[valid_aug_mask]
y_aug_valid     = y_aug_patches[valid_aug_mask]
y_aug_indices   = np.array([class_to_idx[v] for v in y_aug_valid], dtype=np.int32)
aug_coords      = np.full((len(y_aug_indices), 2), -1, dtype=np.int32)

print(f"Augmented patches: {len(X_aug_scaled):,}  (filtered to known classes)")

# Concatenate
X_all  = np.concatenate([np.array(patches, dtype=np.float32), X_aug_scaled], axis=0)
y_all  = np.concatenate([np.array(labels,  dtype=np.int32),   y_aug_indices], axis=0)
coords = np.concatenate([np.array(coords),                     aug_coords],    axis=0)

print(f"Combined patch tensor : {X_all.shape}")
print(f"Combined label vector : {y_all.shape}")
print(f"Total memory          : {X_all.nbytes/1024**2:.1f} MB")

# Balanced per-class sampling with augmentation flag 

MAX_PER_CLASS = 10000   

keep = []
for cls in np.unique(y_all):
    idx = np.where(y_all == cls)[0]
    if len(idx) > MAX_PER_CLASS:
        idx = np.random.choice(idx, MAX_PER_CLASS, replace=False)
    keep.extend(idx.tolist())

keep  = np.array(sorted(keep))
X_all = X_all[keep]
y_all = y_all[keep]
coords = coords[keep]

print("After balanced sampling:")
for cls in np.unique(y_all):
    name = lulc_classes[idx_to_class[cls]][0]
    print(f"  Class {cls:2d} ({name:20s}): {int((y_all==cls).sum()):>5} samples")

# Train / Validation / Test split  (60 / 20 / 20)

X_train, X_temp, y_train, y_temp, c_train, c_temp = train_test_split(
    X_all, y_all, coords, test_size=0.40, random_state=42, stratify=y_all
)
X_val, X_test, y_val, y_test, c_val, c_test = train_test_split(
    X_temp, y_temp, c_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print(f"Train : {X_train.shape}  Val : {X_val.shape}  Test : {X_test.shape}")

# CNN architecture with BatchNorm + residual-style
# skip connection and class-weighted loss

from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight

def build_improved_cnn(input_shape, num_classes,
                       filters=(32, 64, 128),
                       dense_units=256,
                       dropout=0.4):
    """
    3-block CNN with:
      • Batch Normalisation after every Conv layer
      • Global Average Pooling instead of Flatten (fewer parameters, less overfitting)
      • Two Dense heads with Dropout
      • He-normal initialisation
    """
    inp = tf.keras.Input(shape=input_shape)

    # Block 1
    x = layers.Conv2D(filters[0], 3, padding="same",
                      kernel_initializer="he_normal")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(filters[0], 3, padding="same",
                      kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.SpatialDropout2D(0.1)(x)

    # Block 2 
    x = layers.Conv2D(filters[1], 3, padding="same",
                      kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(filters[1], 3, padding="same",
                      kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.SpatialDropout2D(0.15)(x)

    # Block 3 
    x = layers.Conv2D(filters[2], 3, padding="same",
                      kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    # Global Average Pooling + classifier 
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(dense_units, activation="relu",
                     kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(dense_units // 2, activation="relu",
                     kernel_initializer="he_normal")(x)
    x = layers.Dropout(dropout / 2)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)

    return tf.keras.Model(inputs=inp, outputs=out)


# Compute class weights to handle imbalance
cw_vals = compute_class_weight("balanced",
                                classes=np.unique(y_train),
                                y=y_train)
class_weight_dict = dict(enumerate(cw_vals))
print("Class weights:", {k: round(v, 3) for k, v in class_weight_dict.items()})

model_summary = build_improved_cnn(X_train.shape[1:], num_classes)
model_summary.summary()

