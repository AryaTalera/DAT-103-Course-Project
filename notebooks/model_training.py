import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, f1_score, jaccard_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from cnn_model import build_improved_cnn

# Hyperparameter tuning on validation set

trials = [
    {"filters": (32,  64,  128), "dense_units": 256, "dropout": 0.20, "lr": 1e-3},
    {"filters": (64,  128, 256), "dense_units": 256, "dropout": 0.20, "lr": 5e-4},
    {"filters": (32,  64,  128), "dense_units": 128, "dropout": 0.10, "lr": 1e-3},
    {"filters": (64,  128, 256), "dense_units": 512, "dropout": 0.25, "lr": 1e-3},
]

best_model    = None
best_val_acc  = -1
best_trial    = None
trial_results = []

for t_idx, trial in enumerate(trials):
    print(f"\n{'='*55}")
    print(f"Trial {t_idx+1}/{len(trials)}: {trial}")
    print(f"{'='*55}")

    model = build_improved_cnn(
        input_shape  = X_train.shape[1:],
        num_classes  = num_classes,
        filters      = trial["filters"],
        dense_units  = trial["dense_units"],
        dropout      = trial["dropout"]
    )

    model.compile(
        optimizer = tf.keras.optimizers.Adam(
                        learning_rate=trial["lr"],
                        weight_decay=1e-4),   # L2 regularisation via AdamW
        loss      = "sparse_categorical_crossentropy",
        metrics   = ["accuracy"]
    )

    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=5,
                      restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=3, min_lr=1e-6, verbose=1),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data = (X_val, y_val),
        epochs          = 50,
        batch_size      = 64,
        class_weight    = class_weight_dict,
        callbacks       = callbacks,
        verbose         = 1
    )

    val_acc = max(history.history["val_accuracy"])
    trial_results.append({"trial": trial, "val_acc": val_acc})
    print(f"  → Best val accuracy: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc  = val_acc
        best_model    = model
        best_trial    = trial
        best_history  = history

print("\n" + "="*55)
print("HYPERPARAMETER TUNING SUMMARY")
print("="*55)
for r in trial_results:
    print(f"  {r['trial']}  →  val_acc={r['val_acc']:.4f}")
print(f"\nBest trial : {best_trial}")
print(f"Best val acc: {best_val_acc:.4f}")

# Evaluating best model on test set

test_loss, test_acc = best_model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss    : {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

y_prob = best_model.predict(X_test, verbose=0)
y_pred = np.argmax(y_prob, axis=1)

# Confusion matrix (normalised + raw counts)

import itertools

target_names = [lulc_classes[idx_to_class[i]][0] for i in range(num_classes)]
cm = confusion_matrix(y_test, y_pred)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for ax, data, title, fmt in zip(
    axes,
    [cm, cm_norm],
    ["Confusion Matrix (counts)", "Confusion Matrix (normalised)"],
    [True, False]
):
    im = ax.imshow(data, cmap="Blues")
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(target_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(target_names, fontsize=8)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("True", fontsize=10)
    ax.set_title(title, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046)
    thresh = data.max() / 2.0
    for i, j in itertools.product(range(data.shape[0]), range(data.shape[1])):
        val = f"{data[i,j]:.2f}" if not fmt else f"{data[i,j]}"
        ax.text(j, i, val, ha="center", va="center",
                color="white" if data[i, j] > thresh else "black", fontsize=7)

plt.tight_layout()
plt.show()

# Per-class F1, IoU + macro/weighted averages

from sklearn.metrics import f1_score, jaccard_score, classification_report

f1_per   = f1_score(y_test, y_pred, average=None,        zero_division=0)
iou_per  = jaccard_score(y_test, y_pred, average=None,   zero_division=0)

f1_mac   = f1_score(y_test, y_pred, average="macro",     zero_division=0)
f1_wt    = f1_score(y_test, y_pred, average="weighted",  zero_division=0)
iou_mac  = jaccard_score(y_test, y_pred, average="macro",    zero_division=0)
iou_wt   = jaccard_score(y_test, y_pred, average="weighted", zero_division=0)

print(f"\n{'Class':<22} {'F1':>8} {'IoU':>8}")
print("-" * 40)
for i, name in enumerate(target_names):
    print(f"  {name:<20} {f1_per[i]:>8.3f} {iou_per[i]:>8.3f}")
print("-" * 40)
print(f"  {'Macro avg':<20} {f1_mac:>8.3f} {iou_mac:>8.3f}")
print(f"  {'Weighted avg':<20} {f1_wt:>8.3f} {iou_wt:>8.3f}")

# Bar chart
x = np.arange(num_classes)
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(x - 0.2, f1_per,  0.4, label="F1",  color="steelblue")
ax.bar(x + 0.2, iou_per, 0.4, label="IoU", color="coral")
ax.set_xticks(x)
ax.set_xticklabels(target_names, rotation=30, ha="right", fontsize=8)
ax.set_ylim(0, 1)
ax.set_title("Per-class F1 and IoU", fontweight="bold")
ax.axhline(f1_mac,  color="steelblue", linestyle="--", linewidth=1, label=f"F1 macro={f1_mac:.3f}")
ax.axhline(iou_mac, color="coral",     linestyle="--", linewidth=1, label=f"IoU macro={iou_mac:.3f}")
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))

# ROC curves (one-vs-rest) with AUC per class

from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import matplotlib.cm as cm_module

y_test_bin = label_binarize(y_test, classes=np.arange(num_classes))
colors = cm_module.tab10(np.linspace(0, 1, num_classes))

fig, ax = plt.subplots(figsize=(9, 7))
for i in range(num_classes):
    if y_test_bin[:, i].sum() == 0:
        continue
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=colors[i], linewidth=1.8,
            label=f"{target_names[i]} (AUC={roc_auc:.3f})")

ax.plot([0,1],[0,1], "k--", linewidth=1)
ax.set_xlabel("False Positive Rate", fontsize=11)
ax.set_ylabel("True Positive Rate",  fontsize=11)
ax.set_title("ROC Curves – One-vs-Rest",  fontsize=13, fontweight="bold")
ax.legend(fontsize=8, loc="lower right")
plt.tight_layout()
plt.show()
