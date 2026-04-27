

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
