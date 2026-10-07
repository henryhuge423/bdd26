"""PanNuke constants shared across data, training and evaluation code."""

# Channel order of PanNuke masks.npy (channel 5 is background).
# Our type maps use 0 = background and 1..5 = these classes in this order.
CLASS_NAMES = ["Neoplastic", "Inflammatory", "Connective", "Dead", "Epithelial"]
NUM_CLASSES = len(CLASS_NAMES)

# Type-map id of Dead (4 of 1..5); the single source for the dead-specialist branch.
DEAD_TYPE = CLASS_NAMES.index("Dead") + 1

# Tissue order used by the official PanNuke-metrics script.
TISSUES = [
    "Adrenal_gland", "Bile-duct", "Bladder", "Breast", "Cervix", "Colon", "Esophagus",
    "HeadNeck", "Kidney", "Liver", "Lung", "Ovarian", "Pancreatic", "Prostate", "Skin",
    "Stomach", "Testis", "Thyroid", "Uterus",
]

# Official 3-split protocol (Warwick PanNuke page): (train, val, test) fold ids.
SPLITS = {
    1: (1, 2, 3),
    2: (2, 1, 3),
    3: (3, 2, 1),
}

PATCH_SIZE = 256
MPP = 0.25  # 40x
