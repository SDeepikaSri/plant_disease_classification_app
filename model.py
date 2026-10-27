import torch
import torch.nn as nn
from torchvision import models


CLASS_NAMES = [
    "Apple_Scab_Leaf",
    "Apple_leaf",
    "Apple_rust_leaf",
    "Bell_pepper_leaf",
    "Bell_pepper_leaf_spot",
    "Blueberry_leaf",
    "Cherry_leaf",
    "Corn_Gray_leaf_spot",
    "Corn_leaf_blight",
    "Corn_rust_leaf",
    "Peach_leaf",
    "Potato_leaf_early_blight",
    "Potato_leaf_late_blight",
    "Raspberry_leaf",
    "Soyabean_leaf",
    "Squash_Powdery_mildew_leaf",
    "Strawberry_leaf",
    "Tomato_Early_blight_leaf",
    "Tomato_Septoria_leaf_spot",
    "Tomato_leaf",
    "Tomato_leaf_bacterial_spot",
    "Tomato_leaf_late_blight",
    "Tomato_leaf_mosaic_virus",
    "Tomato_leaf_yellow_virus",
    "Tomato_mold_leaf",
    "grape_leaf",
    "grape_leaf_black_rot",
]


NUM_CLASSES = 27


CLASS_TO_SD = {
    "Apple_Scab_Leaf": ("Apple", "scab"),
    "Apple_leaf": ("Apple", "healthy"),
    "Apple_rust_leaf": ("Apple", "rust"),

    "Bell_pepper_leaf": ("Bell_pepper", "healthy"),
    "Bell_pepper_leaf_spot": ("Bell_pepper", "leaf_spot"),

    "Blueberry_leaf": ("Blueberry", "healthy"),
    "Cherry_leaf": ("Cherry", "healthy"),

    "Corn_Gray_leaf_spot": ("Corn", "gray_leaf_spot"),
    "Corn_leaf_blight": ("Corn", "leaf_blight"),
    "Corn_rust_leaf": ("Corn", "rust"),

    "Peach_leaf": ("Peach", "healthy"),

    "Potato_leaf_early_blight": ("Potato", "early_blight"),
    "Potato_leaf_late_blight": ("Potato", "late_blight"),

    "Raspberry_leaf": ("Raspberry", "healthy"),
    "Soyabean_leaf": ("Soyabean", "healthy"),

    "Squash_Powdery_mildew_leaf": ("Squash", "powdery_mildew"),

    "Strawberry_leaf": ("Strawberry", "healthy"),

    "Tomato_Early_blight_leaf": ("Tomato", "early_blight"),
    "Tomato_Septoria_leaf_spot": ("Tomato", "septoria_leaf_spot"),
    "Tomato_leaf": ("Tomato", "healthy"),
    "Tomato_leaf_bacterial_spot": ("Tomato", "bacterial_spot"),
    "Tomato_leaf_late_blight": ("Tomato", "late_blight"),
    "Tomato_leaf_mosaic_virus": ("Tomato", "mosaic_virus"),
    "Tomato_leaf_yellow_virus": ("Tomato", "yellow_virus"),
    "Tomato_mold_leaf": ("Tomato", "mold"),

    "grape_leaf": ("grape", "healthy"),
    "grape_leaf_black_rot": ("grape", "black_rot"),
}


def build_model(num_classes=NUM_CLASSES):

    model = models.efficientnet_b4(
        weights=None
    )

    in_features = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(
        in_features,
        num_classes
    )

    return model


def load_model_from_checkpoint(checkpoint):

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Checkpoint must be a dictionary."
        )

    num_classes = checkpoint.get(
        "num_classes",
        NUM_CLASSES
    )

    model = build_model(
        num_classes
    )

    state_dict = checkpoint[
        "model_state_dict"
    ]

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):

            key = key[7:]

        cleaned_state_dict[key] = value

    model.load_state_dict(
        cleaned_state_dict,
        strict=True
    )

    return model