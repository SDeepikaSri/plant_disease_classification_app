import base64
import io
import os
import time
import threading
import zipfile

from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

from PIL import Image, ImageOps
from torchvision import transforms

from matplotlib import colormaps

from scipy.ndimage import (
    binary_closing,
    binary_fill_holes,
    gaussian_filter,
    label
)

from skimage.measure import find_contours

import model as M


# ============================================================
# PATHS
# ============================================================

BASE = Path(__file__).resolve().parent

MODEL_PATH = BASE / "best_model"

INDEX_PATH = BASE / "index.html"


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# CONFIG
# ============================================================

DEFAULT_IMG_SIZE = 380

MAX_BYTES = 15 * 1024 * 1024

SEED = 42

GRADCAM_ALPHA = 0.45

GRADCAM_THRESHOLD_PERCENTILE = 82

CONTOUR_LINEWIDTH = 2.0

CONTOUR_ALPHA = 0.95

MIN_REGION_AREA = 100


# ============================================================
# RANDOM SEED
# ============================================================

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# MODEL STATE
# ============================================================

state = {
    "model": None,
    "classes": M.CLASS_NAMES,
    "num_classes": M.NUM_CLASSES,
    "img_size": DEFAULT_IMG_SIZE,
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
    "best_accuracy": None,
}


# Grad-CAM uses gradients.
# Requests are therefore serialized.
lock = threading.Lock()


# ============================================================
# CHECKPOINT LOADER
# ============================================================

def load_checkpoint(path):
    """
    Loads either:

    1. A normal .pt/.pth checkpoint
    2. An extracted PyTorch checkpoint folder

    The user's best_model folder contains:
        data.pkl
        data/
        .data/
        byteorder
        version
        etc.
    """

    path = str(path)

    if os.path.isdir(path):

        buffer = io.BytesIO()

        root_name = os.path.basename(
            os.path.normpath(path)
        )

        with zipfile.ZipFile(
            buffer,
            "w",
            zipfile.ZIP_STORED
        ) as archive:

            for root, _, files in os.walk(path):

                for filename in files:

                    full_path = os.path.join(
                        root,
                        filename
                    )

                    relative_path = os.path.relpath(
                        full_path,
                        path
                    ).replace(os.sep, "/")

                    archive.write(
                        full_path,
                        f"{root_name}/{relative_path}"
                    )

        buffer.seek(0)

        return torch.load(
            buffer,
            map_location="cpu",
            weights_only=False
        )

    return torch.load(
        path,
        map_location="cpu",
        weights_only=False
    )


# ============================================================
# PRETTY CLASS NAME
# ============================================================

def pretty_class(name):

    name = name.replace(
        "___",
        " "
    )

    name = name.replace(
        "_",
        " "
    )

    return name.strip()


# ============================================================
# BASE64 IMAGE
# ============================================================

def image_to_base64(image_array):

    image_array = np.clip(
        image_array,
        0,
        255
    ).astype(np.uint8)

    buffer = io.BytesIO()

    Image.fromarray(
        image_array
    ).save(
        buffer,
        format="PNG"
    )

    return (
        "data:image/png;base64,"
        + base64.b64encode(
            buffer.getvalue()
        ).decode()
    )


# ============================================================
# MODEL STARTUP
# ============================================================

@asynccontextmanager
async def lifespan(app):

    if not MODEL_PATH.exists():

        raise RuntimeError(
            f"\nCheckpoint not found:\n"
            f"{MODEL_PATH}\n\n"
            f"Make sure the 'best_model' folder is "
            f"inside the same directory as app.py."
        )

    print("=" * 70)
    print("LOADING PLANT DISEASE MODEL")
    print("=" * 70)

    print("Checkpoint:", MODEL_PATH)
    print("Device:", DEVICE)

    checkpoint = load_checkpoint(
        MODEL_PATH
    )

    print(
        "Checkpoint keys:",
        list(checkpoint.keys())
    )

    num_classes = checkpoint.get(
        "num_classes",
        M.NUM_CLASSES
    )

    img_size = checkpoint.get(
        "img_size",
        DEFAULT_IMG_SIZE
    )

    mean = checkpoint.get(
        "mean",
        [0.485, 0.456, 0.406]
    )

    std = checkpoint.get(
        "std",
        [0.229, 0.224, 0.225]
    )

    classes = checkpoint.get(
        "classes",
        M.CLASS_NAMES
    )

    best_accuracy = checkpoint.get(
        "best_val_accuracy"
    )

    print("Classes:", num_classes)
    print("Image size:", img_size)

    if best_accuracy is not None:

        print(
            f"Best validation accuracy: "
            f"{best_accuracy * 100:.2f}%"
        )

    # --------------------------------------------------------
    # BUILD MODEL
    # --------------------------------------------------------

    net = M.load_model_from_checkpoint(
        checkpoint
    )

    net = net.to(DEVICE)

    net.eval()

    # --------------------------------------------------------
    # SAVE STATE
    # --------------------------------------------------------

    state["model"] = net
    state["classes"] = classes
    state["num_classes"] = num_classes
    state["img_size"] = img_size
    state["mean"] = mean
    state["std"] = std
    state["best_accuracy"] = best_accuracy

    print("=" * 70)
    print("MODEL LOADED SUCCESSFULLY")
    print("=" * 70)

    yield


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Plant Disease Detection",
    description="PlantDoc 27-class disease detection with Grad-CAM",
    lifespan=lifespan
)


# ============================================================
# PREPROCESSING
# ============================================================

def get_transform():

    return transforms.Compose([
        transforms.Resize(
            (
                state["img_size"],
                state["img_size"]
            )
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=state["mean"],
            std=state["std"]
        )
    ])


# ============================================================
# GRAD-CAM
# ============================================================

class GradCAM:

    def __init__(self, model, target_layer):

        self.model = model

        self.target_layer = target_layer

        self.activations = None

        self.gradients = None

        self.forward_handle = (
            target_layer.register_forward_hook(
                self._forward_hook
            )
        )

        self.backward_handle = (
            target_layer.register_full_backward_hook(
                self._backward_hook
            )
        )

    def _forward_hook(
        self,
        module,
        inputs,
        output
    ):

        self.activations = output

    def _backward_hook(
        self,
        module,
        grad_input,
        grad_output
    ):

        self.gradients = grad_output[0]

    def generate(
        self,
        image_tensor,
        target_class
    ):

        self.model.zero_grad(
            set_to_none=True
        )

        output = self.model(
            image_tensor
        )

        score = output[
            0,
            target_class
        ]

        score.backward()

        activations = self.activations

        gradients = self.gradients

        # Global average pooling of gradients
        weights = gradients.mean(
            dim=(2, 3),
            keepdim=True
        )

        cam = (
            weights * activations
        ).sum(
            dim=1,
            keepdim=True
        )

        cam = F.relu(cam)

        cam = F.interpolate(
            cam,
            size=(
                state["img_size"],
                state["img_size"]
            ),
            mode="bilinear",
            align_corners=False
        )

        cam = cam[
            0,
            0
        ].detach().cpu().numpy()

        # Normalize
        cam_min = cam.min()
        cam_max = cam.max()

        cam = (
            cam - cam_min
        ) / (
            cam_max - cam_min + 1e-8
        )

        return cam

    def close(self):

        self.forward_handle.remove()

        self.backward_handle.remove()


# ============================================================
# GRAD-CAM TARGET LAYER
# ============================================================

def create_gradcam():

    model = state["model"]

    target_layer = model.features[-1]

    return GradCAM(
        model,
        target_layer
    )


# ============================================================
# DISEASE REGION EXTRACTION
# ============================================================

def create_disease_regions(
    original,
    heatmap
):

    # Ignore very weak activation.
    positive = heatmap[
        heatmap > 0
    ]

    if positive.size == 0:

        return np.zeros_like(
            heatmap,
            dtype=bool
        )

    threshold = np.percentile(
        positive,
        GRADCAM_THRESHOLD_PERCENTILE
    )

    mask = heatmap >= threshold

    # Smooth the activation region.
    mask = gaussian_filter(
        mask.astype(np.float32),
        sigma=2.0
    ) > 0.35

    # Join nearby regions.
    mask = binary_closing(
        mask,
        structure=np.ones(
            (7, 7),
            dtype=bool
        )
    )

    # Fill small holes.
    mask = binary_fill_holes(
        mask
    )

    # Remove tiny regions.
    labeled, count = label(mask)

    cleaned = np.zeros_like(
        mask,
        dtype=bool
    )

    for region_id in range(
        1,
        count + 1
    ):

        region = (
            labeled == region_id
        )

        if region.sum() >= MIN_REGION_AREA:

            cleaned |= region

    return cleaned


# ============================================================
# CREATE GRAD-CAM IMAGE
# ============================================================

def create_gradcam_image(
    original,
    heatmap
):

    heatmap_rgb = colormaps[
        "jet"
    ](
        heatmap
    )[
        ..., :3
    ]

    overlay = (
        original * (1 - GRADCAM_ALPHA)
        +
        heatmap_rgb * GRADCAM_ALPHA
    )

    return (
        np.clip(
            overlay,
            0,
            1
        ) * 255
    ).astype(
        np.uint8
    )


# ============================================================
# CREATE CIRCLED DISEASE IMAGE
# ============================================================

def create_region_image(
    original,
    heatmap
):

    region_mask = create_disease_regions(
        original,
        heatmap
    )

    result = (
        original.copy()
    )

    # Slightly fade outside highlighted regions.
    faded = (
        result * 0.72
    )

    result[
        ~region_mask
    ] = faded[
        ~region_mask
    ]

    # Draw natural contours.
    contours = find_contours(
        region_mask.astype(
            np.float32
        ),
        0.5
    )

    for contour in contours:

        if len(contour) < 15:

            continue

        # Smooth contour.
        y = gaussian_filter(
            contour[:, 0],
            sigma=1.2
        )

        x = gaussian_filter(
            contour[:, 1],
            sigma=1.2
        )

        # Draw yellow contour.
        for offset in range(-1, 2):

            yy = np.clip(
                np.round(y).astype(int)
                + offset,
                0,
                result.shape[0] - 1
            )

            xx = np.clip(
                np.round(x).astype(int),
                0,
                result.shape[1] - 1
            )

            result[
                yy,
                xx
            ] = np.array(
                [255, 220, 0]
            )

    return result.astype(
        np.uint8
    )


# ============================================================
# PREDICTION
# ============================================================

def predict_image(
    image
):

    transform = get_transform()

    tensor = transform(
        image
    ).unsqueeze(0).to(
        DEVICE
    )

    model = state["model"]

    with torch.no_grad():

        output = model(
            tensor
        )

        probabilities = torch.softmax(
            output,
            dim=1
        )

        confidence, prediction = (
            probabilities.max(
                dim=1
            )
        )

    pred_index = int(
        prediction.item()
    )

    confidence_value = float(
        confidence.item()
    )

    probs = probabilities[
        0
    ].cpu().numpy()

    return (
        tensor,
        pred_index,
        confidence_value,
        probs
    )


# ============================================================
# API HEALTH
# ============================================================

@app.get("/api/health")
def health():

    return {
        "status": "ok",
        "device": str(DEVICE),
        "model": "EfficientNet-B4",
        "num_classes": state["num_classes"],
        "image_size": state["img_size"],
        "best_validation_accuracy": state[
            "best_accuracy"
        ]
    }


# ============================================================
# PREDICT API
# ============================================================

@app.post("/api/predict")
async def predict(
    file: UploadFile = File(...)
):

    # --------------------------------------------------------
    # CHECK FILE
    # --------------------------------------------------------

    if not file.content_type:

        raise HTTPException(
            status_code=400,
            detail="No file type detected."
        )

    allowed_types = {
        "image/jpeg",
        "image/png",
        "image/webp",
        "image/bmp"
    }

    if file.content_type not in allowed_types:

        raise HTTPException(
            status_code=400,
            detail=(
                "Please upload a JPG, PNG, WEBP "
                "or BMP image."
            )
        )

    # --------------------------------------------------------
    # READ FILE
    # --------------------------------------------------------

    data = await file.read()

    if len(data) > MAX_BYTES:

        raise HTTPException(
            status_code=413,
            detail="Image must be smaller than 15 MB."
        )

    # --------------------------------------------------------
    # OPEN IMAGE
    # --------------------------------------------------------

    try:

        image = Image.open(
            io.BytesIO(data)
        )

        image = ImageOps.exif_transpose(
            image
        ).convert(
            "RGB"
        )

    except Exception:

        raise HTTPException(
            status_code=400,
            detail=(
                "Unable to read this image. "
                "Please use JPG, PNG or WEBP."
            )
        )

    # --------------------------------------------------------
    # ORIGINAL IMAGE
    # --------------------------------------------------------

    original = image.resize(
        (
            state["img_size"],
            state["img_size"]
        )
    )

    original_array = (
        np.asarray(
            original
        ).astype(
            np.float32
        ) / 255.0
    )

    start = time.time()

    # --------------------------------------------------------
    # PREDICTION + GRAD-CAM
    # --------------------------------------------------------

    with lock:

        tensor, pred_index, confidence, probabilities = (
            predict_image(
                image
            )
        )

        gradcam = create_gradcam()

        try:

            # Grad-CAM needs gradients.
            with torch.enable_grad():

                heatmap = gradcam.generate(
                    tensor,
                    pred_index
                )

        finally:

            gradcam.close()

    # --------------------------------------------------------
    # CLASS
    # --------------------------------------------------------

    classes = state["classes"]

    predicted_class = classes[
        pred_index
    ]

    pretty_prediction = pretty_class(
        predicted_class
    )

    # --------------------------------------------------------
    # SPECIES / DISEASE
    # --------------------------------------------------------

    if predicted_class in M.CLASS_TO_SD:

        species, disease = (
            M.CLASS_TO_SD[
                predicted_class
            ]
        )

    else:

        species = "Unknown"
        disease = "Unknown"

    pretty_species = pretty_class(
        species
    )

    pretty_disease = pretty_class(
        disease
    )

    # --------------------------------------------------------
    # TOP 5
    # --------------------------------------------------------

    top_indices = np.argsort(
        -probabilities
    )[:5]

    top5 = []

    for index in top_indices:

        top5.append({
            "label": pretty_class(
                classes[index]
            ),
            "confidence": float(
                probabilities[index]
            )
        })

    # --------------------------------------------------------
    # CREATE VISUALS
    # --------------------------------------------------------

    gradcam_image = create_gradcam_image(
        original_array,
        heatmap
    )

    region_image = create_region_image(
        original_array,
        heatmap
    )

    original_image = (
        original_array * 255
    ).astype(
        np.uint8
    )

    # --------------------------------------------------------
    # RESPONSE
    # --------------------------------------------------------

    elapsed = int(
        (time.time() - start) * 1000
    )

    return {
        "success": True,

        "prediction": pretty_prediction,

        "class_id": predicted_class,

        "species": pretty_species,

        "disease": pretty_disease,

        "healthy": (
            disease.lower() == "healthy"
        ),

        "confidence": confidence,

        "top5": top5,

        "images": {

            "original": image_to_base64(
                original_image
            ),

            "gradcam": image_to_base64(
                gradcam_image
            ),

            "regions": image_to_base64(
                region_image
            )
        },

        "elapsed_ms": elapsed,

        "device": str(DEVICE)
    }


# ============================================================
# FRONTEND
# ============================================================

@app.get(
    "/",
    response_class=HTMLResponse
)
def index():

    if not INDEX_PATH.exists():

        return HTMLResponse(
            "<h1>index.html not found</h1>",
            status_code=500
        )

    return HTMLResponse(
        INDEX_PATH.read_text(
            encoding="utf-8"
        )
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )