# comparison_app.py

from pathlib import Path
import base64
import io
import os
import tempfile
import time

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image


ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Underwater Enhancement Inspector",
    page_icon="🌊",
    layout="wide"
)


# ==========================================================
# IMAGE STATISTICS
# ==========================================================

def image_stats(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    contrast = float(gray.std())
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    saturation = float(hsv[:, :, 1].mean()) / 255.0

    return {
        "contrast": contrast,
        "sharpness": sharpness,
        "saturation": saturation
    }


def encode_png(rgb):
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def resize_to_original(image, width, height):
    if image.shape[:2] != (height, width):
        image = cv2.resize(
            image,
            (width, height),
            interpolation=cv2.INTER_LANCZOS4
        )
    return image


# ==========================================================
# MODEL LOADING
# ==========================================================

@st.cache_resource
def load_engine(checkpoint, device):
    from infer import UnderwaterEnhancer
    return UnderwaterEnhancer(checkpoint_path=None, device=device)



# ==========================================================
# UI STYLE
# ==========================================================

st.markdown("""
<style>
.stApp {
    background: #0b1220;
    color: #e6edf7;
}

.block-container {
    max-width: 1500px;
    padding-top: 1.5rem;
}

[data-testid="stHeader"] {
    background: transparent;
}

.metric-card {
    border: 1px solid #263650;
    border-radius: 14px;
    padding: 15px;
    background: #111c2e;
}

</style>
""", unsafe_allow_html=True)


# ==========================================================
# HEADER
# ==========================================================

st.title("🌊 Underwater Enhancement Inspector")

st.caption(
    "Interactive inspection of underwater image enhancement "
    "with synchronized comparison and zoom."
)


# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:

    st.header("Inference Settings")

    checkpoint = st.text_input(
        "Checkpoint",
        value="checkpoints/best_model.pth"
    )

    device = st.selectbox(
        "Device",
        ["cuda", "cpu", "mps"],
        index=0
    )

    uploaded_file = st.file_uploader(
        "Upload underwater image",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"]
    )

    st.caption(
        "The existing inference engine is used to generate "
        "the enhanced image."
    )


# ==========================================================
# MAIN WORKFLOW
# ==========================================================

if uploaded_file is not None:

    try:

        # --------------------------------------------------
        # Read original image
        # --------------------------------------------------

        pil_image = Image.open(uploaded_file).convert("RGB")

        original = np.asarray(pil_image)

        original_height, original_width = original.shape[:2]

        st.write(
            f"**Image:** {uploaded_file.name} | "
            f"{original_width} × {original_height}px"
        )

        # --------------------------------------------------
        # Check checkpoint
        # --------------------------------------------------

       

        # --------------------------------------------------
        # Temporary input file for existing inferencer
        # --------------------------------------------------

        with tempfile.NamedTemporaryFile(
            suffix=Path(uploaded_file.name).suffix,
            delete=False
        ) as temp_file:

            temp_file.write(uploaded_file.getvalue())

            temp_path = temp_file.name

        # --------------------------------------------------
        # Existing inference engine
        # --------------------------------------------------

        try:

            engine = load_engine(
                str(Path(checkpoint).resolve()),
                device
            )

            # Measure inference execution time
            start_time = time.perf_counter()

            result = engine.infer_single(temp_path)

            # Synchronize CUDA for more accurate timing
            if device == "cuda":

                import torch

                if torch.cuda.is_available():
                    torch.cuda.synchronize()

            inference_time_ms = (
                time.perf_counter() - start_time
            ) * 1000.0

        finally:

            if os.path.exists(temp_path):
                os.unlink(temp_path)

        # --------------------------------------------------
        # Obtain inference images
        # --------------------------------------------------

        original_result = result["original"]
        enhanced = result["enhanced"]

        original_result = resize_to_original(
            original_result,
            original_width,
            original_height
        )

        enhanced = resize_to_original(
            enhanced,
            original_width,
            original_height
        )

        # --------------------------------------------------
        # Image statistics
        # --------------------------------------------------

        original_stats = image_stats(original_result)
        enhanced_stats = image_stats(enhanced)

        # Descriptive indicators, not validated quality scores.
        original_score = (
            original_stats["contrast"] / 64.0
            + original_stats["sharpness"] / 1000.0
            + original_stats["saturation"]
        )

        enhanced_score = (
            enhanced_stats["contrast"] / 64.0
            + enhanced_stats["sharpness"] / 1000.0
            + enhanced_stats["saturation"]
        )

        enhancement_gain = enhanced_score - original_score

        # --------------------------------------------------
        # INFORMATION PANEL
        # --------------------------------------------------

        st.subheader("Image Information")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Resolution",
            f"{original_width} × {original_height}"
        )

        c2.metric(
            "Inference time",
            f"{inference_time_ms:.2f} ms"
        )

        c3.metric(
            "Statistics delta",
            f"{enhancement_gain:+.3f}"
        )

        c4.metric(
            "Checkpoint",
            Path(checkpoint).name
        )

        # --------------------------------------------------
        # COMPARISON VIEWER
        # --------------------------------------------------

        st.subheader("Interactive Comparison")

        viewer_mode = st.radio(
            "Comparison mode",
            ["Split Slider", "Side-by-Side"],
            horizontal=True
        )

        original_b64 = encode_png(original_result)
        enhanced_b64 = encode_png(enhanced)

        # --------------------------------------------------
        # HTML COMPARISON VIEWER
        # --------------------------------------------------

        viewer_html = f"""
<!DOCTYPE html>

<html>

<head>

<meta charset="UTF-8">

<style>

* {{
    box-sizing: border-box;
}}

body {{
    margin: 0;
    background: #0b1220;
    color: #e6edf7;
    font-family: Arial, sans-serif;
}}

.toolbar {{
    display: flex;
    justify-content: center;
    align-items: center;
    flex-wrap: wrap;
    gap: 10px;
    padding: 10px;
}}

button {{
    background: #18263c;
    color: #e6edf7;
    border: 1px solid #34465f;
    border-radius: 8px;
    padding: 9px 14px;
    cursor: pointer;
}}

button:hover {{
    background: #263954;
}}

.hint {{
    color: #8fa4bf;
    font-size: 12px;
}}

.viewer {{
    width: 100%;
    height: 70vh;
    min-height: 350px;
    max-height: 750px;
    position: relative;
    overflow: hidden;
    background: #080e18;
    border: 1px solid #263650;
    border-radius: 12px;
    touch-action: none;
    user-select: none;
}}

.canvas {{
    width: 100%;
    height: 100%;
    display: block;
}}

.label {{
    position: absolute;
    top: 12px;
    z-index: 5;
    background: rgba(16,26,43,0.9);
    padding: 8px 12px;
    border: 1px solid #34465f;
    border-radius: 8px;
    pointer-events: none;
}}

.left-label {{
    left: 12px;
}}

.right-label {{
    right: 12px;
}}

.hidden {{
    display: none !important;
}}

</style>

</head>

<body>

<div class="toolbar">

<button id="zoomOut">− Zoom</button>

<button id="zoomIn">+ Zoom</button>

<button id="reset">Reset</button>

<span class="hint">
Scroll to zoom · Drag to pan · Slider to compare
</span>

</div>

<div id="splitViewer" class="viewer">

<canvas id="splitCanvas" class="canvas"></canvas>

<div class="label left-label">
Original
</div>

<div class="label right-label">
Enhanced
</div>

</div>

<div id="sideViewer" class="viewer hidden">

<canvas id="sideCanvas" class="canvas"></canvas>

</div>

<script>

const originalImage = new Image();
const enhancedImage = new Image();

originalImage.src =
    "data:image/png;base64,{original_b64}";

enhancedImage.src =
    "data:image/png;base64,{enhanced_b64}";

const splitCanvas =
    document.getElementById("splitCanvas");

const sideCanvas =
    document.getElementById("sideCanvas");

const splitCtx =
    splitCanvas.getContext("2d");

const sideCtx =
    sideCanvas.getContext("2d");

const splitViewer =
    document.getElementById("splitViewer");

const sideViewer =
    document.getElementById("sideViewer");

let sliderPosition = 0.5;

let zoom = 1;

let panX = 0;

let panY = 0;

let draggingSlider = false;

let draggingPan = false;

let lastX = 0;

let lastY = 0;

let activeMode = "{viewer_mode}";


function resizeCanvas(canvas) {{

    const rect = canvas.getBoundingClientRect();

    const dpr = window.devicePixelRatio || 1;

    canvas.width = rect.width * dpr;

    canvas.height = rect.height * dpr;

    return {{
        width: rect.width,
        height: rect.height,
        dpr: dpr
    }};

}}


function drawImagePair(ctx, canvas, splitMode) {{

    const dimensions = resizeCanvas(canvas);

    const width = dimensions.width;

    const height = dimensions.height;

    const dpr = dimensions.dpr;

    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    ctx.clearRect(0, 0, width, height);

    if (!originalImage.complete ||
        !enhancedImage.complete) {{
        return;
    }}

    const imageWidth = originalImage.naturalWidth;

    const imageHeight = originalImage.naturalHeight;

    const fitScale = Math.min(
        width / imageWidth,
        height / imageHeight
    );

    const scale = fitScale * zoom;

    const drawWidth = imageWidth * scale;

    const drawHeight = imageHeight * scale;

    const centerX = width / 2 + panX;

    const centerY = height / 2 + panY;

    const x = centerX - drawWidth / 2;

    const y = centerY - drawHeight / 2;

    // Enhanced image is the background.
    ctx.drawImage(
        enhancedImage,
        x,
        y,
        drawWidth,
        drawHeight
    );

    if (splitMode) {{

        // Clip original image to slider region.
        ctx.save();

        ctx.beginPath();

        ctx.rect(
            0,
            0,
            width * sliderPosition,
            height
        );

        ctx.clip();

        ctx.drawImage(
            originalImage,
            x,
            y,
            drawWidth,
            drawHeight
        );

        ctx.restore();

        // Divider
        const dividerX = width * sliderPosition;

        ctx.beginPath();

        ctx.moveTo(dividerX, 0);

        ctx.lineTo(dividerX, height);

        ctx.strokeStyle = "#ffffff";

        ctx.lineWidth = 3;

        ctx.stroke();

        // Handle
        ctx.beginPath();

        ctx.arc(
            dividerX,
            height / 2,
            22,
            0,
            Math.PI * 2
        );

        ctx.fillStyle = "#ffffff";

        ctx.fill();

        ctx.strokeStyle = "#1d2939";

        ctx.lineWidth = 2;

        ctx.stroke();

        ctx.fillStyle = "#111827";

        ctx.font = "bold 16px Arial";

        ctx.textAlign = "center";

        ctx.textBaseline = "middle";

        ctx.fillText(
            "↔",
            dividerX,
            height / 2
        );

    }} else {{

        // Side-by-side rendering with identical scale.
        ctx.clearRect(0, 0, width, height);

        const gap = 12;

        const panelWidth = (width - gap) / 2;

        const panelHeight = height;

        const fitScaleSide = Math.min(
            panelWidth / imageWidth,
            panelHeight / imageHeight
        );

        const scaleSide = fitScaleSide * zoom;

        const drawWidthSide = imageWidth * scaleSide;

        const drawHeightSide = imageHeight * scaleSide;

        const panelCenterY = height / 2 + panY;

        const leftCenterX = panelWidth / 2 + panX;

        const rightCenterX = panelWidth + gap
            + panelWidth / 2 + panX;

        const leftX = leftCenterX - drawWidthSide / 2;

        const rightX = rightCenterX - drawWidthSide / 2;

        const drawY = panelCenterY - drawHeightSide / 2;

        // Panel boundaries
        ctx.fillStyle = "#080e18";

        ctx.fillRect(
            0,
            0,
            width,
            height
        );

        ctx.save();

        ctx.beginPath();

        ctx.rect(
            0,
            0,
            panelWidth,
            panelHeight
        );

        ctx.clip();

        ctx.drawImage(
            originalImage,
            leftX,
            drawY,
            drawWidthSide,
            drawHeightSide
        );

        ctx.restore();

        ctx.save();

        ctx.beginPath();

        ctx.rect(
            panelWidth + gap,
            0,
            panelWidth,
            panelHeight
        );

        ctx.clip();

        ctx.drawImage(
            enhancedImage,
            rightX,
            drawY,
            drawWidthSide,
            drawHeightSide
        );

        ctx.restore();

        // Panel divider
        ctx.fillStyle = "#263650";

        ctx.fillRect(
            panelWidth,
            0,
            gap,
            height
        );

        // Labels
        ctx.fillStyle = "#101a2b";

        ctx.fillRect(12, 12, 100, 36);

        ctx.fillRect(
            panelWidth + gap + 12,
            12,
            100,
            36
        );

        ctx.fillStyle = "#e6edf7";

        ctx.font = "14px Arial";

        ctx.textAlign = "left";

        ctx.fillText(
            "Original",
            22,
            34
        );

        ctx.fillText(
            "Enhanced",
            panelWidth + gap + 22,
            34
        );

    }}

}}


function render() {{

    if (activeMode === "Split Slider") {{

        drawImagePair(
            splitCtx,
            splitCanvas,
            true
        );

    }} else {{

        drawImagePair(
            sideCtx,
            sideCanvas,
            false
        );

    }}

}}


function setMode(mode) {{

    activeMode = mode;

    if (mode === "Split Slider") {{

        splitViewer.classList.remove("hidden");

        sideViewer.classList.add("hidden");

    }} else {{

        splitViewer.classList.add("hidden");

        sideViewer.classList.remove("hidden");

    }}

    render();

}}


// Slider interaction
splitCanvas.addEventListener(
    "pointerdown",
    function(e) {{

        const rect = splitCanvas.getBoundingClientRect();

        const x = e.clientX - rect.left;

        const dividerX =
            rect.width * sliderPosition;

        if (Math.abs(x - dividerX) < 35) {{

            draggingSlider = true;

        }} else {{

            draggingPan = true;

            lastX = e.clientX;

            lastY = e.clientY;

        }}

        splitCanvas.setPointerCapture(e.pointerId);

    }}
);


splitCanvas.addEventListener(
    "pointermove",
    function(e) {{

        if (draggingSlider) {{

            const rect = splitCanvas.getBoundingClientRect();

            sliderPosition = Math.max(
                0,
                Math.min(
                    1,
                    (e.clientX - rect.left) / rect.width
                )
            );

            render();

        }} else if (draggingPan) {{

            panX += e.clientX - lastX;

            panY += e.clientY - lastY;

            lastX = e.clientX;

            lastY = e.clientY;

            render();

        }}

    }}
);


splitCanvas.addEventListener(
    "pointerup",
    function() {{

        draggingSlider = false;

        draggingPan = false;

    }}
);


// Pan in side-by-side mode
sideCanvas.addEventListener(
    "pointerdown",
    function(e) {{

        draggingPan = true;

        lastX = e.clientX;

        lastY = e.clientY;

        sideCanvas.setPointerCapture(e.pointerId);

    }}
);


sideCanvas.addEventListener(
    "pointermove",
    function(e) {{

        if (draggingPan) {{

            panX += e.clientX - lastX;

            panY += e.clientY - lastY;

            lastX = e.clientX;

            lastY = e.clientY;

            render();

        }}

    }}
);


sideCanvas.addEventListener(
    "pointerup",
    function() {{

        draggingPan = false;

    }}
);


// Zoom
function setZoom(value) {{

    zoom = Math.max(
        1,
        Math.min(
            5,
            value
        )
    );

    render();

}}


document.getElementById("zoomIn").onclick =
    () => setZoom(zoom * 1.2);


document.getElementById("zoomOut").onclick =
    () => setZoom(zoom / 1.2);


document.getElementById("reset").onclick =
    function() {{

        zoom = 1;

        panX = 0;

        panY = 0;

        sliderPosition = 0.5;

        render();

    }};


// Mouse wheel zoom
splitViewer.addEventListener(
    "wheel",
    function(e) {{

        e.preventDefault();

        setZoom(
            zoom * (e.deltaY < 0 ? 1.1 : 0.9)
        );

    }},
    {{passive: false}}
);


sideViewer.addEventListener(
    "wheel",
    function(e) {{

        e.preventDefault();

        setZoom(
            zoom * (e.deltaY < 0 ? 1.1 : 0.9)
        );

    }},
    {{passive: false}}
);


window.addEventListener(
    "resize",
    render
);


originalImage.onload = render;

enhancedImage.onload = render;


// Initialize
setMode("{viewer_mode}");

</script>

</body>

</html>
"""

        components.html(
            viewer_html,
            height=800,
            scrolling=False
        )

        # ==================================================
        # STATISTICS
        # ==================================================

        with st.expander(
            "Detailed image statistics",
            expanded=False
        ):

            st.caption(
                "Reference-free image statistics. "
                "These are descriptive measurements, not "
                "ground-truth perceptual quality scores."
            )

            st.table({

                "Statistic": [
                    "Contrast (grayscale standard deviation)",
                    "Sharpness (Laplacian variance)",
                    "Mean saturation"
                ],

                "Original": [
                    f"{original_stats['contrast']:.3f}",
                    f"{original_stats['sharpness']:.3f}",
                    f"{original_stats['saturation']:.4f}"
                ],

                "Enhanced": [
                    f"{enhanced_stats['contrast']:.3f}",
                    f"{enhanced_stats['sharpness']:.3f}",
                    f"{enhanced_stats['saturation']:.4f}"
                ],

                "Change": [
                    f"{enhanced_stats['contrast'] - original_stats['contrast']:+.3f}",
                    f"{enhanced_stats['sharpness'] - original_stats['sharpness']:+.3f}",
                    f"{enhanced_stats['saturation'] - original_stats['saturation']:+.4f}"
                ]

            })

        # ==================================================
        # EXISTING VISUALIZATION OUTPUTS
        # ==================================================

        st.subheader("Existing inference visualizations")

        visualization_keys = [
            "noise_map",
            "illum_map",
            "colour_map",
            "overlay_noise",
            "attn_stage1",
            "attn_stage2",
            "attn_stage3"
        ]

        visualization_columns = st.columns(4)

        for index, key in enumerate(visualization_keys):

            if key in result:

                visualization_columns[index % 4].image(
                    result[key],
                    caption=key.replace("_", " ").title(),
                    use_container_width=True
                )

        # ==================================================
        # DOWNLOAD
        # ==================================================

        output_buffer = io.BytesIO()

        Image.fromarray(enhanced).save(
            output_buffer,
            format="PNG"
        )

        st.download_button(
            "Download enhanced image",
            data=output_buffer.getvalue(),
            file_name="enhanced.png",
            mime="image/png"
        )

    except Exception as exception:

        st.error(
            f"Image processing failed: {exception}"
        )

        st.exception(exception)

else:

    st.info(
        "Upload an underwater image from the sidebar "
        "to begin image enhancement and comparison."
    )