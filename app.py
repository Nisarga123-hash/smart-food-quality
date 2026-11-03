import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Smart Food Quality Detection",
    page_icon="🍎",
    layout="wide"
)

st.title("🍎 Smart Food Quality Detection")
st.write(
    "Upload a fruit image to analyze its visible appearance "
    "using digital image processing."
)

st.info(
    "This is an experimental visual screening tool. "
    "It does not determine food safety or detect internal spoilage."
)

uploaded_file = st.file_uploader(
    "Upload a fruit image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Step 1: Read the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    original = np.array(image)

    # Step 2: Resize the image
    height, width = original.shape[:2]

    scale = min(800 / width, 800 / height, 1.0)

    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))

    resized = cv2.resize(
        original,
        (new_width, new_height)
    )

    # Step 3: Convert RGB to HSV
    hsv = cv2.cvtColor(
        resized,
        cv2.COLOR_RGB2HSV
    )

    # Step 4: Estimate the fruit region
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    fruit_candidate = np.where(
        (saturation > 35) | (value < 190),
        255,
        0
    ).astype(np.uint8)

    # Remove small noise and fill small gaps
    kernel = np.ones((5, 5), np.uint8)

    fruit_candidate = cv2.morphologyEx(
        fruit_candidate,
        cv2.MORPH_OPEN,
        kernel
    )

    fruit_candidate = cv2.morphologyEx(
        fruit_candidate,
        cv2.MORPH_CLOSE,
        kernel
    )

    # Find the largest candidate region
    contours, _ = cv2.findContours(
        fruit_candidate,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    fruit_mask = np.zeros(
        (new_height, new_width),
        dtype=np.uint8
    )

    if contours:
        largest_contour = max(
            contours,
            key=cv2.contourArea
        )

        cv2.drawContours(
            fruit_mask,
            [largest_contour],
            -1,
            255,
            thickness=cv2.FILLED
        )

    # Check whether a usable fruit region was found
    fruit_pixels = cv2.countNonZero(fruit_mask)
    total_pixels = new_width * new_height

    if fruit_pixels < 0.05 * total_pixels:
        st.warning(
            "The fruit region could not be identified reliably. "
            "Try a clear image with a plain background."
        )

    else:
        # Step 5: Identify potential brown blemishes
        hue = hsv[:, :, 0]

        brown_mask = (
            (hue >= 5) &
            (hue <= 25) &
            (saturation > 40) &
            (value < 180)
        )

        brown_mask = (
            brown_mask.astype(np.uint8) * 255
        )

        # Keep suspected blemishes inside the fruit region
        blemish_mask = cv2.bitwise_and(
            brown_mask,
            fruit_mask
        )

        # Step 6: Calculate the estimated blemish percentage
        blemish_pixels = cv2.countNonZero(
            blemish_mask
        )

        blemish_percentage = (
            blemish_pixels / fruit_pixels
        ) * 100

        # Step 7: Estimate the visual quality category
        # These are illustrative thresholds, not validated standards.
        if blemish_percentage < 2:
            category = "Good apparent visual quality"
            explanation = (
                "Few brown-coloured pixels were detected."
            )

        elif blemish_percentage < 8:
            category = "Moderate apparent visual quality"
            explanation = (
                "Some brown-coloured regions were detected."
            )

        else:
            category = "Poor apparent visual quality"
            explanation = (
                "A relatively large proportion of brown-coloured "
                "pixels was detected."
            )

        # Step 8: Display results
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Image")
            st.image(
                resized,
                use_container_width=True
            )

        with col2:
            st.subheader("Detected Fruit Region")

            fruit_display = cv2.bitwise_and(
                resized,
                resized,
                mask=fruit_mask
            )

            st.image(
                fruit_display,
                use_container_width=True
            )

        st.subheader("Suspected Brown Regions")
        st.image(
            blemish_mask,
            caption="White pixels indicate detected brown-coloured regions.",
            clamp=True,
            use_container_width=True
        )

        st.subheader("Analysis Results")

        st.metric(
            "Estimated blemish percentage",
            f"{blemish_percentage:.2f}%"
        )

        st.write("**Estimated category:**", category)
        st.write("**Observation:**", explanation)

        st.warning(
            "This prototype uses simple colour thresholds. "
            "Lighting, fruit variety, natural markings and background "
            "can affect the results. The category is not a verified "
            "measure of freshness, edibility or food safety."
        )