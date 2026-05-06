import streamlit as st
from PIL import Image
import numpy as np
import cv2
import mediapipe as mp
import torch
# from diffusers import StableDiffusionImg2ImgPipeline
print("OK")

# -----------------------------
# Load Model (only once)
# -----------------------------


@st.cache_resource
def load_pipeline():
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)

    return pipe


pipe = load_pipeline()

# -----------------------------
# MediaPipe Setup
# -----------------------------
mp_face = mp.solutions.face_detection
mp_pose = mp.solutions.pose


def analyze_image(image):
    img = np.array(image)

    face_detection = mp_face.FaceDetection()
    pose = mp_pose.Pose()

    results_face = face_detection.process(img)
    results_pose = pose.process(img)

    features = {
        "face_detected": results_face.detections is not None,
        "pose_detected": results_pose.pose_landmarks is not None,
    }

    avg_color = img.mean(axis=(0, 1))
    features["skin_tone"] = "light" if avg_color[0] > 140 else "medium"

    return features

# -----------------------------
# Prompt Builder
# -----------------------------


def build_prompt(age, weather, budget, style, features):
    skin = features.get("skin_tone", "medium")

    return f"""
    ultra realistic full body fashion photo,
    {age} year old person,
    {skin} skin tone,
    wearing {style} outfit,
    suitable for {weather} weather,
    budget {budget},
    trendy fashion 2025,
    highly detailed, 4k, professional photoshoot,
    perfect fit, cinematic lighting
    """

# -----------------------------
# Generate Image
# -----------------------------


def generate_image(prompt, init_image):
    init_image = init_image.resize((512, 512))

    image = pipe(
        prompt=prompt,
        image=init_image,
        strength=0.75,
        guidance_scale=7.5
    ).images[0]

    return image


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("👗 AI Fashion Stylist (Local GenAI)")

camera_image = st.camera_input("Take a photo")

age = st.slider("Age", 10, 60)
weather = st.selectbox("Weather", ["Sunny", "Rainy", "Winter", "Humid"])
budget = st.selectbox("Budget", ["Low", "Medium", "High"])
style = st.selectbox("Style", [
    "Indian", "Korean", "Indo-Western",
    "Indo-Korean", "American", "Canadian"
])

if st.button("Generate Style"):
    if camera_image:
        image = Image.open(camera_image)

        st.image("https://via.placeholder.com/512", caption="Demo Output")

        # Step 1: Analyze
        features = analyze_image(image)

        # Step 2: Prompt
        prompt = build_prompt(age, weather, budget, style, features)
        st.write("🧠 Prompt:", prompt)

        # Step 3: Generate
        output = generate_image(prompt, image)

        st.image(output, caption="✨ Styled Output")
    else:
        st.warning("Please capture an image first")
