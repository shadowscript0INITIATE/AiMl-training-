import streamlit as st
from ultralytics import YOLO
import cv2
import os

st.title("Object Detection using YOLOv11")

st.write(
    "Upload an image to detect people and birds, and count how many of each are present."
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    os.makedirs("uploads", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    image_path = os.path.join("uploads", uploaded_file.name)

    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image(image_path, caption="Uploaded Image", use_container_width=True)

    if st.button("Detect Objects"):

        model = YOLO("yolo11n.pt")

        predict_class = [0, 14]

        pred = model(
            image_path,
            classes=predict_class,
            conf=0.4
        )

        class_names = model.names
        counts = {}

        for cls in pred[0].boxes.cls:
            cls = int(cls)
            name = class_names[cls]
            counts[name] = counts.get(name, 0) + 1

        pred_img = pred[0].plot()

        output_path = os.path.join(
            "output",
            "detected_" + uploaded_file.name
        )

        cv2.imwrite(output_path, pred_img)

        st.subheader("Detected Image")
        st.image(output_path, use_container_width=True)

        st.subheader("Object Counts")

        if counts:
            for name, count in counts.items():
                st.write(f"**{name.title()}** : {count}")
        else:
            st.write("No specified objects detected.")