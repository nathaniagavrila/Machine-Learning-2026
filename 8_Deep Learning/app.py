"""Aplikasi Streamlit - Klasifikasi Gambar Pakaian (Fashion-MNIST).

Menjalankan secara lokal:
    pip install -r requirements.txt
    streamlit run app.py
File model.keras harus berada di folder yang sama dengan app.py.
"""
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
from tensorflow import keras

CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

st.set_page_config(page_title="Klasifikasi Pakaian Fashion-MNIST", page_icon="👕", layout="centered")


@st.cache_resource
def load_model():
    return keras.models.load_model("model.keras")


def preprocess(image: Image.Image, invert: bool) -> np.ndarray:
    """Samakan gambar dengan format data latih: grayscale, 28x28, objek terang di atas latar hitam, skala 0-1."""
    img = image.convert("L")                      # grayscale
    if invert:
        img = ImageOps.invert(img)                # latar putih -> latar hitam
    img = ImageOps.pad(img, (28, 28), color=0)    # resize tanpa mengubah rasio, sisanya diisi hitam
    return np.asarray(img, dtype="float32") / 255.0


def to_model_input(arr: np.ndarray, model) -> np.ndarray:
    """ANN butuh (1, 784), CNN butuh (1, 28, 28, 1). Bentuk dipilih otomatis dari input_shape model."""
    if len(model.input_shape) == 2:
        return arr.reshape(1, 784)
    return arr.reshape(1, 28, 28, 1)


model = load_model()

st.title("👕 Klasifikasi Gambar Pakaian")
st.write("Unggah gambar pakaian, lalu model CNN yang dilatih pada dataset Fashion-MNIST "
         "akan memprediksi jenisnya dari 10 kelas.")

with st.sidebar:
    st.header("Pengaturan")
    invert_mode = st.radio(
        "Latar belakang gambar",
        ["Otomatis", "Terang (foto produk biasa)", "Gelap (seperti Fashion-MNIST)"],
        help="Model dilatih dengan objek terang di atas latar hitam. Gambar berlatar terang perlu dibalik warnanya.",
    )
    st.header("Daftar kelas")
    for i, name in enumerate(CLASS_NAMES):
        st.write(f"{i} — {name}")

uploaded = st.file_uploader("Unggah gambar (PNG / JPG)", type=["png", "jpg", "jpeg"])

if uploaded is not None:
    image = Image.open(uploaded)
    if image.mode in ("RGBA", "LA", "P"):         # gambar transparan -> tempel di atas latar putih
        image = image.convert("RGBA")
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, image)
    image = image.convert("RGB")

    if invert_mode == "Otomatis":
        gray = np.asarray(image.convert("L"))
        border = np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]])
        invert = border.mean() > 127             # tepi gambar terang -> latar terang -> perlu dibalik
    else:
        invert = invert_mode.startswith("Terang")

    arr = preprocess(image, invert)
    proba = model.predict(to_model_input(arr, model), verbose=0)[0]
    pred = int(proba.argmax())

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Gambar asli", use_container_width=True)
    with col2:
        st.image(arr, caption="Input model (28×28, grayscale)", use_container_width=True, clamp=True)

    st.subheader(f"Prediksi: {CLASS_NAMES[pred]}")
    st.write(f"Tingkat keyakinan: **{proba[pred] * 100:.2f}%**")

    chart_df = pd.DataFrame({"Probabilitas": proba}, index=CLASS_NAMES)
    st.bar_chart(chart_df)
else:
    st.info("Silakan unggah gambar untuk memulai. Hasil terbaik didapat dari gambar satu objek pakaian "
            "dengan latar polos, karena model dilatih pada gambar 28×28 piksel.")