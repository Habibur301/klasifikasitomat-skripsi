import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2

# Memuat model yang telah dilatih 
model = tf.keras.models.load_model("resnettt.h5")

# Label kelas sesuai dengan urutan saat pelatihan
class_labels = ['Reject', 'Ripe', 'Unripe'] 

# Fungsi prediksi
def predict_image(image, model):
    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (224, 224))
    img_array = np.expand_dims(img, axis=0)
    # Tanpa normalisasi, karena model dilatih tanpa rescale
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions, axis=1)[0]
    predicted_label = class_labels[predicted_index]
    return predicted_label

# Fungsi klasifikasi gambar
def classify_image(uploaded_file):
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Gambar yang Diunggah", use_container_width=True)
    img_array = np.array(image)
    class_label = predict_image(img_array, model)
    st.success(f"Hasil Klasifikasi: **{class_label}**")

# Halaman utama aplikasi
def main():
    st.markdown("<h1 style='text-align: center; color: green;'>🍅 Klasifikasi Kematangan Tomat</h1>", unsafe_allow_html=True)
    st.markdown("Upload gambar tomat untuk mengetahui status kematangannya: **Ripe**, **Unripe**, atau **Reject**.")
    
    uploaded_file = st.file_uploader("Unggah Gambar Tomat", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        classify_image(uploaded_file)

if __name__ == "__main__":
    main()