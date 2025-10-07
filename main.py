import streamlit as st
import numpy as np
from keras.models import load_model
from PIL import Image, ImageOps
from huggingface_hub import hf_hub_download


# Judul aplikasi
st.title("🖐️ Deteksi Gestur Tangan BISINDO")
st.header("Klasifikasi Gestur Bahasa Isyarat Indonesia (BISINDO): Maaf, Baik, Saya, Ibu, dan Bapak")
st.write("""
Aplikasi ini menggunakan model **CNN** yang dilatih dengan **Teachable Machine** dan **MediaPipe** 
untuk mengenali gestur tangan BISINDO.
""")

# Inisialisasi Mediapipe
# mpHands = mp.solutions.hands
# hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
# mpDraw = mp.solutions.drawing_utils

# Pilihan model
models = st.selectbox(
    'Pilih Model',
    ('CNN - Epochs = 5', 'CNN - Epochs = 50', 'Teachable Machine')
)

# Unduh dan muat model dari Hugging Face
if models == 'CNN - Epochs = 5':
    model_path = hf_hub_download(repo_id="irulBES/model", filename="anit.h5")
    model = load_model(model_path)
    namaModel, epo = 'CNN', 5
elif models == 'CNN - Epochs = 50':
    model_path = hf_hub_download(repo_id="irulBES/model", filename="fadil.h5")
    model = load_model(model_path)
    namaModel, epo = 'CNN', 50
else:
    model_path = hf_hub_download(repo_id="irulBES/model", filename="anit.h5")
    model = load_model(model_path)
    namaModel, epo = 'Teachable Machine', 50

# Info model terpilih
st.write(f"**Model Terpilih:** {namaModel}")
st.write(f"**Epochs:** {epo}")

# Label klasifikasi
labels = ["Baik", "Bapak", "Ibu", "Maaf", "Saya"]

# Fungsi preprocessing (tanpa OpenCV)
def preprocess_image(img):
    """Preprocessing gambar untuk model CNN"""
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    img = ImageOps.fit(img, (224, 224))
    img_array = np.asarray(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Tabs: Upload File dan Kamera
tab1, tab2 = st.tabs(["📁 Upload File", "📷 Kamera"])

# ==== TAB 1: UPLOAD FILE ====
with tab1:
    uploaded_image = st.file_uploader("Silakan unggah gambar tangan Anda", type=["jpg", "png", "jpeg"])

    if uploaded_image is None:
        st.info('Menunggu gambar diunggah...')
    else:
        slot = st.empty()
        slot.text('🔍 Sedang memproses prediksi...')

        test_image = Image.open(uploaded_image)
        st.image(test_image, caption="🖼️ Gambar Input", width=400)

        # Preprocessing dan prediksi
        preprocessed_image = preprocess_image(test_image)
        with st.spinner("Menganalisis gestur..."):
            prediction = model.predict(preprocessed_image)

        # Hasil prediksi
        label = labels[np.argmax(prediction)]
        score = f"{np.max(prediction) * 100:.2f}"
        output = f'**Gestur Terdeteksi:** {label} \n\n**Tingkat Kepercayaan:** {score}%'

        slot.text('✅ Prediksi selesai!')
        st.success(output)

# ==== TAB 2: KAMERA ====
with tab2:
    st.write("Gunakan kamera untuk mengambil gambar gestur tangan.")
    st.info("📸 Klik tombol di bawah untuk mengambil gambar dari kamera Anda.")

    # Ambil gambar dari kamera (bisa dijalankan lokal)
    camera_image = st.camera_input("Ambil gambar dengan kamera")

    if camera_image is not None:
        st.text("🔍 Sedang memproses hasil...")

        # Buka gambar hasil kamera
        test_image = Image.open(camera_image)
        st.image(test_image, caption="Gambar dari Kamera", width=400)

        # Preprocessing dan prediksi
        preprocessed_image = preprocess_image(test_image)
        with st.spinner("Menganalisis gestur..."):
            prediction = model.predict(preprocessed_image)

        # Hasil prediksi
        label = labels[np.argmax(prediction)]
        score = f"{np.max(prediction) * 100:.2f}"
        output = f'**Gestur Terdeteksi:** {label} \n\n**Tingkat Kepercayaan:** {score}%'
        st.success(output)

    else:
        st.caption("⚠️ Jika kamera tidak muncul, jalankan aplikasi ini secara **lokal** dengan perintah:")
        st.code("streamlit run app.py", language="bash")



