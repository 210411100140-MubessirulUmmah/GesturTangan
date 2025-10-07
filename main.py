import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
from PIL import Image
from huggingface_hub import hf_hub_download

# Create a Streamlit app
st.title("Deteksi Gestur Tangan BISINDO")
st.header("Aplikasi web untuk mengklasifikasikan gestur tangan Bahasa Isyarat Indonesia (BISINDO) dari Maaf, Baik, Saya, Ibu, dan Bapak")
st.write("Aplikasi ini menggunakan model Convolutional Neural Network (CNN) yang dilatih dengan kerangka kerja Teachable Machine dan pustaka MediaPipe untuk mendeteksi dan mengenali isyarat tangan dari BISINDO. Aplikasi ini dapat mengklasifikasikan 5 isyarat tangan dari Maaf, Baik, Saya, Ibu, dan Bapak")

# Initialize Mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Choose the model
models = st.selectbox(
    'Pilih Model',
    ('CNN - Epochs = 5', 'CNN - EPOCHS = 50', 'Teachable Machine')
)

model = None
namaModel = None
epo = None

if models == 'CNN - Epochs = 5':
    model = load_model("irulBES/model","anit.h5")
    namaModel = 'CNN'
    epo = 5
elif models == 'CNN - EPOCHS = 50':
    model = load_model("irulBES/model","fadil.h5")
    namaModel = 'CNN'
    epo = 50
elif models == 'Teachable Machine':
    model = load_model("irulBES/model","anit.h5")
    namaModel = 'Teachable Machine'
    epo = 50

st.write('Model Terpilih :', namaModel)
st.write("Epochs = ", epo)

# Define the labels
labels = ["Baik", "Bapak", "Ibu", "Maaf", "Saya"]

st.subheader("Prediksi")
tab1, tab2 = st.tabs(["Upload File", "Camera"])

def preprocess_image(img):
    # Convert the image from RGBA to RGB if necessary
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    # Convert the image to a numpy array
    img_array = np.array(img)
    # Resize the image
    img_array = cv2.resize(img_array, (224, 224))
    # Normalize the image
    img_array = img_array / 255.0
    # Expand dimensions to match the input shape of the model (batch size of 1)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

with tab1:
    uploaded_image = st.file_uploader("Silahkan Unggah Gambar Tangan Anda", type=["jpg", "png", "jpeg"])
    if uploaded_image is None:
        st.text('Menunggu Gambar Diunggah....')
    else:
        slot = st.empty()
        slot.text('Sedang Memprediksi....')

        test_image = Image.open(uploaded_image)
        st.image(uploaded_image, caption="Input Image", width=400)
        # Preprocess the image
        preprocessed_image = preprocess_image(test_image)
        # Predict the label
        prediction = model.predict(preprocessed_image)
        label = labels[np.argmax(prediction)]
        score = format(np.max(prediction) * 100, '.2f')
        output = f'Label yang diprediksi adalah {label} dengan Skor kepercayaan : {score}%'
        slot.text('Prediksi Selesai!')
        st.success(output)

with tab2:
    st.subheader("Webcam Feed")
    st.write("Izinkan aplikasi untuk mengakses kamera Anda")

    mulai = st.button('Mulai Kamera')

    if mulai:
        # Create a placeholder for the webcam feed
        placeholder = st.empty()
        # Open the webcam
        cap = cv2.VideoCapture(0)
        # Loop over the frames
        while True:
            # Read a frame
            success, frame = cap.read()
            if not success:
                break
            # Flip the frame horizontally
            frame = cv2.flip(frame, 1)
            # Convert the frame to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Get the height and width of the frame
            h, w, _ = frame.shape
            # Copy the frame
            frame_copy = frame.copy()
            # Process the frame with Mediapipe
            results = hands.process(frame)
            # Check if a hand is detected
            if results.multi_hand_landmarks:
                # Get the hand landmarks
                hand_landmarks = results.multi_hand_landmarks[0]
                # Draw the hand landmarks
                mpDraw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)
                # Get the bounding box coordinates
                x_min = int(min([landmark.x for landmark in hand_landmarks.landmark]) * w)
                x_max = int(max([landmark.x for landmark in hand_landmarks.landmark]) * w)
                y_min = int(min([landmark.y for landmark in hand_landmarks.landmark]) * h)
                y_max = int(max([landmark.y for landmark in hand_landmarks.landmark]) * h)
                # Crop the hand region
                hand_region = frame_copy[y_min:y_max, x_min:x_max]
                # Check if the hand region is not empty
                if hand_region.size != 0:
                    # Resize the hand region
                    hand_region = cv2.resize(hand_region, (224, 224))
                    # Preprocess the hand region
                    hand_region = hand_region / 255.0
                    hand_region = np.expand_dims(hand_region, axis=0)
                    # Predict the label
                    prediction = model.predict(hand_region)
                    label = labels[np.argmax(prediction)]
                    # Display the label and the confidence score
                    cv2.putText(frame, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    cv2.putText(frame, f"{np.max(prediction):.2f}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            # Update the webcam feed
            placeholder.image(frame, channels="RGB")
        # Release the webcam
        cap.release()
