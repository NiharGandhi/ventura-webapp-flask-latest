import os
import cv2
import firebase_admin
from firebase_admin import credentials, storage
from flask import Flask, render_template, request, send_file, jsonify
import face_recognition
import pickle
import shutil

# Initialize Flask app
app = Flask(__name__)

# Initialize Firebase Admin SDK
if not firebase_admin._apps:
    cred = credentials.Certificate('ventura-5d1fe-firebase-adminsdk-q6x4i-2a488de72f.json')
    firebase_admin.initialize_app(cred, {
        'storageBucket': 'ventura-5d1fe.appspot.com'
    })

# Function to capture and save the face images
def capture_face_images(class_name, num_images):
    bucket = storage.bucket()
    captured_images = []

    # Initialize camera
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while len(captured_images) < num_images:
        ret, frame = camera.read()

        if ret:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                cropped_frame = frame[y:y + h, x:x + w]
                image_path = f'{class_name}_{len(captured_images)}.jpg'
                cv2.imwrite(image_path, cropped_frame)

                blob_name = f'ImagesAttendance/{class_name}/{class_name}_{len(captured_images)}.jpg'
                blob = bucket.blob(blob_name)
                blob.upload_from_filename(image_path)
                os.remove(image_path)

                captured_images.append(cropped_frame)

    camera.release()

    return captured_images

# Function to download 'ImagesAttendance' data from Firebase Storage
def download_images_from_firebase():
    print('Downloading images from Firebase')
    bucket = storage.bucket()
    blobs = bucket.list_blobs(prefix='ImagesAttendance/')
    num_downloaded_images = 0

    for blob in blobs:
        if blob.name.endswith('/'):
            continue

        # Extract the class name from the blob name
        class_name = os.path.dirname(blob.name).split('/')[-1]
        os.makedirs(os.path.join('ImageData', class_name), exist_ok=True)

        image_name = os.path.basename(blob.name)
        destination_blob_name = os.path.join('ImageData', class_name, image_name)
        blob.download_to_filename(destination_blob_name)
        num_downloaded_images += 1

    return num_downloaded_images

# Function to train encoding file
def train_encoding_file():
    folderpath = 'ImageData'
    encode_list = []
    class_names = []

    def process_images(file_path, label):
        image = cv2.imread(file_path)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect faces in the image
        face_locations = face_recognition.face_locations(rgb_image)

        if len(face_locations) == 1:
            # If only one face is detected, extract its encoding
            face_encoding = face_recognition.face_encodings(rgb_image, face_locations)[0]
            encode_list.append(face_encoding)
            class_names.append(label)

    for subfolder in os.listdir(folderpath):
        subfolder_path = os.path.join(folderpath, subfolder)
        if os.path.isdir(subfolder_path):
            for root, _, files in os.walk(subfolder_path):
                for file_name in files:
                    file_path = os.path.join(root, file_name)
                    process_images(file_path, subfolder)

    # Prepare the data in the required format
    data = {'encodings': encode_list, "names": class_names}
    encoding_file = 'face_enc'

    # Save the data to a file for later use
    with open(encoding_file, 'wb') as f:
        pickle.dump(data, f)

    # Delete the 'ImageData' folder and its contents
    shutil.rmtree(folderpath)

    return encoding_file


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/capture_images', methods=['POST'])
def capture_images():
    class_name = request.form.get('name')
    num_images = 10  # You can change this number as per your requirement
    capture_face_images(class_name, num_images)
    return jsonify({'message': 'Images captured successfully.'})

@app.route('/train_encoding', methods=['POST'])
def train_encoding():
    download_images_from_firebase()
    train_encoding_file()
    return jsonify({'message': 'Encoding file trained and saved successfully.'})

if __name__ == '__main__':
    app.run(debug=False)