from flask import Flask, render_template, request, redirect, url_for
import os
import uuid
import numpy as np
import tensorflow as tf
import cv2
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model

# Set TensorFlow to run eagerly for easier debugging (optional but helpful)
tf.config.run_functions_eagerly(True)

app = Flask(__name__)

# --- Configuration ---
UPLOAD_FOLDER = 'static/uploads'
HEATMAP_FOLDER = 'static/heatmaps'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['HEATMAP_FOLDER'] = HEATMAP_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(HEATMAP_FOLDER, exist_ok=True)

# --- Load Model ---
try:
    model = load_model('diabetic_retinopathy_model.h5')
except Exception as e:
    print(f"Error loading model: {e}")
    model = None 


# --- Helper Function ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# --- Routes ---
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        print(f"New user registered: {username}, {email}")
        return redirect(url_for('login'))
    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        print(f"Login attempt from: {email}")
        return redirect(url_for('predict'))
    return render_template('login.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if model is None:
        return "Model not loaded. Please check the 'diabetic_retinopathy_model.h5' file.", 500

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            # --- Save Uploaded Image ---
            unique_name = str(uuid.uuid4()) + "_" + secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
            file.save(filepath)

            # --- Preprocess Image ---
            img = tf.keras.preprocessing.image.load_img(filepath, target_size=(224, 224))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0).astype(np.float32) / 255.0
            img_tensor = tf.convert_to_tensor(img_array)

            # --- Model Prediction ---
            preds = model.predict(img_tensor)
            predicted_class = int(np.argmax(preds[0]))
            confidence = round(float(np.max(preds[0])) * 100, 2)

            class_labels = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
            prediction = class_labels[predicted_class]

            # --- Grad-CAM Heatmap Setup ---
            last_conv_layer = None
            for layer in reversed(model.layers):
                if isinstance(layer, tf.keras.layers.Conv2D):
                    last_conv_layer = layer.name
                    break

            if not last_conv_layer:
                return "Error: No Conv2D layer found in the model for Grad-CAM.", 500

            grad_model = tf.keras.models.Model(
                [model.inputs],
                [model.get_layer(last_conv_layer).output, model.output]
            )


            index_coords = tf.constant([[predicted_class]], dtype=tf.int32) 

            # --- Grad-CAM Calculation ---
            with tf.GradientTape() as tape:
                tape.watch(img_tensor)
                
                conv_outputs, predictions_tensor_batch = grad_model(img_tensor)
                

                predictions_vector = tf.squeeze(predictions_tensor_batch) 
                

                loss = tf.gather_nd(predictions_vector, index_coords)
                
            grads = tape.gradient(loss, conv_outputs)
            
            # --- Error Handling and Post-Processing ---
            if grads is None:
                print("Error: Gradients are None. Grad-CAM failed.")
                return render_template(
                    'prediction.html',
                    prediction=prediction,
                    confidence=confidence,
                    image_path=filepath,
                    heatmap_path=None
                )
            

            conv_outputs_np = tf.squeeze(conv_outputs).numpy()
            grads_np = tf.squeeze(grads).numpy()


            if len(conv_outputs_np.shape) != 3 or len(grads_np.shape) != 3:
                return "Error: Could not correctly shape model outputs for heatmap generation (expected 3D).", 500

            pooled_grads = np.mean(grads_np, axis=(0, 1))
            heatmap = np.mean(conv_outputs_np * pooled_grads, axis=-1)
            heatmap = np.maximum(heatmap, 0)
            heatmap /= np.max(heatmap) if np.max(heatmap) != 0 else 1

            # --- Overlay heatmap ---
            img_original = cv2.imread(filepath)
            img_original = cv2.resize(img_original, (224, 224))
            heatmap_resized = cv2.resize(heatmap, (img_original.shape[1], img_original.shape[0]))
            heatmap_resized = np.uint8(255 * heatmap_resized)
            heatmap_resized = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
            superimposed_img = cv2.addWeighted(img_original, 0.6, heatmap_resized, 0.4, 0)

            heatmap_filename = f"heatmap_{unique_name}"
            heatmap_path = os.path.join(app.config['HEATMAP_FOLDER'], heatmap_filename)
            cv2.imwrite(heatmap_path, superimposed_img)

            return render_template(
                'prediction.html',
                prediction=prediction,
                confidence=confidence,
                image_path=filepath,
                heatmap_path=heatmap_path.replace('\\', '/') 
            )

    return render_template('prediction.html')


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)