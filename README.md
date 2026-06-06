A deep learning web application that detects Diabetic Retinopathy from retinal fundus images and classifies it into 5 severity grades using Xception (Transfer Learning) with Grad-CAM explainability.

🔍 What is Diabetic Retinopathy?
Diabetic Retinopathy (DR) is a diabetes complication that affects the eyes. Early detection is critical to prevent vision loss. This system automates the detection process using AI.

🎯 Severity Grades
GradeClassDescription0No DRNo signs of retinopathy1MildMinor microaneurysms2ModerateMore severe than mild3SevereExtensive retinal damage4Proliferative DRMost advanced stage

🚀 Features

📸 Upload retinal fundus image and get instant prediction
🔥 Grad-CAM heatmap visualization showing which region the model focused on
⚖️ Handles class imbalance using balanced class weights
🌐 Clean Flask web interface with Login/Register pages
💾 Model trained on preprocessed fundus image dataset


🛠️ Tech Stack

Model: Xception (pretrained on ImageNet) + Fine-tuning (last 60 layers)
Framework: TensorFlow / Keras
Backend: Flask (Python)
Frontend: HTML, CSS, JavaScript
Explainability: Grad-CAM
Data Augmentation: ImageDataGenerator (rotation, zoom, flip)


📁 Project Structure
fundus-ai/
│
├── static/
│   ├── css/          # Stylesheets
│   ├── js/           # JavaScript files
│   └── images/       # Static images
│
├── templates/
│   ├── index.html        # Home page
│   ├── login.html        # Login page
│   ├── register.html     # Register page
│   └── prediction.html   # Prediction result page
│
├── app1.py               # Flask application
├── requirements.txt      # Python dependencies
└── .gitignore

⚙️ How to Run Locally
1. Clone the repository
bashgit clone https://github.com/deevihimavanthsai-0513/fundus-ai.git
cd fundus-ai
2. Create virtual environment
bashpython -m venv venv
venv\Scripts\activate        # Windows
# or
source venv/bin/activate     # Mac/Linux
3. Install dependencies
bashpip install -r requirements.txt
4. Download the trained model

The model file diabetic_retinopathy_model.h5 is too large for GitHub.
📥 Download from Google Drive
https://drive.google.com/file/d/1UsLvMM0cV_tJRD7rnZB_QH7OPOl3c5-4/view?usp=sharing

5. Run the app
bashpython app1.py
Visit http://127.0.0.1:5000 in your browser.

📊 Model Performance
MetricValueTest Accuracy75–80%Base ModelXception (ImageNet)Fine-tuned LayersLast 60 layersInput Size224 × 224Classes5

🔥 Grad-CAM Visualization
The model uses Gradient-weighted Class Activation Mapping (Grad-CAM) to highlight which regions of the retina influenced the prediction — making the AI decision explainable and trustworthy for clinical use.

📦 Requirements
tensorflow
flask
numpy
opencv-python
matplotlib
scikit-learn
seaborn
Install all with:
bashpip install -r requirements.txt

👨‍💻 Author
Deevi Himavanth Sai

📧 deevihimavanthsai@gmail.com
🐙 GitHub
💼 LinkedIn
www.linkedin.com/in/himavanth-sai-deevi-014004264



