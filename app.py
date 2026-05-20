import os
import re
import json
import sqlite3
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import numpy as np

# TensorFlow imports
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Use environment variable for secret key in production
app.secret_key = os.environ.get('SECRET_KEY', 'dev-key-for-local-only')

# Database path
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "database.db")

# Load model - UPDATED PATH
model_path = 'models/leaf_model.keras'
if os.path.exists(model_path):
    model = load_model(model_path, compile=False)
else:
    # Fallback for backward compatibility
    model = load_model('static/models/leaf_model.keras', compile=False)
    print("Warning: Using old model path. Move model to 'models/' folder.")

# File upload settings
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Regex patterns
EMAIL_REGEX = r'^[\w\.-]+@[\w\.-]+\.\w{2,4}$'
USERNAME_REGEX = r'^[A-Za-z][A-Za-z0-9_ ]{4,30}$'
PASSWORD_REGEX = r'^(?=.*[A-Za-z])(?=.*\d)[A-Za-z\d@#$%^&+=!]{8,}$'

# Initialize DB
def init_db():
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            email TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL,
            profile TEXT NOT NULL
        )
        """)

# Load plant data from JSON file
try:
    with open('medplants.json', encoding='utf-8') as f:
        plant_data = json.load(f)
except FileNotFoundError:
    print("Warning: medplants.json not found")
    plant_data = []

# List of class names (must match model training order)
class_names = [
    'Aloevera', 'Amla', 'Amruthaballi', 'Ashoka', 'Ashwagandha',
    'Astma_weed', 'Avacado', 'Badipala', 'Balloon_Vine', 'Bamboo',
    'Basale', 'Beans', 'Betel', 'Betel_Nut', 'Bhrami',
    'Bringaraja', 'Camphor', 'Castor', 'Catharanthus', 'Cathedral Bells',
    'Chakte', 'Chilly', 'Citron lime (herelikai)', 'Coffee', 'Common rue(naagdalli)',
    'Coriander', 'Curry', 'Doddpathre', 'Drumstick', 'Ekka',
    'Eucalyptus', 'Ganigale', 'Ganike', 'Gasagase', 'Geranium',
    'Ginger', 'Globe Amarnath', 'Guava', 'Henna', 'Hibiscus',
    'Honge', 'Insulin', 'Jackfruit', 'Jasmine', 'Kambajala',
    'Kasambruga', 'Kohlrabi', 'Lantana', 'Lemon', 'Lemongrass',
    'Malabar_Nut', 'Malabar_Spinach', 'Mango', 'Marigold', 'Mint',
    'Nagadali', 'Neem', 'Nelavembu', 'Nerale', 'Nithyapushpa',
    'Nooni', 'Onion', 'Padri', 'Palak(Spinach)', 'Papaya',
    'Parijatha', 'Pea', 'Peepal', 'Pepper', 'Pomoegranate',
    'Pumpkin', 'Raddish', 'Raktachandini', 'Rose', 'Sampige',
    'Sapota', 'Seethaashoka', 'Seethapala', 'Tamarind', 'Taro',
    'Tecoma', 'Thumbe', 'Tomato', 'Tulsi', 'Turmeric',
    'Wood_sorel', 'banana', 'kamakasturi', 'karanj', 'kepala'
]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_plant(img_path):
    try:
        img = image.load_img(img_path, target_size=(256, 256))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) 
        img_array = img_array / 255.0
        prediction = model.predict(img_array)
        class_index = np.argmax(prediction)
        return class_names[class_index]
    except Exception as e:
        print(f"Prediction error: {e}")
        return None

# ROUTES
@app.route("/")
def home_redirect():
    return redirect("/index")

@app.route("/index")
def index():
    return render_template("index.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"].strip()
        email = request.form["email"].strip()
        password = request.form["password"].strip()
        profile = request.form["profile"].strip()

        if not re.match(USERNAME_REGEX, username):
            return redirect("/register?message=Invalid+Username")
        if not re.match(EMAIL_REGEX, email):
            return redirect("/register?message=Invalid+Email")
        if not re.match(PASSWORD_REGEX, password):
            return redirect("/register?message=Weak+Password")
        if profile not in ["student", "doctor", "botanist"]:
            return redirect("/register?message=Invalid+Profile")

        hashed_password = generate_password_hash(password)
        try:
            with sqlite3.connect(DB_PATH) as con:
                con.execute("INSERT INTO users (username, email, password, profile) VALUES (?, ?, ?, ?)",
                            (username, email, hashed_password, profile))
                con.commit()
                return redirect(f"/login?message=Registration+Successful&user={username}")
        except sqlite3.IntegrityError:
            return redirect("/register?message=Email+already+registered")
    return render_template("register.html")

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password'].strip()

        if not username or not password:
            return redirect("/login?message=Please+fill+all+fields")

        with sqlite3.connect(DB_PATH) as con:
            cur = con.cursor()
            cur.execute("SELECT * FROM users WHERE username = ?", (username,))
            user = cur.fetchone()

            if user and check_password_hash(user[3], password):
                session['username'] = user[1]  # Set username in session
                session['profile'] = user[4]   # Set profile in session
                return redirect("/home")
            else:
                return redirect("/login?message=Invalid+Credentials")

    return render_template('login.html')

@app.route('/home')
def home():
    if 'username' not in session:
        return redirect('/login')
    return render_template('home.html', user=session.get('username'))

@app.route('/analysis')
def analysis():
    if 'username' not in session:
        return redirect('/login')
    return render_template('analysis.html', user=session.get('username'))

@app.route('/guidance')
def guidance():
    if 'username' not in session:
        return redirect('/login')
    return render_template('guidance.html', user=session.get('username'))

@app.route('/test_prediction', methods=['GET', 'POST'])
def test_prediction():
    if 'username' not in session:
        return redirect('/login')
        
    if request.method == 'POST':
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # Create uploads folder if it doesn't exist
            if not os.path.exists(app.config['UPLOAD_FOLDER']):
                os.makedirs(app.config['UPLOAD_FOLDER'])
            
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            try:
                predicted_plant = predict_plant(file_path)
                
                if predicted_plant is None:
                    return jsonify({'error': 'Prediction failed'}), 500

                # Get additional plant information
                plant_info = next((item for item in plant_data if item.get('A', '').lower() == predicted_plant.lower()), None)

                if plant_info is None:
                    return jsonify({
                        'plantName': predicted_plant,
                        'plantInfo': "No additional information available."
                    })

                return jsonify({
                    'plantName': predicted_plant,
                    'plantInfo': plant_info
                })
            except Exception as e:
                print(f"Error during prediction: {e}")
                return jsonify({'error': str(e)}), 500
            finally:
                # Clean up uploaded file after prediction
                if os.path.exists(file_path):
                    os.remove(file_path)

        return jsonify({'error': 'Invalid file type'}), 400

    return render_template('test_prediction.html', user=session.get('username'))

@app.route('/table')
def table():
    if 'username' not in session:
        return redirect('/login')
    return render_template('table.html', user=session.get('username'))

@app.route("/logout")
def logout():
    session.pop("username", None)
    session.pop("profile", None)
    return redirect("/index")

# Initialize on startup
if __name__ != "__main__":
    # For production (Gunicorn)
    init_db()
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

if __name__ == "__main__":
    # For local development
    init_db()
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)