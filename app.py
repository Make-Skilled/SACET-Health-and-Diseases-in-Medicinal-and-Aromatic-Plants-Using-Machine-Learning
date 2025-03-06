from flask import Flask, request, render_template, jsonify, redirect, send_from_directory, url_for, flash, session
from werkzeug.security import generate_password_hash, check_password_hash
from keras.models import load_model
import numpy as np
from PIL import Image, ImageOps
import sqlite3
import os
from plant_data import plant_data  # Import plant_data
import os
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Change this to something secure

# Set up the path for the SQLite database
DATABASE = 'database.db'

# Load the Keras model and labels
model = load_model("model/keras_Model.h5", compile=False)

with open("model/labels.txt", "r") as f:
    class_names = f.readlines()

# Path to the folder where uploaded images will be saved
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Make sure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Function to check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Database setup - Create table if not exists
def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

# Initialize the database and create necessary tables if they do not exist
def init_db():
    with get_db() as conn:
        cursor = conn.cursor()
        
        # Create users table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL
        )
        """)

        # Create scan history table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS scan_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            class_name TEXT NOT NULL,
            confidence_score REAL NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            image TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        """)

        conn.commit()

# Home page (index route)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# Sign-up page
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        # Hash the password before saving it
        password_hash = generate_password_hash(password)
        
        try:
            with get_db() as conn:
                cursor = conn.cursor()
                cursor.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
                               (username, email, password_hash))
                conn.commit()
            
            flash('Account created successfully! You can now log in.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Email already exists. Please use another email.', 'error')

    return render_template('signup.html')

# Login page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        try:
            with get_db() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
                user = cursor.fetchone()

            if user and check_password_hash(user['password'], password):
                session['user_id'] = user['id']  # Store user ID in session
                flash('Login successful!', 'success')
                return redirect(url_for('dashboard'))  # Redirect to dashboard
            else:
                flash('Invalid email or password. Please try again.', 'error')
        except Exception as e:
            flash(f'Error logging in: {str(e)}', 'error')

    return render_template('login.html')

# Dashboard page (Leaf Diagnosis)
@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    result = {
        'class': 'No prediction yet',
        'confidence_score': None,
        'plant_info': {}
    }

    if 'user_id' not in session:
        flash('Please log in to access the dashboard.', 'info')
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'error')
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            try:
                # Save the image to the upload folder
                filename = secure_filename(file.filename)
                file_path = os.path.join(UPLOAD_FOLDER, filename)
                file.save(file_path)

                # Process the uploaded image
                image = Image.open(file_path).convert("RGB")
                size = (224, 224)
                image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
                image_array = np.asarray(image)
                normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
                data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
                data[0] = normalized_image_array

                # Prediction
                prediction = model.predict(data)
                index = np.argmax(prediction)
                class_name = class_names[index].strip()
                confidence_score = prediction[0][index] * 100  # Convert to percentage

                # Store scan result in database
                with get_db() as conn:
                    cursor = conn.cursor()
                    cursor.execute("""INSERT INTO scan_history (user_id, class_name, confidence_score,image) 
                                      VALUES (?, ?, ?,?)""", (session['user_id'], class_name, confidence_score,f'{filename}'))
                    conn.commit()

                # Retrieve plant information
                plant_info = plant_data.get(class_name, {})
                result = {
                    'class': class_name,
                    'confidence_score': confidence_score,
                    'plant_info': plant_info
                }

            except Exception as e:
                flash(f'Error in prediction: {str(e)}', 'error')
                return redirect(request.url)
        else:
            flash('Invalid file type. Only JPG, PNG, JPEG, and GIF are allowed.', 'error')
            return redirect(request.url)

    return render_template('dashboard.html', result=result)
# History page (Show past scans)


@app.route('/history')
def history():
    if 'user_id' not in session:
        flash('Please log in to view your scan history.', 'info')
        return redirect(url_for('login'))

    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT class_name, confidence_score, timestamp ,image
            FROM scan_history 
            WHERE user_id = ? 
            ORDER BY timestamp DESC
        """, (session['user_id'],))
        
        scan_results = cursor.fetchall()

    return render_template('history.html', scans=scan_results)

# Logout user
@app.route('/logout')
def logout():
    session.clear()  # Clear the session
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))

# Main function to run the app
if __name__ == '__main__':
    init_db()  # Initialize the database and create the table if not exists
    app.run(debug=True)