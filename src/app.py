from flask import Flask, render_template, request, redirect, jsonify, session
from predictor import TritonInferenceServer
import os

app = Flask(__name__)

app.secret_key = 'my_secret_key'

# Folder where uploaded images will be stored
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed file extensions for images
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle image upload
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        # Save the file to the upload folder
        session['filename'] = file.filename

        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        return render_template('index.html', filename=file.filename)
    return redirect(request.url)

# Route to handle image captioning
@app.route('/generate_caption', methods=['POST'])
def generate_caption():
    client = TritonInferenceServer(model_name="imagecaptioning", port="8000")

    generated_caption = "No caption was generated"
    if "filename" in session:
        generated_caption = client.predict(local_image_path=os.path.join(UPLOAD_FOLDER, session["filename"]))
    
    return jsonify({"caption" : generated_caption})

if __name__ == '__main__':
    app.run(debug=True)
