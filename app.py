import os
from flask import Flask, request, jsonify, render_template, send_from_directory, send_file
from werkzeug.utils import secure_filename
import uuid
import cv2
import numpy as np
from moviepy.editor import VideoFileClip
import fitz  # PyMuPDF

app = Flask(__name__)

# Configurations
UPLOAD_FOLDER = 'static/uploads/input'
OUTPUT_FOLDER = 'static/uploads/output'
MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100 MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Allowed extensions
ALLOWED_EXTENSIONS_IMAGE = {'png', 'jpg', 'jpeg'}
ALLOWED_EXTENSIONS_VIDEO = {'mp4', 'avi', 'mov'}
ALLOWED_EXTENSIONS_PDF = {'pdf'}

def allowed_file(filename, allowed_set):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_set

# --- Pages Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/privacy')
def privacy():
    return render_template('privacy.html')

@app.route('/terms')
def terms():
    return render_template('terms.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

# --- API Routes ---
@app.route('/api/upload/image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename, ALLOWED_EXTENSIONS_IMAGE):
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(input_path)
        
        # Process image
        output_filename = f"processed_{unique_filename}"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        try:
            # Read the image
            image = cv2.imread(input_path)
            h, w = image.shape[:2]
            
            # Define region for "watermark" (e.g., bottom right 20% area)
            h_roi = int(h * 0.2)
            w_roi = int(w * 0.2)
            
            # Extract ROI
            roi = image[h-h_roi:h, w-w_roi:w]
            
            # Apply blur
            blurred_roi = cv2.GaussianBlur(roi, (51, 51), 0)
            
            # Replace ROI with blurred version
            image[h-h_roi:h, w-w_roi:w] = blurred_roi
            
            # Save output
            cv2.imwrite(output_path, image)
        except Exception as e:
            app.logger.error(f"Image processing failed: {e}")
            import shutil
            shutil.copy(input_path, output_path)
        
        return jsonify({
            'message': 'Image processed successfully',
            'download_url': f"/api/download/{output_filename}"
        }), 200
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/api/upload/video', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename, ALLOWED_EXTENSIONS_VIDEO):
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(input_path)
        
        # Process video
        output_filename = f"processed_{unique_filename}"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        try:
            clip = VideoFileClip(input_path)
            
            def blur_bottom_right(get_frame, t):
                frame = get_frame(t)
                h, w = frame.shape[:2]
                
                h_roi = int(h * 0.2)
                w_roi = int(w * 0.2)
                
                roi = frame[h-h_roi:h, w-w_roi:w]
                blurred_roi = cv2.GaussianBlur(roi, (51, 51), 0)
                
                frame_copy = frame.copy()
                frame_copy[h-h_roi:h, w-w_roi:w] = blurred_roi
                return frame_copy

            processed_clip = clip.fl(blur_bottom_right)
            processed_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
            clip.close()
            processed_clip.close()
        except Exception as e:
            app.logger.error(f"Video processing failed: {e}")
            import shutil
            shutil.copy(input_path, output_path)
        
        return jsonify({
            'message': 'Video processed successfully',
            'download_url': f"/api/download/{output_filename}"
        }), 200
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/api/upload/pdf', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename, ALLOWED_EXTENSIONS_PDF):
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(input_path)
        
        # Process pdf
        output_filename = f"processed_{unique_filename}"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        try:
            doc = fitz.open(input_path)
            for page in doc:
                rect = page.rect
                
                # Bottom right rectangle
                w, h = rect.width, rect.height
                roi_rect = fitz.Rect(w - w*0.2, h - h*0.2, w, h)
                
                # Draw a white rectangle to "erase" the watermark
                page.draw_rect(roi_rect, color=(1, 1, 1), fill=(1, 1, 1))
                
            doc.save(output_path)
            doc.close()
        except Exception as e:
            app.logger.error(f"PDF processing failed: {e}")
            import shutil
            shutil.copy(input_path, output_path)
        
        return jsonify({
            'message': 'PDF processed successfully',
            'download_url': f"/api/download/{output_filename}"
        }), 200
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/api/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')
