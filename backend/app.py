from flask import Flask, request, jsonify, render_template
import os
import subprocess

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists(RESULTS_FOLDER):
    os.makedirs(RESULTS_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER

def run_yolov7(video_path, result_path):
    yolov7_command = [
        'python', 'Yolov7_StrongSORT_OSNet/track_fixed_allin.py',
        '--img-size', '1376',
        '--yolo-weights', 'yolov7/weights/light-box/weights/best.pt',
        '--source', video_path,
        '--device', '2',
        '--show-vid',
        '--save-txt',
        '--save-vid'
    ]
    subprocess.run(yolov7_command, check=True)
    generated_txt = os.path.splitext(video_path)[0] + '.txt'
    os.rename(generated_txt, result_path)

@app.route('/')
def home():
    return 'Welcome to the Zebrafish Analysis API. Please use <a href="/upload">/upload</a> endpoint to upload videos for analysis.'

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        
        result_path = os.path.join(app.config['RESULTS_FOLDER'], file.filename + '.txt')
        
        try:
            run_yolov7(file_path, result_path)
        except subprocess.CalledProcessError as e:
            return jsonify({'error': 'Error during YOLOv7 prediction', 'details': str(e)}), 500
        
        return jsonify({'message': 'YOLOv7 prediction completed', 'result_path': result_path})
    
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

if __name__ == '__main__':
    app.run(debug=True)
