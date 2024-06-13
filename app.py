from flask import Flask, render_template, request, send_from_directory
import os
from model.sr_model import generate_super_resolution
from utils.image_utils import save_image, load_image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

@app.route('/', methods=['GET', 'POST'])
def index():
    print('Request method:', request.method)
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            print('File uploaded successfully')
            # 生成高分辨率图片
            hr_image_path = generate_super_resolution(filepath)
            return render_template('index.html', hr_image=hr_image_path, lr_image=filepath)
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)