from flask import Flask, render_template, request, send_from_directory, jsonify
import os
from model.sr_model import generate_super_resolution
import threading

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# 定义一个锁
lock = threading.Lock()

@app.route('/', methods=['GET', 'POST'])
def index():
    global lock
    if request.method == 'POST':
        if lock.locked():
            # 如果锁已经被占用，返回繁忙信息的JSON
            return jsonify({'status': 'locked', 'message': '服务访问繁忙，请稍后再试，点击任意区域返回。'})
        else:
            with lock:  # 锁定资源
                if 'file' not in request.files:
                    return jsonify({'status': 'error', 'message': 'No file part'})
                file = request.files['file']
                if file.filename == '':
                    return jsonify({'status': 'error', 'message': 'No selected file'})
                if file:
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                    file.save(filepath)
                    print('File uploaded successfully')
                    # 生成高分辨率图片
                    hr_image_path = generate_super_resolution(filepath)
                    # 返回图片路径的JSON
                    return jsonify({'status': 'success', 'lr_image': filepath, 'hr_image': hr_image_path})
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)