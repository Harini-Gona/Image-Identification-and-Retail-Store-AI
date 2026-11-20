from flask import Flask, render_template, request
import cv2
import numpy as np
import os

app = Flask(__name__)
def try_on(user_img_path, cloth_img_path):
    person = cv2.imread(user_img_path)
    cloth = cv2.imread(cloth_img_path, cv2.IMREAD_UNCHANGED)

    h, w = person.shape[:2]

    body_top = int(h * 0.25)
    body_bottom = int(h * 0.65)
    body_left = int(w * 0.25)
    body_right = int(w * 0.75)

    body_width = body_right - body_left
    body_height = body_bottom - body_top

    cloth_resized = cv2.resize(cloth, (body_width, body_height))

    if cloth_resized.shape[2] == 4:
        alpha = cloth_resized[:, :, 3] / 255.0
        for c in range(3):
            person[body_top:body_bottom, body_left:body_right, c] = (
                alpha * cloth_resized[:, :, c] +
                (1 - alpha) * person[body_top:body_bottom, body_left:body_right, c]
            )
    else:
        person[body_top:body_bottom, body_left:body_right] = cloth_resized

    output_path = "static/output.png"
    cv2.imwrite(output_path, person)

    return output_path

@app.route('/tryon', methods=['POST'])
def tryon_api():
    user_img = request.files['user_img']
    cloth_name = request.form['cloth_img']

    user_path = os.path.join("static/uploads", user_img.filename)
    user_img.save(user_path)

    # extract filename from URL/path
    cloth_filename = os.path.basename(cloth_name)
    cloth_path = os.path.join("static/clothes", cloth_filename)

    output = try_on(user_path, cloth_path)

    return open(output, 'rb').read(), 200, {'Content-Type': 'image/png'}

if __name__ == '__main__':
    app.run(debug=True)