from flask import flash, Flask, render_template, request
from preprocess import data_prep
import time



app = Flask(__name__)


@app.route('/', methods=['GET'])
def homepage():
    return render_template('content.html', data = data_prep.get_all_path())

@app.route('/', methods=['POST'])
def image_upload():
    status, message = data_prep.verify_image_upload()
    # file uploaded doesn't met the requirement
    if not status:
        return render_template("content.html", data=data_prep.get_all_path())
    # save
    data_prep.save_image_to_upload(image_to_save=request.files['uploadFile'], extension=message)
    return render_template("content.html", data=data_prep.get_all_path())




@app.route('/flash', methods=["GET"])
def render_flash_test():
    return render_template("test.html")

@app.route('/flash', methods=["POST"])
def flash_test():
    if 'uploadFile' not in request.files:
        print("Hellow")
    uploaded_file = request.files['uploadFile']
    if uploaded_file.filename == '':
        flash("nibba")

    return render_template('test.html')


if __name__ == "__main__":
    # create folder check (if none, create)
    status = False
    if not status:
        print("HELLO")
    # app.config['UPLOAD_FOLDER'] = image_editor.get_upload_path()
    # app.secret_key = "testdoang"
    # app.run(debug=True)

"""
TO-DO:

[ ] implement load model
[ ] change model

Extra stuff.
[ ] make 
[ ] message flash
"""
