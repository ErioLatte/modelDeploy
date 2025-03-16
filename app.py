from flask import flash, Flask, render_template, request
from preprocess import data_prep
from model import model_loader
import time



app = Flask(__name__)


@app.route('/', methods=['GET'])
def homepage():
    return render_template('content.html', data = data_prep.get_all_path())


def generate(prompt):
    data_prep.preprocess()
    output_path = model_loader.generate_image(
        canny_image_path=data_prep.canny_image_path,
        prompt=prompt,
        upload_dir=data_prep.get_upload_folder(),
        image_name=data_prep.get_curr_image_name(),
        extension=data_prep.get_curr_image_extension()
    )
    data_prep.result_image_path = output_path


@app.route('/', methods=['POST'])
def image_upload():
    image_status = data_prep.verify_image_upload(request)

    # file uploaded doesn't met the requirement
    if not image_status["status"]:
        return render_template("content.html", data=data_prep.get_all_path())
    
    # save
    data_prep.save_image_to_upload(image_to_save=request.files['uploadFile'], extension=image_status["message"])
    
    text_prompt = request.form['textPrompt']
    # in case  of something
    if text_prompt is None:
        text_prompt = ""
    
    generate(text_prompt)

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
        flash("nofile")

    return render_template('test.html')


if __name__ == "__main__":
    # create folder check (if none, create)
    app.config['UPLOAD_FOLDER'] = data_prep.get_upload_path()
    app.secret_key = "testdoang"
    app.run(debug=True)

"""
TO-DO:

[ ] implement load model
[ ] change model
[ ] implement logger for easy read
[ ] make load model a button (easier debug purpose :)

Extra stuff.
[ ] make 
[ ] message flash
"""
