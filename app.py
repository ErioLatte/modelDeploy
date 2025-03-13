from flask import flash, Flask, render_template, request
from preprocess import image_editor
import time



app = Flask(__name__)


@app.route('/', methods=['GET'])
def hello_world():
    return render_template('content.html', data = image_editor.get_all_path())

@app.route('/', methods=['POST'])
def image_upload():
    # check if the post request has the file part
    if 'uploadFile' not in request.files: # because in HTMl input name is uploadFile
        print("NO FILE PATH")
        return render_template("content.html", data = image_editor.get_all_path())
    uploaded_file = request.files['uploadFile']
    # If the user does not select a file, the browser submits an
    # empty file without a filename.
    if uploaded_file.filename == '':
        print("NO FILE SELECTED")
        return render_template("content.html", data = image_editor.get_all_path())
    
    # check file extension is {'png', 'jpg', 'jpeg'}, else return null
    file_extension = image_editor.verify_image(uploaded_file.filename)
    # file is not within extension 
    if not file_extension:
        # TO-DO give response to in the web (flash)
        return render_template('content.html', data = image_editor.get_all_path())


    now = time.time()
    filename = f"{int(float(now)*10000)}_up"
    image_path = f"./static/upload/{filename}.{file_extension}"
    print("saving file in upload")
    uploaded_file.save(image_path)

    # image_editor.set_image_path(image_path=image_path)


    return render_template("content.html", data=image_editor.get_all_path())

def process_image_input():
    pass

def process_text_input():
    pass

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
    app.config['UPLOAD_FOLDER'] = image_editor.get_upload_path()
    app.secret_key = "testdoang"
    app.run(debug=True)

"""
TO-DO:

1. ganti upload file format
2. implement model load
3. implement canny
4. fix gui
5. 
Keyword arguments:
argument -- description
Return: return_description
"""
