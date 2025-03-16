import os
import cv2
import time
from PIL import Image
 
class DataPrep():
    def __init__(self):
        self.up_image_path = "./static/placeholder.jpg"
        self.canny_image_path = "./static/placeholder1.jpg"
        self.result_image_path = "./static/placeholder2.jpg"


        # tracker
        self._curr_img_name = None
        self._curr_img_extension = None

        # private stuff
        self._upload_folder = "./static/upload/"
        self._allowed_format = {'png', 'jpg', 'jpeg'}
        self._canny_t_upper = 200
        self._canny_t_lower = 70 
        self.result_image_counter = 0
        
        # additional
        self.initiate_upload_folder()

    def initiate_upload_folder(self):
        if not os.path.exists(self._upload_folder):
            os.makedirs(self._upload_folder)


    def get_all_path(self):
        data = {
            "ori": self.up_image_path,
            "canny": self.canny_image_path,
            "result": self.result_image_path
        }
        print(f'PREP: GET ALL PATH: {data}')
        return data

    def _verify_image_format(self, filename):
        """
        get last string till '.', then crosscheck using self._allowed_format
        return False if failed 
        """
        
        file_extension = filename.rsplit('.', 1)[1].lower()
        if '.' in filename and file_extension in self._allowed_format:
            return file_extension

    def verify_image_upload(self, request):
        """
        this function will only check image related stuff
        
        argument -- request (why not request image? in case there's no uploadFile in request)
        Return: data contain status and message to flash. (status -> True = file is image, with proper format)

        """
        return_data = {
            "status":False,
            "message":None
        }
        
        if 'uploadFile' not in request.files: # because in HTMl input name is uploadFile
            return_data['status'] = False
            return_data['message'] = "NO FILE PATH"
            return return_data
        uploaded_file = request.files['uploadFile']
        
        # If the user does not select a file, the browser submits an empty file without a filename.
        if uploaded_file.filename == '':
            return_data["status"] = False
            return_data['message'] = "NO FILE SELECTED"
            return return_data
        
        # check file format is {'png', 'jpg', 'jpeg'}, else return null
        file_format = self._verify_image_format(uploaded_file.filename)
        print(f"file format: {file_format}")
        # file is not within format 
        if not file_format:
            # TO-DO give response to in the web (flash)
            return_data["status"] = False
            return_data['message'] = "NO FILE SELECTED"
            return return_data
        
        # image with proper extension
        return_data["status"] = True
        return_data['message'] = file_format
        
        print(f"data to return: {return_data}")

        return return_data

    def save_image_to_upload(self, image_to_save, extension):
        now = time.time()
        filename = f"{int(float(now)*10000)}"
        self.up_image_path = f"{self._upload_folder}{filename}_up.{extension}"
        image_to_save.save(self.up_image_path)

        # save current image filename and extension
        self._curr_img_name = filename
        self._curr_img_extension = extension


    def get_upload_path(self):
        return self._upload_folder

    def get_curr_image_name(self):
        return self._curr_img_name
    
    def get_curr_image_extension(self):
        return self._curr_img_extension

    def get_upload_folder(self):
        return self._upload_folder

    def preprocess(self):
        self._resize_image()
        self._generate_canny()

    def _resize_image(self):
        my_img = Image.open(self.up_image_path)
        resized_image = my_img.resize((512,512))
        resized_image.save(f"{self.up_image_path}")
        

    def _generate_canny(self):
        my_img = cv2.imread(self.up_image_path)
        canny_image = cv2.Canny(my_img, self._canny_t_lower, self._canny_t_upper)
        self.canny_image_path = f"{self._upload_folder}{self._curr_img_name}_cn.{self._curr_img_extension}"
        print(f"write: {self.canny_image_path}")
        cv2.imwrite(self.canny_image_path, canny_image)
        

data_prep = DataPrep()

if __name__ == "__main__":
    upload_image_path = "static/upload/tsumuo.pdf"
    # image_editor.set_image_path(image_path=upload_image_path)
    # image_editor._resize_image()



