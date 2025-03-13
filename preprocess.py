import cv2
from PIL import Image
import time
 
class ImagePrep():
    def __init__(self):
        self.up_image_path = "./static/placeholder.jpg"
        self.canny_image_path = "./static/placeholder1.jpg"
        self.result_image_path = "./static/placeholder2.jpg"

        # private stuff
        self._upload_folder = "./static/upload/"
        self._allowed_extensions = {'png', 'jpg', 'jpeg'}
        self.result_image_counter = 0
        
    
    def set_image_path(self, image_path):

        self.up_image_path = image_path
        return

    def verify_image(self, filename):
        file_extension = filename.rsplit('.', 1)[1].lower()
        if '.' in filename and file_extension in self._allowed_extensions:
            return file_extension
        
        return None

    def get_all_path(self):
        data = {
            "ori": self.up_image_path,
            "canny": self.canny_image_path,
            "result": self.result_image_path
        }
        return data

    def get_upload_path(self):
        return self._upload_folder

    def _resize_image(self):
        my_img = Image.open(self.up_image_path)
        resized_image = my_img.resize((512,512))
        resized_image.save(f"{self.up_image_path}")
        print(resized_image)

    def _generate_canny(self):
        print("Heelow")

image_editor = ImagePrep()

if __name__ == "__main__":
    upload_image_path = "static/upload/tsumuo.pdf"
    # image_editor.set_image_path(image_path=upload_image_path)
    # image_editor._resize_image()



