import numpy as np
from keras.models import load_model
import cv2


class SignPredicter:
    def __init__(self, model_name):
        self.model = load_model(model_name)
        
    def predict(self, image):
        image = image[0:240, 320:640, :]
        
        red_mask = 1.75 * image[:, :, 2] - 0.75 * image[:, :, 1] - 0.75 * image[:, :, 0]
        red_mask[red_mask < 120] = 0
        red_mask[red_mask >= 120] = 255
        red_mask = np.uint8(red_mask)
        red_mask = cv2.dilate(red_mask, None, iterations=5)
        
        filling_contours = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        filling_mask = np.zeros(image.shape[:2], np.uint8)
        red_mask = cv2.drawContours(filling_mask, filling_contours, -1, 255, -1)
        
        red_mask = cv2.erode(red_mask, None, iterations=12)
        red_mask = cv2.dilate(red_mask, None, iterations=8)
        
        contours, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        
        rectangles = list(map(lambda contour: cv2.boundingRect(contour), contours))
        rectangles = list(filter(lambda rect: (rect[2] > 25 and rect[3] > 25), rectangles))
        
        if len(rectangles) == 0:
            return None
        
        rectangle_choice = min(rectangles, key=lambda rectangle: rectangle[1])
        
        x, y, w, h = rectangle_choice
        
        x, y, w, h = x, y, w, w
        image = image[y:y+h, x:x+w, :]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (32, 32))
        image = cv2.equalizeHist(image)
        
        image = np.reshape(image, (32, 32, 1))
        image = np.asarray([image])
        
        return self.model.predict_classes(image)


sign_predicter = SignPredicter("./MODELS/working-357983-10epochs-100steps-100batch.h5")
prediction = sign_predicter.predict(cv2.resize(cv2.imread("./real-stop-test.jpg"), (640, 480)))

if prediction == 0:
    print("STOP")
elif predicition == 1:
    print("5")
elif prediction == 2:
    print("10")
elif predicition == None:
    print("No valid region for sign could be found.")  # != no sign, no sign not yet implemented
else:
    print("What did you do to make this happen?")
