import numpy as np
from keras.models import load_model
import cv2


class SignPredicter:
    def __init__(self, model_name):
        self.model = load_model(model_name)
        
    def predict(self, image):
        image = image[0:180, 460:640, :]
        image = cv2.resize(image, (32, 32))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.equalizeHist(image)
        image = np.reshape(image, (32, 32, 1))
        image = np.asarray([image])
        return self.model.predict_classes(image)

sign_predicter = SignPredicter("./MODELS/model-350374-0.8641199952363968-0.9964000034332275-32.h5")
print(sign_predicter.predict(cv2.imread("./40/image091.jpeg")))
