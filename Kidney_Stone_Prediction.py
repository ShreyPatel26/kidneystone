import numpy as np
from PIL import Image
from keras.models import load_model

IMAGE_SIZE = 150

# Load the image and resize it to the desired size patient with kidney stone
#image_name = 'C:/Users/matts/OneDrive/Desktop/7_Kidney_Stone_Detection/Dataset/Test/Stone/Stone- (1174).jpg'
image_name = 'C:/Users/matts/OneDrive/Desktop/7_Kidney_Stone_Detection/Dataset/Test/Stone/Stone- (1175).jpg'

# Load the image and resize it to the desired size patient with kidney stone
#image_name = 'C:/Users/matts/OneDrive/Desktop/7_Kidney_Stone_Detection/Dataset/Test/Normal/Normal- (1004).jpg'

img = Image.open(image_name).resize((IMAGE_SIZE, IMAGE_SIZE))

# Load the model
model = load_model('kidney_stone_model.h5')

# Function to Predict CNN
def predict_btn_cnn(image):
    test_img = np.expand_dims(image, axis=0)
    result = model.predict(test_img)
    if result[0][0] == 1:
        print("Patient has kidney stone")
    elif result[0][0] == 0:
        print("Patient is Healthy")

# Call the function with the image
predict_btn_cnn(img)
