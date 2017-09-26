import numpy as np
from keras.preprocessing import image
from keras.applications import inception_v3
import os

# Load pre-trained image recognition model
model = inception_v3.InceptionV3()

path = '../ImageInput/'
#directory = os.fsencode(directory_in_str)

for image_path in os.listdir(path):
    # create the full input path and read the file
    input_path = os.path.join(path, image_path)

    if input_path.endswith(".png") or input_path.endswith(".jpg"):
        #os.path.join(directory, filename))
        # Load the image file and convert it to a numpy array
        #print("success")
        img = image.load_img(input_path, target_size=(299, 299))
        input_image = image.img_to_array(img)

        # Scale the image so all pixel intensities are between [-1, 1] as the model expects
        input_image /= 255.
        input_image -= 0.5
        input_image *= 2.

        # Add a 4th dimension for batch size (as Keras expects)
        input_image = np.expand_dims(input_image, axis=0)

        # Run the image through the neural network
        predictions = model.predict(input_image)

        # Convert the predictions into text and print them
        #predicted_classes = inception_v3.decode_predictions(predictions, top=1)
        predicted_classes = inception_v3.decode_predictions(predictions, top=1)
        imagenet_id, name, confidence = predicted_classes[0][0]
        print("This is a {} with {:.4}% confidence!".format(name, confidence * 100))
        print(image_path)
        continue
    else:
        continue
