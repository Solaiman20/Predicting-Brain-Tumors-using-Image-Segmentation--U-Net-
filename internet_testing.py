import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def preprocess_local(image_path):
    """
    Loads a local image file, resizes it to (256, 256),
    converts it to a normalized tensor, and adds a batch dimension.
    
    Arguments:
        image_path -- Path to the local image file.
    
    Returns:
        A tuple containing the original resized image and the preprocessed image tensor.
    """
    # Load the image with target size (256, 256)
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(256, 256))
    # Convert the PIL image to a NumPy array
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    # Normalize pixel values to the range [0, 1]
    img_array_normalized = tf.cast(img_array, tf.float32) / 255.0
    # Add a batch dimension so the shape becomes (1, 256, 256, 3)
    img_array_normalized = tf.expand_dims(img_array_normalized, axis=0)
    return img_array, img_array_normalized

# Loading the trained U-Net model saved.
model = keras.models.load_model('unet_tumor_model.keras', compile=False)

# The local path to the image getting tested.
image_path = "WebTest.png" 

# Preprocess the local image.
original_image, input_image = preprocess_local(image_path)

# Run the preprocessed image through the model to predict the segmentation mask.
pred_mask = model.predict(input_image)

# Remove the batch dimension.
pred_mask = np.squeeze(pred_mask, axis=0)

# Scale the predictions to the range [0, 255] and convert to uint8.
pred_mask = (pred_mask * 255).astype(np.uint8)

# Display the original image and the predicted mask side by side.
plt.figure(figsize=(12, 6))

# Display the original image.
plt.subplot(1, 2, 1)
plt.imshow(original_image.astype(np.uint8))
plt.title('Original Image')
plt.axis('off')

# Display the predicted mask.
plt.subplot(1, 2, 2)
plt.imshow(pred_mask, cmap='gray')
plt.title('Predicted Mask')
plt.axis('off')

plt.show()
