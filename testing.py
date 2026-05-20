from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Load your model
model = load_model('static/models/leaf_model.keras')

# Path to a sample image you want to test
test_img_path = 'dataset2/test/Ashoka/2.jpg'  # Change this to an actual file path

# Preprocess image
img = image.load_img(test_img_path, target_size=(256, 256))  # match training size
 # Fix target size
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0  # normalize same as during training

prediction = model.predict(img_array)

class_index = np.argmax(prediction)

# Class labels (make sure this list has exactly the same number as your model output)
class_names = [
    'Aloevera', 'Amla', 'Amruthaballi', 'Ashoka', 'Ashwagandha',
    'Astma_weed', 'Avacado', 'Badipala', 'Balloon_Vine', 'Bamboo',
    'Basale', 'Beans', 'Betel', 'Betel_Nut', 'Bhrami',
    'Bringaraja', 'Camphor', 'Castor', 'Catharanthus', 'Cathedral Bells',
    'Chakte', 'Chilly', 'Citron lime (herelikai)', 'Coffee', 'Common rue(naagdalli)',
    'Coriander', 'Curry', 'Doddpathre', 'Drumstick', 'Ekka',
    'Eucalyptus', 'Ganigale', 'Ganike', 'Gasagase', 'Geranium',
    'Ginger', 'Globe Amarnath', 'Guava', 'Henna', 'Hibiscus',
    'Honge', 'Insulin', 'Jackfruit', 'Jasmine', 'Kambajala',
    'Kasambruga', 'Kohlrabi', 'Lantana', 'Lemon', 'Lemongrass',
    'Malabar_Nut', 'Malabar_Spinach', 'Mango', 'Marigold', 'Mint',
    'Nagadali', 'Neem', 'Nelavembu', 'Nerale', 'Nithyapushpa',
    'Nooni', 'Onion', 'Padri', 'Palak(Spinach)', 'Papaya',
    'Parijatha', 'Pea', 'Peepal', 'Pepper', 'Pomoegranate',
    'Pumpkin', 'Raddish', 'Raktachandini', 'Rose', 'Sampige',
    'Sapota', 'Seethaashoka', 'Seethapala', 'Tamarind', 'Taro',
    'Tecoma', 'Thumbe', 'Tomato', 'Tulsi', 'Turmeric',
    'Wood_sorel', 'banana', 'kamakasturi', 'karanj', 'kepala'
]

# Output
print("Predicted class index:", class_index)
print("Predicted class label:", class_names[class_index])
