import tensorflow as tf
import tensorflow_hub as hub
import tf_keras
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import numpy as np
import io
from fastapi.responses import HTMLResponse

# 1-create the FastAPI app
app = FastAPI()

# 2-load the trained model
model = tf_keras.models.load_model(
    '26-151417-full_dog_breed_effectivenetv2b0.h5',
    custom_objects={'KerasLayer': hub.KerasLayer}
)
print("Model loaded successfully!")

# 3-image preprocessing function matching your training pipeline


def prepare_image(image_bytes):
    # convert raw bytes to a TensorFlow string tensor
    img_tensor = tf.constant(image_bytes)

    # decode the JPEG image into a numerical tensor with 3 color channels
    image = tf.image.decode_jpeg(img_tensor, channels=3)

    # convert the color channel values from 0-255 to 0-1
    image = tf.image.convert_image_dtype(image, tf.float32)

    # resize our image to the desired image size
    image = tf.image.resize(image, size=[224, 224])

    # add batch dimension: shape becomes (1, 224, 224, 3)
    image = tf.expand_dims(image, axis=0)

    return image


# 4-list of 120 dog breeds
class_names = [
    'affenpinscher', 'afghan_hound', 'african_hunting_dog', 'airedale',
    'american_staffordshire_terrier', 'appenzeller', 'australian_terrier',
    'basenji', 'basset', 'beagle', 'bedlington_terrier', 'bernese_mountain_dog',
    'black-and-tan_coonhound', 'blenheim_spaniel', 'bloodhound', 'bluetick',
    'border_collie', 'border_terrier', 'borzoi', 'boston_bull',
    'bouvier_des_flandres', 'boxer', 'brabancon_griffon', 'briard',
    'brittany_spaniel', 'bull_mastiff', 'cairn', 'cardigan',
    'chesapeake_bay_retriever', 'chihuahua', 'chow', 'clumber',
    'cocker_spaniel', 'collie', 'curly-coated_retriever', 'dandie_dinmont',
    'dhole', 'dingo', 'doberman', 'english_foxhound', 'english_setter',
    'english_springer', 'entlebucher', 'eskimo_dog', 'flat-coated_retriever',
    'french_bulldog', 'german_shepherd', 'german_short-haired_pointer',
    'giant_schnauzer', 'golden_retriever', 'gordon_setter', 'great_dane',
    'great_pyrenees', 'greater_swiss_mountain_dog', 'groenendael',
    'ibizan_hound', 'irish_setter', 'irish_terrier', 'irish_water_spaniel',
    'irish_wolfhound', 'italian_greyhound', 'japanese_spaniel', 'keeshond',
    'kelpie', 'kerry_blue_terrier', 'komondor', 'kuvasz', 'labrador_retriever',
    'lakeland_terrier', 'leonberg', 'lhasa', 'malamute', 'malinois',
    'maltese_dog', 'mexican_hairless', 'miniature_pinscher', 'miniature_poodle',
    'miniature_schnauzer', 'newfoundland', 'norfolk_terrier', 'norwegian_elkhound',
    'norwich_terrier', 'old_english_sheepdog', 'otterhound', 'papillon',
    'pekinese', 'pembroke', 'pomeranian', 'pug', 'redbone', 'rhodesian_ridgeback',
    'rottweiler', 'saint_bernard', 'saluki', 'samoyed', 'schipperke',
    'scotch_terrier', 'scottish_deerhound', 'sealyham_terrier', 'shetland_sheepdog',
    'shih-tzu', 'siberian_husky', 'silky_terrier', 'soft-coated_wheaten_terrier',
    'staffordshire_bullterrier', 'standard_poodle', 'standard_schnauzer',
    'sussex_spaniel', 'tibetan_mastiff', 'tibetan_terrier', 'toy_poodle',
    'toy_terrier', 'vizsla', 'walker_hound', 'weimaraner', 'welsh_springer_spaniel',
    'west_highland_white_terrier', 'whippet', 'wire-haired_fox_terrier', 'yorkshire_terrier'
]

# 5-route to serve the HTML interface


@app.get("/", response_class=HTMLResponse)
async def get_webpage():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

# 6-route to handle image prediction


@app.post("/predict")
async def predict_dog_breed(file: UploadFile = File(...)):
    image_bytes = await file.read()
    processed_image = prepare_image(image_bytes)

    predictions = model.predict(processed_image)
    predicted_class_index = np.argmax(predictions[0])
    confidence = np.max(predictions[0])

    predicted_breed = class_names[predicted_class_index]
    confidence_percentage = round(float(confidence) * 100, 2)

    return {
        "filename": file.filename,
        "breed": predicted_breed,
        "confidence": confidence_percentage
    }
