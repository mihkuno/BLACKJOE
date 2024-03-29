import warnings
warnings.filterwarnings("ignore")

from tflite_model_maker import model_spec
from tflite_model_maker import object_detector

print('-'*100)
print('preparing dataset...')


EPOCHS = 1
BATCH_SIZE = 5
MODEL_NAME = 'blackjack.tflite'
CLASSES = [
    "10C", "10D", "10H", "10S",
    "2C", "2D", "2H", "2S",
    "3C", "3D", "3H", "3S",
    "4C", "4D", "4H", "4S",
    "5C", "5D", "5H", "5S",
    "6C", "6D", "6H", "6S",
    "7C", "7D", "7H", "7S",
    "8C", "8D", "8H", "8S",
    "9C", "9D", "9H", "9S",
    "AC", "AD", "AH", "AS",
    "JC", "JD", "JH", "JS",
    "KC", "KD", "KH", "KS",
    "QC", "QD", "QH", "QS"
]
MODEL_PATH = 'data/'
TRAIN_DATASET_PATH = 'data/train'
VALID_DATASET_PATH = 'data/valid'
MODEL = 'efficientdet_lite2'

train_data = object_detector.DataLoader.from_pascal_voc(
    TRAIN_DATASET_PATH,
    TRAIN_DATASET_PATH,
    CLASSES
)

val_data = object_detector.DataLoader.from_pascal_voc(
    VALID_DATASET_PATH,
    VALID_DATASET_PATH,
    CLASSES
)

print('-'*100)
print('training start!')

spec = model_spec.get(MODEL)

model = object_detector.create(
    train_data,
    model_spec=spec,
    batch_size=BATCH_SIZE,
    train_whole_model=True,
    epochs=EPOCHS,
    validation_data=val_data
)

print('-'*100)
print('exporting the model..')
model.export(export_dir=MODEL_PATH, tflite_filename=MODEL_NAME)

print('-'*100)
print('Training completed.')
print('See the model folder.')


# model.evaluate(val_data)
