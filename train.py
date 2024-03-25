import os
import json
import tensorflow as tf
from mediapipe_model_maker import object_detector

# Prepare data
print('\n\n====================================================\n')

train_dataset_path = "data/train"
validation_dataset_path = "data/validation"


print(os.path.join(train_dataset_path, "labels.json"))

# Review the dataset
with open(os.path.join(train_dataset_path, "labels.json"), "r") as f:
  labels_json = json.load(f)
for category_item in labels_json["categories"]:
  print(f"{category_item['id']}: {category_item['name']}")
  
  
print('\n====================================================\n')

# Create dataset
train_data = object_detector.Dataset.from_coco_folder(train_dataset_path, cache_dir="/tmp/od_data/train")
validation_data = object_detector.Dataset.from_coco_folder(validation_dataset_path, cache_dir="/tmp/od_data/validation")
print("train_data size: ", train_data.size)
print("validation_data size: ", validation_data.size)

# Set retraining options
spec = object_detector.SupportedModels.MOBILENET_MULTI_AVG
hparams = object_detector.HParams(export_dir='exported_model', epochs=100, batch_size=16)
options = object_detector.ObjectDetectorOptions(
    supported_model=spec,
    hparams=hparams
)

# Run retraining
model = object_detector.ObjectDetector.create(
    train_data=train_data,
    validation_data=validation_data,
    options=options)


# Evaluate the model performance
loss, coco_metrics = model.evaluate(validation_data, batch_size=4)
print(f"Validation loss: {loss}")
print(f"Validation coco metrics: {coco_metrics}")

print('EXPORTING THE MODEL..')
# Export the model
model.export_model()