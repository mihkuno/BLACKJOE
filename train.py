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
spec = object_detector.SupportedModels.MOBILENET_V2_I320
hparams = object_detector.HParams(export_dir='model', epochs=100, batch_size=8)
options = object_detector.ObjectDetectorOptions(
    supported_model=spec,
    hparams=hparams
)

# Run retraining
model = object_detector.ObjectDetector.create(
    train_data=train_data,
    validation_data=validation_data,
    options=options)


# Export model
print('-'*100)
print('exporting float model')
model.export_model()


print('-'*100)
print('quantization aware training')

qat_hparams = object_detector.QATHParams(learning_rate=0.3, batch_size=4, epochs=10, decay_steps=6, decay_rate=0.96)
model.quantization_aware_training(train_data, validation_data, qat_hparams=qat_hparams)
qat_loss, qat_coco_metrics = model.evaluate(validation_data)
print(f"QAT validation loss: {qat_loss}")
print(f"QAT validation coco metrics: {qat_coco_metrics}")


print('-'*100)
print('exporting quantized model')
model.export_model('first_model_int8_qat.tflite')


print('-'*100)
print('second quantization aware training')

new_qat_hparams = object_detector.QATHParams(learning_rate=0.9, batch_size=4, epochs=15, decay_steps=5, decay_rate=0.96)
model.restore_float_ckpt()
model.quantization_aware_training(train_data, validation_data, qat_hparams=new_qat_hparams)
qat_loss, qat_coco_metrics = model.evaluate(validation_data)
print(f"QAT validation loss: {qat_loss}")
print(f"QAT validation coco metrics: {qat_coco_metrics}")



print('-'*100)
print('exporting last quantized model')
model.export_model('second_model_int8_qat.tflite')



# Quantization to increase speed and reduce model size
# print('-'*100)
# print('quantization aware training')
# qat_hparams = object_detector.QATHParams(learning_rate=0.3, batch_size=4, epochs=10, decay_steps=6, decay_rate=0.96)
# model.quantization_aware_training(train_data, validation_data, qat_hparams=qat_hparams)
# qat_loss, qat_coco_metrics = model.evaluate(validation_data)
# print(f"QAT validation loss: {qat_loss}")
# print(f"QAT validation coco metrics: {qat_coco_metrics}")

# # Export quantized model
# model.export_model('model_int8_qat.tflite')


# Evaluate model
# print('-'*100)
# print('evaluating model')
# loss, coco_metrics = model.evaluate(validation_data, batch_size=4)
# print(f"Validation loss: {loss}")
# print(f"Validation coco metrics: {coco_metrics}")
