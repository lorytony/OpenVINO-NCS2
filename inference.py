
import time
from pathlib import Path
import numpy as np
import openvino.runtime as ov
from openvino.preprocess import PrePostProcessor, ResizeAlgorithm
import tensorflow as tf
from tensorflow import keras
from keras import layers, models, optimizers
from sklearn import metrics
from sklearn.metrics import classification_report
import torch

#Tensorflow used 2.9
print(tf.__version__)

# Checking if Myriad device is correctly installed
core = ov.Core()
for device in core.available_devices:
    print(device)
cpu_device_name = core.get_property("MYRIAD","FULL_DEVICE_NAME")
print(cpu_device_name)


# upload testset as a numpy array
#keras model was trained on 300x200 size images
# X_test_300_200 dimension: (248,300,200,3)
# y_test_300_200 dimension: (248,)

with open('<ADD_PATH>/X_test_300_200.npy', 'rb') as f:
    X = np.load(f,allow_pickle=True)
with open('<ADD_PATH>/y_test_300_200.npy', 'rb') as f:
    y = np.load(f)



# Salvo i path dei modelli
path_model_pytorch = '<ADD_PATH>/models/openivno/from_keras/model.xml'
path_model_keras = '<ADD_PATH>/models/openvino/from_pytorch/model.xml'


# ***************** Keras model - CPU 
model = core.read_model(model=path_model_keras)
compiled_model = core.compile_model(model=model, device_name="CPU")
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

y_tmp = []
dim = len(X)
for i in range(0,dim):
    input_arr = np.array([X[i]])  # Convert single image to a batch.
    result_infer = compiled_model([input_arr])[output_layer]
    result_index = np.argmax(result_infer)
    res = result_index
    y_tmp.append(res)
    #print(str(i) + " predicted on " + str(dim))
y_pred = np.asarray(y_tmp)

print(classification_report(y, y_pred, output_dict = True)['weighted avg'])
print(classification_report(y, y_pred, output_dict = True)['accuracy'])

# ***************** Keras model - MYRIAD 
model = core.read_model(model=path_model_xml_opt4)
model.reshape([1,300,200,3])
compiled_model = core.compile_model(model=model, device_name="MYRIAD")
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

y_tmp = []
dim = len(X)
for i in range(0,dim):
    input_arr = np.array([X[i]])  # Convert single image to a batch.
    result_infer = compiled_model([input_arr])[output_layer]
    result_index = np.argmax(result_infer)
    res = result_index
    y_tmp.append(res)
    #print(str(i) + " predicted on " + str(dim))

y_pred = np.asarray(y_tmp)

print(classification_report(y, y_pred, output_dict = True)['weighted avg'])
print(classification_report(y, y_pred, output_dict = True)['accuracy'])
#{'precision': 0.7588057046488884, 'recall': 0.7217741935483871, 'f1-score': 0.7242074908245159, 'support': 248}
# 'accuracy': 0.7217741935483871


#pytorch model was trained on 244x244 size images
# X_test_244_244 dimension: (248,3,244,244)
# y_test_244_244 dimension: (248,)

with open('<ADD_PATH>/X_test_224_224.npy', 'rb') as f:
    X = np.load(f,allow_pickle=True)
with open('<ADD_PATH>/y_test_224_224.npy', 'rb') as f:
    y = np.load(f)


# ******************* Pytorch model - CPU
model = core.read_model(model=path_model_pytorch)
model.reshape([1,3,224,224]) #OBBLIGATORIO QUANDO SI USA MYRIAD INVECE DI CPU
compiled_model = core.compile_model(model=model, device_name="MYRIAD")
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

y_tmp = []
dim = len(X)
for i in range(0,dim):
    input_arr = np.array([X[i]])  # Convert single image to a batch.
    #input_arr = X[i]
    result_infer = compiled_model([input_arr])[output_layer]
    result_index = np.argmax(result_infer)
    res = result_index
    y_tmp.append(res)
    #print(str(i) + " predicted out of " + str(dim))

y_pred = np.asarray(y_tmp)
print("***  CPU - results ****")
print(classification_report(y, y_pred, output_dict = True)['weighted avg'])
print(classification_report(y, y_pred, output_dict = True)['accuracy'])
#{'precision': 0.904606455732707, 'recall': 0.8991935483870968, 'f1-score': 0.8994797634848273, 'support': 248}
#accuracy:0.8991935483870968



# ******************** Pytorch model - MYRIAD
model = core.read_model(model=path_model_pytorch)
model.reshape([1,3,224,224]) #OBBLIGATORIO QUANDO SI USA MYRIAD INVECE DI CPU
compiled_model = core.compile_model(model=model, device_name="MYRIAD")
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

y_tmp = []
dim = len(X)
for i in range(0,dim):
    input_arr = np.array([X[i]])  # Convert single image to a batch.
    #input_arr = X[i]
    result_infer = compiled_model([input_arr])[output_layer]
    result_index = np.argmax(result_infer)
    res = result_index
    y_tmp.append(res)
    print(str(i) + " predicted out of " + str(dim))

y_pred = np.asarray(y_tmp)
print("***  MYRIAD - results ****")
print(classification_report(y, y_pred, output_dict = True)['weighted avg'])
print(classification_report(y, y_pred, output_dict = True)['accuracy'])
#{'precision': 0.904606455732707, 'recall': 0.8991935483870968, 'f1-score': 0.8994797634848273, 'support': 248}
#accuracy:0.8991935483870968
 
