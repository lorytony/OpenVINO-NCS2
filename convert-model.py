#import numpy as np
import random as rn
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import VGG16
import PIL
from PIL import Image
import cv2
import os
import tf2onnx
import onnx

print("conversion")
vgg_model = tf.keras.models.load_model('model/model.h5')

onnx_model, _ = tf2onnx.convert.from_keras(vgg_model, opset=11)#I tried also with opset=13
onnx.save(onnx_model, "model-opset10.onnx")
