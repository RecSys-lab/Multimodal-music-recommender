import os
from keras import Model
from utils import logger
from config import featuresDir
from FeatureExtraction.modelRunner import modelRunner
from keras.applications.resnet_v2 import ResNet101V2, preprocess_input


# About ResNet101V2:
# The model expects color images to have the square shape 224×224
# Running the example will load the ResNet101V2 model and download the model weights

# Static variables
resnetInputSize = 224

def ResNet101v2Launcher(foldersList: list):
    logger('Launching ResNet101-v2 network ...')
    # Create a folder for outputs if not existed
    outputPath = f'{featuresDir}/ResNet'
    if not os.path.exists(outputPath):
        os.mkdir(outputPath)
    # Load model
    model = ResNet101V2()
    # Removing the final output layer, so that the second last fully connected layer with 2,048 nodes will be the new output layer
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    modelRunner(outputPath, foldersList, resnetInputSize,
                model, preprocess_input)

