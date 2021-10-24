from utils import logger
from keras import Model
from FeatureExtraction.modelRunner import modelRunner
from keras.applications.vgg19 import VGG19, preprocess_input

# About VGG-19:
# Running the example will load the VGG16 model and download the model weights
# The model can then be used directly to classify a photograph into one of 1,000 classes

# Static variables
vggInputSize = 224


def VGG19Launcher(foldersList: list, framesDir: str, packetSize: int):
    logger('Launching Inception-v3 network ...')
    # Load model
    model = VGG19()
    # Removing the final output layer, so that the second last fully connected layer with 4,096 nodes will be the new output layer
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    modelRunner(foldersList, framesDir, packetSize,
                vggInputSize, model, preprocess_input)
