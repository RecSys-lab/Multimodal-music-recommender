from keras import Model
from utils import logger
from FeatureExtraction.modelRunner import modelRunner
from keras.applications.inception_v3 import InceptionV3, preprocess_input

# About Inception-v3 (GoogleNet):
# The model expects color images to have the square shape 299×299
# Running the example will load the Inception-v3 model and download the model weights

# Static variables
vggInputSize = 299


def Inception3Launcher(foldersList: list, framesDir: str, packetSize: int):
    logger('Launching Inception-v3 network ...')
    # Load model
    model = InceptionV3()
    # Removing the final output layer, so that the second last fully connected layer with 2,048 nodes will be the new output layer
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    modelRunner(foldersList, framesDir, packetSize,
                vggInputSize, model, preprocess_input)
