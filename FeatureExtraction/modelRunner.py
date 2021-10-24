import os
import time
import glob
import numpy as np
import pandas as pd
from utils import logger
from keras.preprocessing.image import load_img, img_to_array
from FeatureExtraction.utils import featuresFolderChecker, packetManager


def modelRunner(foldersList: list, framesDir: str, packetSize: int, inputSize: int, model, preprocess_input):
    for imageFolder in foldersList:
        movieId = imageFolder.rsplit('/', 1)[1]
        # Check if the folder with the same name of the movie containing features exists
        movieFeaturesExists = featuresFolderChecker(movieId, framesDir)
        if (movieFeaturesExists):
            print(
                f'🔥 Features were previously extracted in {framesDir}\\{movieId}')
        else:
            # Extract features
            startTime = time.time()
            # Initially, the whole number of frames
            remainingNumberOfFrames = len(os.listdir(imageFolder))
            # Used to be compared to packetSize, so that all x items saved into one file
            packetCounter = 0
            packetIndex = 1  # Holds the name of the packet, e.g. Packet0001
            dataFrame = pd.DataFrame(columns=['frameId', 'features'])
            for imageFile in glob.glob(f'{imageFolder}/*.jpg'):
                # Finding frameId by removing .jpg from the name
                frameId = ('frame' + imageFile.rsplit('frame', 1)[1])[:-4]
                # Load a frame and convert it into a numpy array
                frame = load_img(imageFile, target_size=(
                    inputSize, inputSize))
                frameData = img_to_array(frame)
                frameData = np.expand_dims(frameData, axis=0)
                # Preprocessing
                frameData = preprocess_input(frameData)
                # Get extracted features
                features = model.predict(frameData)
                # Append rows to dataFrame
                dataFrame = dataFrame.append(
                    {'frameId': frameId, 'features': features[0]}, ignore_index=True)
                packetCounter += 1
                # Reset the counter only if packetCounter reaches the limit (packetSize) and there is no more frames for process
                remainingNumberOfFrames -= 1
                resetCounter = (packetCounter == packetSize) or (
                    remainingNumberOfFrames == 0)
                if (resetCounter):
                    # Save dataFrame as packet in a file
                    packetManager(packetIndex, dataFrame,
                                  movieId, framesDir)
                    # Clear dataFrame rows
                    dataFrame.drop(dataFrame.index, inplace=True)
                    packetCounter = 0
                    packetIndex += 1
            elapsedTime = '{:.2f}'.format(time.time() - startTime)
            logger(
                f'Features extracted in {elapsedTime} seconds in {framesDir} ({features.shape})')
