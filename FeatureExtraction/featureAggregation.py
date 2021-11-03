import os
import time
import glob
import json
import numpy as np
import pandas as pd
from utils import logger
from config import aggFeaturesDir


def featureAggregation(featureFoldersList: list):
    for featuresFolder in featureFoldersList:
        videoId = featuresFolder.rsplit('/', 1)[1]
        # Check if the file with the same name of the video containing aggregated features exists
        aggregatedFile = f'{aggFeaturesDir}/{videoId}.json'
        # True means it has been processed before
        isAggregated = os.path.isfile(aggregatedFile)
        if (isAggregated):
            print(
                f'🔥 Features were previously aggregated for {videoId}')
        else:
            startTime = time.time()
            # Read packet JSON files
            numberOfPackets = len(os.listdir(featuresFolder))
            # Arrays to store each video's columns altogether
            maxAggFeatures = []
            meanAggFeatures = []
            packetCounter = 0
            print(
                f'Processing {numberOfPackets} packets of the video "{videoId}" ...')
            for packetFile in glob.glob(f'{featuresFolder}/*.json'):
                # Reading each packet's data
                jsonFile = open(packetFile,)
                packetData = json.load(jsonFile)
                packetCounter += 1
                # Arrays to store each packet's columns altogether
                packetAggFeatures = []
                # Iterate on each frames of array
                for frameData in packetData:
                    features = np.asarray(frameData['features'])
                    packetAggFeatures.append(features)
                # Using the packet-level aggregated array for max/mean calculations
                meanPacketAggFeatures = np.mean(packetAggFeatures, axis=0)
                maxPacketAggFeatures = np.max(packetAggFeatures, axis=0)
                meanPacketAggFeatures = np.round(meanPacketAggFeatures, 6)
                maxPacketAggFeatures = np.round(maxPacketAggFeatures, 6)
                # Append them to video-level aggregation
                maxAggFeatures.append(maxPacketAggFeatures)
                meanAggFeatures.append(
                    meanPacketAggFeatures)
                if (packetCounter % 25 == 0):
                    print(f'Packet #{packetCounter} has been processed!')
            # Using the video-level aggregated array for max/mean calculations
            maxAggFeatures = np.mean(maxAggFeatures, axis=0)
            meanAggFeatures = np.max(meanAggFeatures, axis=0)
            maxAggFeatures = np.round(maxAggFeatures, 6)
            meanAggFeatures = np.round(meanAggFeatures, 6)
            # Save aggregated arrays in files
            dataFrame = pd.DataFrame(columns=['Max', 'Mean'])
            dataFrame = dataFrame.append(
                {'Max': maxAggFeatures, 'Mean': meanAggFeatures}, ignore_index=True)
            dataFrame.to_json(
                f'{aggFeaturesDir}/{videoId}.json', orient="records")
            elapsedTime = int(time.time() - startTime)
            logger(
                f'Aggregated {packetCounter} packets of {videoId} in {elapsedTime} seconds.')
