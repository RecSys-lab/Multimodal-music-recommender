import os
import time
import glob
import json
import numpy as np
import pandas as pd
from utils import logger


def featureAggregation(featureFoldersList: list, framesDir: str):
    for featuresFolder in featureFoldersList:
        videoId = featuresFolder.rsplit('/', 1)[1]
        # Check if the file with the same name of the video containing aggregated features exists
        aggregatedFile = f'{framesDir}/{videoId}.json'
        # true means it has been processed before
        isAggregated = os.path.isfile(aggregatedFile)
        if (isAggregated):
            print(
                f'🔥 Features were previously aggregated for {videoId}')
        else:
            startTime = time.time()
            # Read packet JSON files
            numberOfPackets = len(os.listdir(featuresFolder))
            # Arrays to store each video's columns altogether
            maxvideoAggregatedFeatures = []
            meanvideoAggregatedFeatures = []
            packetCounter = 0
            print(
                f'Processing {numberOfPackets} packets of the video "{videoId}" ...')
            for packetFile in glob.glob(f'{featuresFolder}/*.json'):
                # Reading each packet's data
                jsonFile = open(packetFile,)
                packetData = json.load(jsonFile)
                packetCounter += 1
                # Arrays to store each packet's columns altogether
                packetAggregatedFeatures = []
                # Iterate on each frames of array
                for frameData in packetData:
                    features = frameData['features']
                    features = np.asarray(features)
                    packetAggregatedFeatures.append(features)
                # Using the packet-level aggregated array for max/mean calculations
                meanPacketAggregatedFeatures = np.mean(
                    packetAggregatedFeatures, axis=0)
                maxPacketAggregatedFeatures = np.max(
                    packetAggregatedFeatures, axis=0)
                meanPacketAggregatedFeatures = np.round(
                    meanPacketAggregatedFeatures, 6)
                maxPacketAggregatedFeatures = np.round(
                    maxPacketAggregatedFeatures, 6)
                # Append them to video-level aggregation
                maxvideoAggregatedFeatures.append(maxPacketAggregatedFeatures)
                meanvideoAggregatedFeatures.append(
                    meanPacketAggregatedFeatures)
                if (packetCounter % 25 == 0):
                    print(f'Packet #{packetCounter} has been processed!')
            # Using the video-level aggregated array for max/mean calculations
            maxvideoAggregatedFeatures = np.mean(
                maxvideoAggregatedFeatures, axis=0)
            meanvideoAggregatedFeatures = np.max(
                meanvideoAggregatedFeatures, axis=0)
            maxvideoAggregatedFeatures = np.round(
                maxvideoAggregatedFeatures, 6)
            meanvideoAggregatedFeatures = np.round(
                meanvideoAggregatedFeatures, 6)
            # Save aggregated arrays in files
            dataFrame = pd.DataFrame(columns=['Max', 'Mean'])
            dataFrame = dataFrame.append(
                {'Max': maxvideoAggregatedFeatures, 'Mean': meanvideoAggregatedFeatures}, ignore_index=True)
            dataFrame.to_json(
                f'{framesDir}/{videoId}.json', orient="records")
            elapsedTime = int(time.time() - startTime)
            logger(
                f'Finished aggregating the packets of video "{videoId}" in {elapsedTime} seconds.')
