import os
from utils import logger
from PyInquirer import prompt
from config import framesDir, featuresDir, aggFeaturesDir
from FeatureExtraction.Models.VGG19 import VGG19Launcher
from FeatureExtraction.utils import SubdirectoryExtractor
from FeatureExtraction.Models.Inception3 import Inception3Launcher
from FeatureExtraction.Models.ResNet101V2 import ResNet101v2Launcher
from FeatureExtraction.featureAggregation import featureAggregation


modules = ['Feature Extraction - InceptionV3', 'Feature Extraction - ResNet101V2',
           'Feature Extraction - VGG19', 'Feature Aggregation']


def getUserInput():
    questions = [
        {
            'type': 'list',
            'name': 'Action',
            'message': 'Select an action from the list below:',
            'choices': modules
        },
    ]
    userInput = prompt(questions)
    return userInput


def featureExtractor():
    logger('Features Extractor Started!')
    # Fetcth the list of video folder(s) containing frames
    framesFoldersList = SubdirectoryExtractor(framesDir)
    # Create a folder for outputs if not existed
    if not os.path.exists(featuresDir):
        os.mkdir(featuresDir)
    # Get action from user
    userInput = getUserInput()['Action']
    if userInput == 'Feature Extraction - VGG19':
        VGG19Launcher(framesFoldersList)
    elif userInput == 'Feature Extraction - InceptionV3':
        Inception3Launcher(framesFoldersList)
    elif userInput == 'Feature Extraction - ResNet101V2':
        ResNet101v2Launcher(framesFoldersList)
    elif userInput == 'Feature Aggregation':
        # Fetcth the list of video folder(s) containing packets
        packetsFoldersList = SubdirectoryExtractor(featuresDir)
        # Aggregates all features for each video and produces a CSV file
        featureAggregation(packetsFoldersList)
