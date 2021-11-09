import os
from glob import glob
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


def selectFolder():
    # Show only folders inside the featuresDir (e.g., VGG19, Incp3)
    featuresFolders = glob(f'{featuresDir}/*/')
    choices = [os.path.dirname(folder).split('\\')[-1]
               for folder in featuresFolders]
    possibilities = [
        {
            'type': 'list',
            'name': 'Action',
            'message': 'Select from which model you want to take the extracted features:',
            'choices': choices
        },
    ]
    return prompt(possibilities)


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
        # Create a folder for outputs if not existed
        if not os.path.exists(aggFeaturesDir):
            os.mkdir(aggFeaturesDir)
        # Prompt the user with folder associated with the models
        selectedFolder = selectFolder()['Action']
        aggFolder = f'{aggFeaturesDir}/{selectedFolder}'
        # Fetch the list of folder(s) containing packets
        packetsFoldersList = SubdirectoryExtractor(
            f'{featuresDir}/{selectedFolder}')
        # Create a folder for outputs if not existed
        if not os.path.exists(aggFolder):
            os.mkdir(aggFolder)
        # Aggregates all features for each video and produces a CSV file
        featureAggregation(packetsFoldersList, aggFolder)
