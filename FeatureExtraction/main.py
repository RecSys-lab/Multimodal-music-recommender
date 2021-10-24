from utils import logger
from PyInquirer import prompt
from FeatureExtraction.utils import SubdirectoryExtractor
from FeatureExtraction.Models.VGG19 import VGG19Launcher
from FeatureExtraction.Models.Inception3 import Inception3Launcher
from FeatureExtraction.featureAggregation import featureAggregation


modules = ['Feature Extraction - InceptionV3',
           'Feature Extraction - VGG19', 'Feature Aggeration']


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


def featureExtractor(movieFramesDirectory: str, movieFeaturesDirectory: str, aggregatedFeaturesDirectory: str, packetSize: int):
    logger('Features Extractor Started!')
    # Fetcth the list of movie folder(s) containing frames
    framesFoldersList = SubdirectoryExtractor(movieFramesDirectory)
    userInput = getUserInput()['Action']
    if userInput == 'Feature Extraction - VGG19':
        VGG19Launcher(framesFoldersList, movieFeaturesDirectory, packetSize)
    elif userInput == 'Feature Extraction - InceptionV3':
        Inception3Launcher(framesFoldersList,
                           movieFeaturesDirectory, packetSize)
    elif userInput == 'Feature Aggeration':
        # Fetcth the list of movie folder(s) containing packets
        packetsFoldersList = SubdirectoryExtractor(movieFeaturesDirectory)
        # Aggregates all features for each movie and produces a CSV file
        featureAggregation(packetsFoldersList, aggregatedFeaturesDirectory)
