import logging
from utils import logger
from PyInquirer import prompt
from FeatureExtraction.main import featureExtractor
from FramesExtraction.frameExtractor import frameExtractor

modules = ['Frame Extraction', 'Visual Features Extraction']


def getUserInput():
    questions = [
        {
            'type': 'list',
            'name': 'Action',
            'message': 'Choose the module you want to use:',
            'choices': modules
        },
    ]
    userInputs = prompt(questions)
    return userInputs


def __init__():
    # Creating log file
    logging.basicConfig(filename='logger.log', level=logging.INFO)
    logger('Framework started!')
    # Getting inputs from users
    userInputs = getUserInput()['Action']
    if userInputs == 'Frame Extraction':
        frameExtractor()
    elif userInputs == 'Visual Features Extraction':
        featureExtractor()


__init__()
