import os
import cv2
import logging
import datetime
from typing_extensions import Literal

logLevelType = Literal["info", "warn", "error"]


def logger(message: str, logLevel: logLevelType = "info", noConsolePrint: bool = False):
    """
    Generates logs for the system in both command line and logger file
    Parameters
    ----------
    message: str
        A message to be shown in both logger file and command line
        example: "My Sample Message"
    logLevel: Literal, optional (default to "info)
        The level of logs. Possible values are "info", "warn", and "error"
        example: "warn"
    noConsolePrint: bool, optional (default to False)
        If True, the log will only be printed in the logger file
        example: True
    """
    # Create a console log by default
    if (not noConsolePrint):
        print(message)
    # Create a log in the log file
    currentMoment = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    printMessage = f'[{currentMoment}] {message}'
    if (logLevel is 'warn'):
        logging.warn(printMessage)
    elif (logLevel is 'error'):
        logging.error(printMessage)
    else:
        logging.info(printMessage)


def frameResizer(image, size):
    """
    Resizes the input frame into a desired width, while keeping the aspect ratio
    Parameters
    ----------
    message: CV image file
        The input image file
        example: "My Sample Message"
    logLevel: Literal, optional (default to "info)
        The level of logs. Possible values are "info", "warn", and "error"
        example: "warn"
    noConsolePrint: bool, optional (default to False)
        If True, the log will only be printed in the logger file
        example: True
    """
    # Calculating the dimensions
    imageHeight, imageWidth = image.shape[:2]
    aspectRatio = imageWidth / imageHeight
    # Resizing frame's width, while keeping its aspect ratio
    generatedImageW = size
    generatedImageH = int(generatedImageW / aspectRatio)
    # Scale the frame
    generatedImage = cv2.resize(
        image, (generatedImageW, generatedImageH), interpolation=cv2.INTER_AREA)
    return generatedImage


def emptyFolderRemover(mainDir: str):
    """
    Removes empty folders (like when no frames extracted from videos)
    Parameters
    ----------
    mainDir: string
        The parent directory containing empty and filled folders
        example: "C:/Some/Path"
    """
    folders = list(os.walk(mainDir))[1:]
    for folder in folders:
        if not folder[2]:
            os.rmdir(folder[0])
