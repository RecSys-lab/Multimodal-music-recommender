import os
import cv2
import time
from config import videosDir, framesDir, networkInputSize
from utils import logger, frameResizer, emptyFolderRemover


# Extracts frames from a given list of videos
def frameExtractor():
    logger('Frames Extractor Started! (Assumption: FPS=1)')
    try:
        videoFiles = os.listdir(videosDir)
        # Filter only video files
        for file in videoFiles:
            if not file.lower().endswith(('.mkv', '.avi', '.mp4')):
                videoFiles.remove(file)
        logger(f'Number of videos: {len(videoFiles)} (FPS: 1)')
        # Create first the frame destination folder
        try:
            if not os.path.exists(framesDir):
                os.mkdir(framesDir)
        except FileExistsError as error:
            errorText = str(error)
            logger(
                f'Error while creating the folder ({errorText})', logLevel="error")
        finally:
            # Iterate on all video files in the given directory
            for file in videoFiles:
                # Accessing video and creating output folder
                videoPath = f'{videosDir}/{file}'
                videoName = file.split('.')[0]
                generatedPath = f'{framesDir}/{videoName}'
                # Do not re-generate frames for videos if there is a folder with their normalized name
                if os.path.exists(generatedPath):
                    print(f'Skipping {file} as it has been extracted before!')
                else:
                    os.mkdir(generatedPath)
                    # Capturing video
                    try:
                        frameCounter = 1
                        startTime = time.time()
                        capturedVideo = cv2.VideoCapture(videoPath)
                        if not capturedVideo.isOpened():
                            logger(f'Error while reading the content of {file}!',
                                   logLevel="error")
                        success, image = capturedVideo.read()
                        while success:
                            # Resizing the image, while preserving its aspect-ratio
                            image = frameResizer(image, networkInputSize)
                            # Format the frame counter as: frame1 --> frame0000001
                            formattedCounter = '{0:07d}'.format(frameCounter)
                            # Save the frame as a file
                            cv2.imwrite(
                                f"{framesDir}/{videoName}/frame{formattedCounter}.jpg", image)
                            success, image = capturedVideo.read()
                            # Showing progress every one minute of the video
                            if (frameCounter % 60 == 0):
                                print(f'Processing frame #{frameCounter} ...')
                            frameCounter += 1
                        # Finished extracting frames
                        elapsedTime = '{:.2f}'.format(time.time() - startTime)
                        logger(
                            f'Extracted {frameCounter} frames of {videoName} in {elapsedTime} seconds!')
                    except Exception as error:
                        errorText = str(error)
                        logger(
                            f'Unexpected error: {errorText}', logLevel="error")
            print(f'Now, removing empty folders in {framesDir}')
            emptyFolderRemover(framesDir)
    except FileNotFoundError:
        logger(
            f'The input directory does not exist or contain video files!', logLevel="error")
    except Exception as error:
        errorText = str(error)
        logger(
            f'Error while running the app ({errorText})', logLevel="error")
