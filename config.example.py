import os

# Frames Extraction
videosDir = 'C:/Some/Path/'
framesDir = 'C:/Some/Path/'
networkInputSize = 420  # The width of the extracted frames

# Feature Extraction
packetSize = 25  # Each 25 frames are considered as a packet
imagesDirectory = 'C:/Some/Path/'
# 1) Where the features should be stored, 2) Where to read feature folders for aggregation
extractedFeaturesDirectory = 'C:/Some/Path/'
aggregatedFeaturesDirectory = 'C:/Some/Path/'
