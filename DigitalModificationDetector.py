# Pledge: "I pledge my honor that I have abided by the Stevens Honor System"
# Signed Jonathan Kubas, Matthew De Le Pas, and Arun Mohan
# Final Project Assignment
# Necessary library imports
import os
import csv
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Features to hopefully be added later on
'''
# Function to perform ELA analysis method on a specific image
def elaAnalysis(imageFilePath):
    # Initialize ELA analysis result varaible
    elaAnalysisResult = 0
    
    # Open png version of inputted image path
    pngImage = cv2.imread(imageFilePath)
    
    # Convert png image to jpeg
    jpegImage = cv2.cvtColor(pngImage, cv2.COLOR_BGR2RGB)
    
    # Save a new compressed version of the previous jpeg image and store it
    tempImageName = "temporaryImage.jpg"
    cv2.imwrite(tempImageName, jpegImage, [cv2.IMWRITE_JPEG_QUALITY, 90])
    elaImage = cv2.imread(tempImageName)
    
    # Calcluate the difference between this new ELA image and the previous jpeg image
    elaDifference = cv2.absdiff(jpegImage, elaImage)
    grayScaleElaImage = cv2.cvtColor(elaDifference, cv2.COLOR_BGR2GRAY)
    
    # Initialize variables for image dimensions and threshold values
    pixelIntensityThreshold = 50
    numberOfDifferentPixelsThreshold = 50
    imageWidth = differenceBetweenImages.shape[1] 
    imageHeight = differenceBetweenImages.shape[0]
    
    # For loop to traverse through all the pixels in the image to count number of pixels above a certain threshold value
    numberOfPixelsAboveThreshold = 0
    for y in range(0, imageHeight):
        for x in range(0, imageWidth):
             if (differenceBetweenImages[x, y] >= pixelIntensityThreshold)
                numberOfPixelsAboveThreshold++

# Function to perform edge analysis method on a specific image
def edgeAnalysis(imageFilePath):
    image = cv2.imread(imageFilePath)
    
    grayScaledImage = cv2.cvtColor(image, COLOR_BGR2GRAY)
    
    edges = cv2.Canny(gray, 50, 150)
'''

# Function to perform noise analysis method on a specific image
def noiseAnalysis(imageFilePath, referenceImageFilePath):
    # Initialize necessary noise analysis results varialble
    noiseAnalysisResult = 0
    
    # Save the images into respective cv2 image variables
    image = cv2.imread(imageFilePath)
    referenceImage = cv2.imread(referenceImageFilePath)
    
    # Convert these images to a grayscale format
    grayScaleImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grayScaleReferenceImage = cv2.cvtColor(referenceImage, cv2.COLOR_BGR2GRAY)
    
    # Apply a median filter to these grayscaled images
    imageMedianFilter = cv2.medianBlur(grayScaleImage, 3)
    referenceMedianFilter = cv2.medianBlur(grayScaleReferenceImage, 3)

    # Get the difference between these two filter images, which will help us determine if there has been any modifications
    differenceBetweenImages = cv2.absdiff(imageMedianFilter, referenceMedianFilter)
    
    # Initialize variables for image dimensions and threshold values
    pixelIntensityThreshold = 100
    numberOfDifferentPixelsThreshold = 100
    imageWidth = differenceBetweenImages.shape[0] 
    imageHeight = differenceBetweenImages.shape[1]
    
    # For loop to traverse through all the pixels in the image to count number of pixels above a certain threshold value
    numberOfPixelsAboveThreshold = 0
    for y in range(0, imageHeight):
        for x in range(0, imageWidth):
             if (differenceBetweenImages[x, y] >= pixelIntensityThreshold):
                numberOfPixelsAboveThreshold += 1
    
    # If statement that checks if the number of different pixels in an image is over a certain amount and if so than image is possible photoshopped
    if (numberOfPixelsAboveThreshold > numberOfDifferentPixelsThreshold):
        noiseAnalysisResult = 1
    
    # Return the noiseAnalysisResult value that will be later stored in csv file
    return noiseAnalysisResult
    
# Function to perform histogramAnalysis method on a specific image
def histogramAnalysis(imageFilePath, referenceImageFilePath):
    # Initialize necessary histogram analysis results varialble
    histogramAnalysisResult = 0
    
    # Save the images into respective cv2 image variables
    image = cv2.imread(imageFilePath)
    referenceImage = cv2.imread(referenceImageFilePath)
    
    # Convert these images to a grayscale format
    grayScaleImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grayScaleReferenceImage = cv2.cvtColor(referenceImage, cv2.COLOR_BGR2GRAY)
    
    # Now calculate the histograms for both of these images
    imageHistogram = cv2.calcHist([grayScaleImage], [0], None, [256], [0, 256])
    referenceImageHistogram = cv2.calcHist([grayScaleReferenceImage], [0], None, [256], [0, 256])
    
    # Now calcluate the general difference between the two histograms
    differenceBetweenHistograms = cv2.compareHist(imageHistogram, referenceImageHistogram, cv2.HISTCMP_CHISQR)
    
    # Initalize difference threshold and peform if statement to see if difference is above threshold
    historgramDifferenceThreshold = 300
    if (differenceBetweenHistograms > historgramDifferenceThreshold):
        histogramAnalysisResult = 1
    
    # Finaly return the historgram anlaysis result value
    return histogramAnalysisResult

# Necessary variables for image storing process
orginalImagesFilePaths = []
photoshoppedImagesFilePaths = []
photoshoppedReferenceImagesFilePaths = []
orginalImageDirectoryFilePath = "C:\\Users\\jonat\\CPE_695_Final_Project\\Dataset\\original\\"
photoshoppedImageDirectoryFilePath = "C:\\Users\\jonat\\CPE_695_Final_Project\\Dataset\\modified\\"
photoshoppedReferenceImageDirectoryFilePath = "C:\\Users\\jonat\\CPE_695_Final_Project\\Dataset\\reference\\"

# Store original images file paths into a dataset array
listOfOrginalImagesFilePaths = os.listdir(orginalImageDirectoryFilePath)
for imageFilePath in listOfOrginalImagesFilePaths:
    orginalImagesFilePaths.append(orginalImageDirectoryFilePath + imageFilePath)

# Store photoshopped images file paths into a sperate dataset array
listOfPhotoshoppedImagesFilePaths = os.listdir(photoshoppedImageDirectoryFilePath)
for imageFilePath in listOfPhotoshoppedImagesFilePaths:
    photoshoppedImagesFilePaths.append(photoshoppedImageDirectoryFilePath + imageFilePath)

# Store photoshopped reference images file paths into another sperate dataset array
listOfPhotoshoppedReferenceImagesFilePaths = os.listdir(photoshoppedReferenceImageDirectoryFilePath)
for imageFilePath in listOfPhotoshoppedReferenceImagesFilePaths:
    photoshoppedReferenceImagesFilePaths.append(photoshoppedReferenceImageDirectoryFilePath + imageFilePath) 

# Create csv file that will be used to hold feature data for test images
with open('photoshop_image_data.csv', 'w', newline='') as imageDataFile:
    # Necessary variables for modification detection for the given images
    noiseAnalysisResults = 0
    histogramAnalysisResults = 0
    
    # Create writer object that will allow data to be placed into csv file
    writer = csv.writer(imageDataFile)
    
    # Write the first row into the given csv file (Acts as headers for rest of data)
    writer.writerow(["Noise_Analysis_Results", "Histogram_Analysis_Results", "Is_Image_Photoshopped"])

    # For loop that goes through each of the orginal images and calculates necessary analysis values
    for imageFilePath in orginalImagesFilePaths:
        # Store new noise analysis result value 
        noiseAnalysisResults = noiseAnalysis(imageFilePath, imageFilePath)
        
        # Store new noise analysis result value 
        histogramAnalysisResults = histogramAnalysis(imageFilePath, imageFilePath)
        
        # Add the various results and photoshoppedStatus values to the csv file
        writer.writerow([noiseAnalysisResults, histogramAnalysisResults, 0])
    
    # For loop that goes through each of the photoshopped images and calculates necessary analysis values
    for counter in range(len(photoshoppedImagesFilePaths)):
        # Store file paths of photoshopped and reference images into variables
        imageFilePath = photoshoppedImagesFilePaths[counter]
        referenceImageFilePath = photoshoppedReferenceImagesFilePaths[counter]
                    
        # Store new noise analysis result value 
        noiseAnalysisResults = noiseAnalysis(imageFilePath, referenceImageFilePath)
        
        # Store new noise analysis result value 
        histogramAnalysisResults = histogramAnalysis(imageFilePath, referenceImageFilePath)
        
        # Add the various results and photoshoppedStatus values to the csv file
        writer.writerow([noiseAnalysisResults, histogramAnalysisResults, 1])

# Now read the csv data into a csv dataframe
photoshopImageData = pd.read_csv("photoshop_image_data.csv")

# Split the photoshop data set into feature and target values
x = photoshopImageData["Noise_Analysis_Results", "Histogram_Analysis_Results"]
y = photoshopImageData["Is_Image_Photoshopped"]

# Split feature and target data into testing and training sets
xTraining, xTesting, yTraining, yTesting = train_test_split(x, y, test_size = 0.2)

# Create the decision tree model to tell whether a image is possible photoshopped or not
decisionTree = DecisionTreeClassifier(criterion = "entropy")
decisionTree = decisionTree.fit(xTraining, yTraining)

# Display the decision tree model that was just created
fig = plt.figure(figsize=((25,20)))
plot_tree(
    decisionTree,
    feature_names = x.columns,
    class_names= ["0", "1"],
    impurity=False,
    proportion=True,
    filled=True
)
fig.savefig('photoshopped_image_decision_tree.png')

# Calculate and print the accuracy of the model using the prediction and true y-values
yPredicted = decisionTree.predict(xTesting)
accuracyOfModel = accuracy_score(yTesting, yPredicted)
print("Accuracy:", accuracy)