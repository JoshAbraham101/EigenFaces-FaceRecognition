import numpy as np
from numpy import linalg as LA
from PIL import Image
from matplotlib import pyplot as plt
np.set_printoptions(threshold='nan')

# Training Stage
# M Training Images set
trainingSet = ['subject01.normal.jpg', 'subject02.normal.jpg', 'subject03.normal.jpg', 'subject07.normal.jpg',
               'subject10.normal.jpg', 'subject11.normal.jpg', 'subject14.normal.jpg', 'subject15.normal.jpg']
width, height = Image.open(trainingSet[1]).size

# Stack pixel rows of each image together for each training image
def stackImageRows(set):
    trainingImages = np.zeros((width * height, len(set)), dtype=np.int32)
    for img in range(len(set)):
        index = 0
        imageObject = Image.open(set[img])
        for i in range(height):
            for j in range(width):
                trainingImages[index][img] = imageObject.getpixel((j,i))
                index+=1

    return trainingImages

trainingImages = stackImageRows(trainingSet)

# Find mean face m by taking average of the M training images
def findMeanFace():
    sum = 0
    meanFace = np.zeros((width*height,1),dtype=np.int32)
    for i in range(trainingImages.shape[0]):
        sum = 0
        for j in range(trainingImages.shape[1]):
            sum = sum + trainingImages[i][j]
        mean = sum/(trainingImages.shape[1])
        meanFace[i][0] = mean

    return meanFace

meanFaceArray = findMeanFace()
img = Image.new('L', (width,height), color=0)
img.putdata(meanFaceArray[:,0])
img.save('Mean Face.jpg')

# Subtract mean face from each training face to get matrix A of dimension (width*height)xM
def getMeanDifference(set):
    meanDifferece = np.zeros((set.shape[0], set.shape[1]), dtype=np.int32)
    for i in range(set.shape[0]):
        for j in range(set.shape[1]):
            meanDifferece[i][j] = set[i][j] - meanFaceArray[i][0]
    return meanDifferece

meanDiffereceMatrix = getMeanDifference(trainingImages)

# Alternative to finding covariance matrix, to get matrix L
covarianceLMatrix = np.zeros((meanDiffereceMatrix.shape[1], meanDiffereceMatrix.shape[1]), dtype=np.int32)
covarianceLMatrix = np.transpose(meanDiffereceMatrix).dot(meanDiffereceMatrix)

# Get eigenVectors of L
eVals, eVecs = LA.eig(covarianceLMatrix)

# Get eigenVectors of Covariance matrix and find U (EigenFaces)
U = np.zeros((meanDiffereceMatrix.shape[0], eVecs.shape[1]), dtype=np.int32)
U = meanDiffereceMatrix.dot(eVecs)

# Save all M EigenFace images from U
def printEigenFaceImages():
    for k in range(U.shape[1]):
        plt.title('Eighen face' + str(k+1))
        plt.imshow((U[:, k].reshape(231, 195)), cmap='gray')
        plt.savefig('EigenFace'+str(k+1)+'.png')


printEigenFaceImages()

# Project U onto face space (PCA Coefficients)
faceSpaceMatrix = np.zeros((U.shape[1], meanDiffereceMatrix.shape[1]), dtype=np.int32)
faceSpaceMatrix = np.transpose(U).dot(meanDiffereceMatrix)

print "PCA Coefficients for each training image"
for i in range(faceSpaceMatrix.shape[1]):
    print "PCA Coefficients for training image "+str(i+1)+" : "
    print faceSpaceMatrix[:,i]
#=============================================================================================
# Recognition Stage
# Test images set
testimageNames = [['subject01.centerlight.jpg'], ['subject01.happy.jpg'], ['subject01.normal.jpg'],
                  ['subject02.normal.jpg'], ['subject03.normal.jpg'], ['subject07.centerlight.jpg'],
                  ['subject07.happy.jpg'], ['subject07.normal.jpg'], ['subject10.normal.jpg'],
                  ['subject11.centerlight.jpg'], ['subject11.happy.jpg'], ['subject11.normal.jpg'],
                  ['subject12.normal.jpg'], ['subject14.happy.jpg'], ['subject14.normal.jpg'], ['subject14.sad.jpg'],
                  ['subject15.normal.jpg'], ['apple1_gray.jpg']]
index = 0
T0 = 6700000000000      # Threshold to detect non face
T1 = 89000000      # Threshold to identify face

for i in testimageNames:
    # Input Test image
    testImgName = i
    testImg = stackImageRows(testImgName)

    # Subtract mean face from input face I
    I = getMeanDifference(testImg)
    plt.title('SubtractedImage'+str(index+1))
    plt.imshow((I[:, 0].reshape(231, 195)), cmap='gray')
    plt.savefig('SubtractedImage'+str(index+1)+'.png')

    # Project onto face space (PCA coefficients)
    inputFaceSpace = np.transpose(U).dot(I)
    print "PCA Coefficients of test image '"+str(i[0])+"'"
    print inputFaceSpace

    # Reconstructed input face image
    reconstructedInput = U.dot(inputFaceSpace)
    plt.title('ReconstructedImage'+str(index+1))
    plt.imshow((reconstructedInput[:, 0].reshape(231, 195)), cmap='gray')
    plt.savefig('ReconstructedImage'+str(index+1)+'.png')

    index += 1
    # Calculating d0 value from reconstructed image
    def reconstructedImageDistace():
        d0 = LA.norm(np.subtract(reconstructedInput, I))
        return d0

    d0 =reconstructedImageDistace()
    print "d0 for test image '" + str(i[0]) + "' : "+str(d0)
    # Check with threshold T0 for non face
    if d0>T0:
        print "Classification: Non-Face\nThe input test image '"+str(i[0])+"' is not a face image"
        print "============================="
        continue

    # Find distance between input image and face image, di
    def faceSpaceDistance():
        distArray = []
        for i in range(faceSpaceMatrix.shape[1]):
            diff = LA.norm(np.subtract(np.transpose(inputFaceSpace), np.transpose(faceSpaceMatrix)[i]))
            distArray.append(diff)
        return distArray

    dist = faceSpaceDistance()
    print "di for test image '"+str(i[0])+"' : "
    print dist

    dj = min(dist)
    print "Minimum distance : "+str(dj)

    # Check with threshold T0 for non face
    if dj >= T1:
        print "Classification: Unknown Face\nThe input test image '"+str(i[0])+"' is an unrecognised face"
        print "============================="
        continue

    print "Classification: Known Face\nThe input test image '"+str(i[0])+"' is matched with '"+str(trainingSet[dist.index(min(dist))])+"' training image"
    print "============================="