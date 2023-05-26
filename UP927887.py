import cv2
import numpy as np
import os
import glob
import sklearn
from sklearn.model_selection import train_test_split, KFold
from sklearn.cluster import KMeans
import pickle
from scipy.spatial.distance import cdist
import time
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neural_network import MLPClassifier

# Testing MHI
globMhi = None

# Generate MHI
def generate_mhi(fileName):
    history_duration = 5 
    motion_threshold = 25
    frame_width, frame_height = 640,640
    motion_history = np.zeros((frame_height, frame_width), dtype=np.float32)
    last_frame = None
    motion_accumulator = np.zeros_like(motion_history, dtype=np.float32)
    cap = cv2.VideoCapture(fileName)

    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.resize(gray_frame, [640,640])
        
        if last_frame is None:
            last_frame = gray_frame
            continue
        
        frame_diff = cv2.absdiff(last_frame, gray_frame)
        _, threshold = cv2.threshold(frame_diff, motion_threshold, 255, cv2.THRESH_BINARY)
        motion_accumulator = cv2.addWeighted(motion_accumulator, 0.5, threshold.astype(np.float32), 0.5, 0, dtype=cv2.CV_32F)
        motion_history = cv2.max(motion_history - (1.0 / history_duration), motion_accumulator)
        last_frame = gray_frame

    return motion_history

# Save MHI to directory
def saveMHI(rawVideoPath):
    folderNames = os.listdir(rawVideoPath)
    # print(folderNames)
    mhiSaveFolder = "database/mhiImages"
    for folderPose in folderNames:
        if ".txt" in folderPose:
            print("NO TEXT FILES HERE!!")
            continue
        # print(folderPose)
        comFPath = rawVideoPath+"/"+folderPose
        with open(comFPath+".txt") as file:
            for fname in file:
                fname = fname.strip()
                toGenerate = comFPath+"/"+fname
                # print(toGenerate)

                videoMHI = generate_mhi(toGenerate)
                mhi2Img = cv2.normalize(videoMHI,None,0,255,cv2.NORM_MINMAX,cv2.CV_8U)

                fname = fname.replace(".avi", "")          
                toMHISaveFolder = mhiSaveFolder+"/"+folderPose+"/"+fname+".jpg"
                # print(toMHISaveFolder)
                cv2.imwrite(toMHISaveFolder,mhi2Img)
                # print(fname,"saved")

# Create labels based on folder names
def labelImage(mhiFolderPath):
    allLabels = os.listdir(mhiFolderPath)

    label2id = {}
    for labelID, labelValue in enumerate(allLabels):
        label2id[labelValue] = labelID

    dataImage = []
    dataLabel = []

    for label in allLabels:
        for mhi in glob.glob(mhiFolderPath+"/"+label+'/*'):
            currentMHI = cv2.imread(mhi)
            currentLabel = label2id[label]

            dataImage.append(currentMHI)
            dataLabel.append(currentLabel)
        # print("label appended to " + label)
    
    return dataImage, dataLabel, label2id

# SIFT/ORB Detection/Extraction
def extractFeatures(imageToExtract):
    imageFeatures = []
    detect = cv2.SIFT_create()
    # detect = cv2.ORB_create()


    for image in imageToExtract:
        _, feature = detect.detectAndCompute(image,None)
        imageFeatures.append(feature)

        # img_copy = image.copy()
        # keypoints, img_copy_kp = detect.detectAndCompute(img_copy, None)
        # img_kp = cv2.drawKeypoints(img_copy,keypoints,cv2.DrawMatchesFlags_DEFAULT)
        # cv2.imshow("detection", img_kp)
        # cv2.waitKey(0)
        # print("image Features appended to")
    
    return imageFeatures

#KMeans Clustering
def kMeansBOW(allFeatures, nC):
    bagOfWords = []

    km = KMeans(n_clusters=nC)
    km.fit(allFeatures)

    bagOfWords = km.cluster_centers_

    if not os.path.isfile('Coursework/bowDictionary.pkl'):
        pickle.dump(bagOfWords, open('Coursework/bowDictionary.pkl', 'wb'))
    
    return bagOfWords

# Calculate Distance for SVM
def createFeature(mhiFeatures, bow, clusters):
    feature = []
    for i in range(len(mhiFeatures)):
        features = np.array([0]* clusters)

        if mhiFeatures[i] is not None:
            dist = cdist(mhiFeatures[i], bow)

            argmin = np.argmin(dist, axis = 1)

            for j in argmin:
                features[j] += 1
        
        feature.append(features)
    
    return feature


def main():
    t1 = time.time()
    rawFolder = 'database/data'
    mhiFolder = 'database/mhiImages'
    # saveMHI(rawFolder)

    t2 = time.time()
    print("MHI Processed in ", (t2-t1))

    mhiToTrain, labels, label2id = labelImage(mhiFolder)
    mhiFeatures = extractFeatures(mhiToTrain)

    t3 = time.time()
    print("Labels and Features Processed in ", (t3-t2))

    allFeatures = []
    for feature in mhiFeatures:
        if feature is not None:
            for fe in feature:
                allFeatures.append(fe)
    
    numClusters = 60
    bagOfWords = kMeansBOW(allFeatures, numClusters)

    t4 = time.time()
    print("BOW Processed in ", (t4-t3))

    bowFeatures = createFeature(mhiFeatures, bagOfWords, numClusters)
    X_train, X_test, Y_train, Y_test = train_test_split(bowFeatures, labels, test_size = 0.2, random_state=1)



    nFolds = 5
    kfolds = KFold(n_splits=nFolds, shuffle=True, random_state=1)

    acc_scores = []

    for train_in, test_in in kfolds.split(bowFeatures):
        xtra = [bowFeatures[i] for i in train_in]
        xtes = [bowFeatures[i] for i in test_in]
        ytra = [labels[i] for i in train_in]
        ytes = [labels[i] for i in test_in]

        # model_svm = sklearn.svm.SVC(C=30, random_state = 0)
        model_svm = MLPClassifier(random_state=1, max_iter=300)
        model_svm.fit(xtra, ytra)

        Y_pred = model_svm.predict(xtes)

        acc = accuracy_score(ytes,Y_pred)
        acc_scores.append(acc)





    # model_svm = sklearn.svm.SVC(C = 30, random_state = 0)
    # model_svm.fit(X_train, Y_train)

    t5 = time.time()
    print("Model Trained in", (t5-t4))

    # Y_pred = model_svm.predict(X_test)

    # acc = accuracy_score(Y_test,Y_pred)*100
    # conf = confusion_matrix(Y_test,Y_pred)


    avg_acc = sum(acc_scores)/nFolds
    t6 = time.time()

    print("Accuracy ",avg_acc)
    # print("Confusion Matrix")
    # print(conf)

    print("Total Time ",(t6-t1))

    # print("training set: ", model_svm.score(X_train,Y_train))
    # print("testing set: ", model_svm.score(X_test,Y_test))




main()

