from dataset_connector import load_dataset

trainX, trainY, testX, testY = load_dataset("../dataset")

print(trainX.shape)
print(trainY.shape)
print(testX.shape)
print(testY.shape)
