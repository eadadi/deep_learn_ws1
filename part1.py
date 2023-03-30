from get_data import DataHandler as gd
from sklearn import svm

"""
Load data batches, Init XY_train, XY_test
"""
TrainingBatch, TestBatch = gd.load_and_sample_data(training = 5000, test = 1000)
XTrain, YTrain = TrainingBatch[0], TrainingBatch[1]
XTest, YTest = TestBatch[0], TestBatch[1]

"""
Train a linear classifier that uses 'one vs one' method
"""
linear = svm.SVC(kernel='linear')
linear.fit(XTrain, YTrain)

"""
Test linear classifier results
"""
YLinear = linear.predict(XTest)
LinearResults = (YLinear==YTest).sum()/len(YTest)

"""
Test linear classifier over training sample
"""
YLinearTrain = linear.predict(XTrain)
LinearTrainResults = (YLinearTrain==YTrain).sum()/len(YTrain)

"""
Train a rbf-kernel classifier that uses 'one vs one' method
"""
rbf = svm.SVC(kernel='rbf')
rbf.fit(XTrain, YTrain)

"""
Test linear classifier results
"""
YRbf = rbf.predict(XTest)
RbfResults = (YRbf==YTest).sum()/len(YTest)

"""
Test linear classifier over training sample
"""
YRbfTrain = rbf.predict(XTrain)
RbfTrainResults = (YRbfTrain==YTrain).sum()/len(YTrain)

"""
Dump results to stdout
"""
NEW_LINE = "\n"
print("Part A".center(50,'='),
        NEW_LINE * 2,
        "Linear Results\n",
        f"Training Sample Results: {LinearTrainResults}\n",
        f"Test Sample Results: {LinearResults}\n",
        NEW_LINE,
        "Rbf Results\n",
        f"Training Sample Results: {RbfTrainResults}\n",
        f"Test Sample Results: {RbfResults}\n",
        NEW_LINE
)
