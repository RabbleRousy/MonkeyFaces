- [Train](#train)
  - [Train on Full Dataset](#train-on-full-dataset)
  - [Train on Selected Dataset](#train-on-selected-dataset)
- [Predict](#predict)
- [Transfer learning / fine-tuning](#transfer-learning--fine-tuning)
  - [Train, validate and test on full dataset](#train-validate-and-test-on-full-dataset)
  - [Train and test on selected dataset](#train-and-test-on-selected-dataset)
    - [Choices in exp5](#choices-in-exp5)
- [Training Logs](#training-logs)
  - [exp1](#exp1)
  - [exp2](#exp2)
  - [exp3](#exp3)
  - [exp4](#exp4)
  - [exp5](#exp5)
  - [exp6](#exp6)
  - [exp7](#exp7)
  - [exp8](#exp8)
  - [exp9](#exp9)
  - [exp10](#exp10)
  - [exp11](#exp11)
  - [exp12](#exp12)
- [Notes](#notes)

# Train
- Notice: dummy category is `Pet’79'87'00'08'14`
## Train on Full Dataset
- `python .\train.py -nc 122 -es 20 -bs 128 -trp E:\datasets\monkeys\facedata_yamada\facedata_yamada\train_Magface --train --val --test`
## Train on Selected Dataset
- Obtain partial dataset: `python utils.py`
- `python .\train.py -nc 28 -es 20 -bs 128 -trp E:\datasets\monkeys\demo --train --test`

# Predict
- Method 1: `python predict.py`
- Method 2: `python .\train.py -nc 122 -bs 32 -trp E:\datasets\monkeys\facedata_yamada\facedata_yamada\train_Magface -tep E:\datasets\monkeys\facedata_yamada\facedata_yamada\test_Magface --checkpoint E:\ws\MonkeyFace\Prototypes\Zhang\logs\exp9\weights\0_0.9067371937639198.pth.tar --test`
  - Another dir: `python .\train.py -nc 122 -bs 32 -trp E:\datasets\monkeys\facedata_yamada\facedata_yamada\train_Magface -tep E:\datasets\monkeys\facedata_yamada\test_dataset --checkpoint E:\ws\MonkeyFace\Prototypes\Zhang\logs\exp9\weights\0_0.9067371937639198.pth.tar  --test`
  - exp11:  `python .\train.py -nc 122 -bs 32 -trp E:\datasets\monkeys\facedata_yamada\facedata_yamada\train_Magface -tep E:\datasets\monkeys\facedata_yamada\test_dataset --checkpoint E:\ws\MonkeyFace\Prototypes\Zhang\logs\exp11\weights\0_0.9893426364142539.pth.tar  --test`
# Transfer learning / fine-tuning
- Set initial training epoch to 20, early stopping will stop training automatically
## Train, validate and test on full dataset
- Based on exp5： `python train.py -es 1 -nc 122 -bs 128 -tep E:\datasets\monkeys\facedata_yamada\facedata_yamada\train_Magface --checkpoint E:\ws\MonkeyFace\Prototypes\Zhang\logs\exp5\weights\12_0.9711587778429074.pth.tar --train --val --test `
- Based on exp9： `python train.py -es 1 -nc 122 -bs 128 -tep E:\datasets\monkeys\facedata_yamada\facedata_yamada\train_Magface --checkpoint E:\ws\MonkeyFace\Prototypes\Zhang\logs\exp9\weights\0_0.9067371937639198.pth.tar --train --val --test `
## Train and test on selected dataset
- `python train.py -es 20 -nc 28 -bs 128 -tep E:\datasets\monkeys\demo --checkpoint E:\ws\MonkeyFace\Prototypes\Zhang\logs\exp5\weights\12_0.9711587778429074.pth.tar --train --test `
### Choices in exp5
- 10_0.9682371043376319.pth.tar
- 12_0.9711587778429074.pth.tar
- 14_0.9736041910902696.pth.tar

# Training Logs
## exp1
- vgg16 on full dataset, stops in epoch 11 (2.3 days) because I forget logging accuracy and f1-score
## exp2
- vgg16 on demo dataset (29 categories)
- Complete metrics
- **Wrong nc parameter, causing high accuracy and f1-score**
## exp3
- vgg 16 on demo dataset (28 categories)
- Complete metrics
- Model is influenced by data imbalance dramatically
- Perform not so good in test dataset

| | |
| --- | --- |
| Loss | 0.088 |
| Accuracy | 0.970 |
| F1-score | 0.956 |
- Visualize the predictions of this model, BAD.
  - Caused by <font color=red>category imbalance</font>.
  - The category names start with "Pet" usually have better performance than others.
  - **Visualize the metrics of each category respectively is necessary**.

## exp4
- Implement validation and testing process, and then train on partial dataset.
- The performance is bad because val/test set contains some cetegories not appearing in the partial training set.
- Skip this exp

## exp5
- Train VGG on the full dataset
- Finished
```txt
Training logs
train set length: 109232 | val set length: 5829 | test set length: 5829
training starts!
Epoch 0| Loss: 2.20050 | Accuracy: 0.443 | F1-score: 0.468 | time: 17313.537s
Validation 0 | Loss: 1.13213 | Accuracy: 0.701 | F1-score: 0.508
Epoch 1| Loss: 0.82358 | Accuracy: 0.778 | F1-score: 0.619 | time: 17279.301s
Epoch 2| Loss: 0.46628 | Accuracy: 0.874 | F1-score: 0.677 | time: 17201.227s
Epoch 3| Loss: 0.31870 | Accuracy: 0.912 | F1-score: 0.742 | time: 17193.508s
Validation 3 | Loss: 0.27317 | Accuracy: 0.929 | F1-score: 0.846
Epoch 4| Loss: 0.24502 | Accuracy: 0.931 | F1-score: 0.830 | time: 17222.053s
Epoch 5| Loss: 0.20076 | Accuracy: 0.944 | F1-score: 0.852 | time: 17263.944s
Epoch 6| Loss: 0.17634 | Accuracy: 0.950 | F1-score: 0.926 | time: 17243.440s
Validation 6 | Loss: 0.10777 | Accuracy: 0.974 | F1-score: 0.941
Epoch 7| Loss: 0.14819 | Accuracy: 0.957 | F1-score: 0.921 | time: 17240.434s
Epoch 8| Loss: 0.13196 | Accuracy: 0.962 | F1-score: 0.858 | time: 17233.743s
Epoch 9| Loss: 0.12468 | Accuracy: 0.964 | F1-score: 0.932 | time: 17311.721s
Validation 9 | Loss: 0.09472 | Accuracy: 0.975 | F1-score: 0.939
Epoch 10| Loss: 0.11175 | Accuracy: 0.968 | F1-score: 0.881 | time: 16979.952s
Epoch 11| Loss: 0.10726 | Accuracy: 0.969 | F1-score: 0.824 | time: 16978.050s
Epoch 12| Loss: 0.09984 | Accuracy: 0.971 | F1-score: 0.990 | time: 16977.623s
Validation 12 | Loss: 0.08785 | Accuracy: 0.977 | F1-score: 0.945
Epoch 13| Loss: 0.09379 | Accuracy: 0.973 | F1-score: 0.967 | time: 16979.206s
Epoch 14| Loss: 0.09257 | Accuracy: 0.974 | F1-score: 0.929 | time: 16971.225s
Epoch 15| Loss: 0.08807 | Accuracy: 0.975 | F1-score: 0.892 | time: 16979.608s
Validation 15 | Loss: 0.11179 | Accuracy: 0.969 | F1-score: 0.932
Testing:
Tests | Loss: 19.79298 | Accuracy: 0.080 | F1-score: 0.015
training finished
```
- Test performance is unexpected low
  - ✅ Potential BUG: test label is not consistent with train/validation label
  - ❌ Or, the model is overfitting. Try to use earlier stage model parameters
## exp6
- random seed: 42
- Train on selected dataset with normalization
```txt
train set length: 2480 | val set length: 1017 | test set length: 1017
training starts!
Epoch 0| Loss: 3.11573 | Accuracy: 0.105 | F1-score: 0.022 | time: 412.898s
Epoch 1| Loss: 2.47712 | Accuracy: 0.272 | F1-score: 0.121 | time: 399.681s
Epoch 2| Loss: 1.76291 | Accuracy: 0.481 | F1-score: 0.301 | time: 399.650s
Epoch 3| Loss: 1.20885 | Accuracy: 0.625 | F1-score: 0.478 | time: 400.258s
Epoch 4| Loss: 0.85590 | Accuracy: 0.730 | F1-score: 0.607 | time: 399.579s
Epoch 5| Loss: 0.69368 | Accuracy: 0.784 | F1-score: 0.684 | time: 399.610s
Epoch 6| Loss: 0.53960 | Accuracy: 0.836 | F1-score: 0.759 | time: 399.759s
Epoch 7| Loss: 0.47232 | Accuracy: 0.854 | F1-score: 0.800 | time: 400.942s
Epoch 8| Loss: 0.35306 | Accuracy: 0.887 | F1-score: 0.816 | time: 399.819s
Epoch 9| Loss: 0.30753 | Accuracy: 0.904 | F1-score: 0.850 | time: 399.649s
Epoch 10| Loss: 0.24296 | Accuracy: 0.921 | F1-score: 0.878 | time: 400.753s
Epoch 11| Loss: 0.23382 | Accuracy: 0.924 | F1-score: 0.885 | time: 399.768s
Epoch 12| Loss: 0.19389 | Accuracy: 0.934 | F1-score: 0.899 | time: 399.884s
Epoch 13| Loss: 0.17391 | Accuracy: 0.946 | F1-score: 0.915 | time: 399.893s
Epoch 14| Loss: 0.16987 | Accuracy: 0.940 | F1-score: 0.911 | time: 399.499s
Epoch 15| Loss: 0.13408 | Accuracy: 0.959 | F1-score: 0.939 | time: 399.558s
Epoch 16| Loss: 0.12290 | Accuracy: 0.963 | F1-score: 0.938 | time: 399.774s
Epoch 17| Loss: 0.09094 | Accuracy: 0.968 | F1-score: 0.945 | time: 399.604s
Epoch 18| Loss: 0.08315 | Accuracy: 0.974 | F1-score: 0.957 | time: 399.555s
Epoch 19| Loss: 0.11316 | Accuracy: 0.961 | F1-score: 0.936 | time: 399.860s
training finished
```
## exp7
- random seed: 42
- Train on selected dataset without normalization
```txt
train set length: 2480 | val set length: 1017 | test set length: 1017
training starts!
Epoch 0| Loss: 3.17678 | Accuracy: 0.070 | F1-score: 0.011 | time: 423.535s
Epoch 1| Loss: 2.81838 | Accuracy: 0.203 | F1-score: 0.071 | time: 405.793s
Epoch 2| Loss: 2.18535 | Accuracy: 0.372 | F1-score: 0.196 | time: 405.626s
Epoch 3| Loss: 1.63581 | Accuracy: 0.511 | F1-score: 0.347 | time: 405.542s
Epoch 4| Loss: 1.26918 | Accuracy: 0.625 | F1-score: 0.478 | time: 405.473s
Epoch 5| Loss: 0.95279 | Accuracy: 0.701 | F1-score: 0.558 | time: 405.647s
Epoch 6| Loss: 0.73360 | Accuracy: 0.782 | F1-score: 0.685 | time: 405.555s
Epoch 7| Loss: 0.63134 | Accuracy: 0.800 | F1-score: 0.710 | time: 405.627s
Epoch 8| Loss: 0.48424 | Accuracy: 0.847 | F1-score: 0.756 | time: 405.834s
Epoch 9| Loss: 0.44736 | Accuracy: 0.855 | F1-score: 0.771 | time: 406.640s
Epoch 10| Loss: 0.35683 | Accuracy: 0.886 | F1-score: 0.828 | time: 408.103s
Epoch 11| Loss: 0.33546 | Accuracy: 0.891 | F1-score: 0.830 | time: 407.452s
Epoch 12| Loss: 0.30643 | Accuracy: 0.901 | F1-score: 0.833 | time: 426.890s
Epoch 13| Loss: 0.24339 | Accuracy: 0.917 | F1-score: 0.866 | time: 412.622s
Epoch 14| Loss: 0.25816 | Accuracy: 0.926 | F1-score: 0.882 | time: 410.584s
Epoch 4| Loss: 1.26918 | Accuracy: 0.625 | F1-score: 0.478 | time: 405.473s
Epoch 5| Loss: 0.95279 | Accuracy: 0.701 | F1-score: 0.558 | time: 405.647s
Epoch 6| Loss: 0.73360 | Accuracy: 0.782 | F1-score: 0.685 | time: 405.555s
Epoch 7| Loss: 0.63134 | Accuracy: 0.800 | F1-score: 0.710 | time: 405.627s
Epoch 8| Loss: 0.48424 | Accuracy: 0.847 | F1-score: 0.756 | time: 405.834s
Epoch 9| Loss: 0.44736 | Accuracy: 0.855 | F1-score: 0.771 | time: 406.640s
Epoch 10| Loss: 0.35683 | Accuracy: 0.886 | F1-score: 0.828 | time: 408.103s
Epoch 11| Loss: 0.33546 | Accuracy: 0.891 | F1-score: 0.830 | time: 407.452s
Epoch 12| Loss: 0.30643 | Accuracy: 0.901 | F1-score: 0.833 | time: 426.890s
Epoch 13| Loss: 0.24339 | Accuracy: 0.917 | F1-score: 0.866 | time: 412.622s
Epoch 14| Loss: 0.25816 | Accuracy: 0.926 | F1-score: 0.882 | time: 410.584s
Epoch 10| Loss: 0.35683 | Accuracy: 0.886 | F1-score: 0.828 | time: 408.103s
Epoch 11| Loss: 0.33546 | Accuracy: 0.891 | F1-score: 0.830 | time: 407.452s
Epoch 12| Loss: 0.30643 | Accuracy: 0.901 | F1-score: 0.833 | time: 426.890s
Epoch 13| Loss: 0.24339 | Accuracy: 0.917 | F1-score: 0.866 | time: 412.622s
Epoch 14| Loss: 0.25816 | Accuracy: 0.926 | F1-score: 0.882 | time: 410.584s
Epoch 12| Loss: 0.30643 | Accuracy: 0.901 | F1-score: 0.833 | time: 426.890s
Epoch 13| Loss: 0.24339 | Accuracy: 0.917 | F1-score: 0.866 | time: 412.622s
Epoch 14| Loss: 0.25816 | Accuracy: 0.926 | F1-score: 0.882 | time: 410.584s
Epoch 15| Loss: 0.23618 | Accuracy: 0.928 | F1-score: 0.884 | time: 411.914s
Epoch 15| Loss: 0.23618 | Accuracy: 0.928 | F1-score: 0.884 | time: 411.914s
Epoch 16| Loss: 0.17868 | Accuracy: 0.941 | F1-score: 0.901 | time: 410.334s
Epoch 17| Loss: 0.18043 | Accuracy: 0.940 | F1-score: 0.906 | time: 407.317s
Epoch 18| Loss: 0.18763 | Accuracy: 0.936 | F1-score: 0.895 | time: 407.726s
Epoch 18| Loss: 0.18763 | Accuracy: 0.936 | F1-score: 0.895 | time: 407.726s
Epoch 19| Loss: 0.19479 | Accuracy: 0.935 | F1-score: 0.906 | time: 413.853s
training finished
```
## exp8
- Train and **log the metrics** on selected dataset
## exp9
- Fine-tuning exp5 and log the metrics on full dataset
## exp10
- results to draw pics in slide
## exp11
- Fine-tuning with logging and storing data
```txt
python train.py -es 1 -nc 122 -bs 128 -tep E:\datasets\monkeys\facedata_yamada\facedata_yamada\train_Magface --checkpoint E:\ws\MonkeyFace\Prototypes\Zhang\logs\exp9\weights\0_0.9067371937639198.pth.tar --train --val --test
train set length: 115061 | val set length: 57531 | test set length: 57530
training starts!
Epoch 0| Loss: 0.03594 | Accuracy: 0.989 | F1-score: 0.975 | time: 35810.914s
Validation 0 | Loss: 0.04602 | Accuracy: 0.987 | F1-score: 0.971
Training finished
Test 0/449 | Loss: 0.03160 | Accuracy: 0.992 | F1-score: 0.958
Test 1/449 | Loss: 0.04120 | Accuracy: 0.992 | F1-score: 0.958
Test 2/449 | Loss: 0.10664 | Accuracy: 0.969 | F1-score: 0.918
Test 3/449 | Loss: 0.09985 | Accuracy: 0.977 | F1-score: 0.953
Test 4/449 | Loss: 0.07095 | Accuracy: 0.992 | F1-score: 0.994
Test 5/449 | Loss: 0.07797 | Accuracy: 0.984 | F1-score: 0.954
Test 6/449 | Loss: 0.01232 | Accuracy: 1.000 | F1-score: 1.000
Test 7/449 | Loss: 0.02295 | Accuracy: 0.984 | F1-score: 0.945
Test 8/449 | Loss: 0.01993 | Accuracy: 0.992 | F1-score: 0.978
Test 9/449 | Loss: 0.06563 | Accuracy: 0.984 | F1-score: 0.981
Test 10/449 | Loss: 0.06888 | Accuracy: 0.984 | F1-score: 0.970
Test 11/449 | Loss: 0.09099 | Accuracy: 0.969 | F1-score: 0.957
Test 12/449 | Loss: 0.02992 | Accuracy: 0.992 | F1-score: 0.992
Test 13/449 | Loss: 0.02665 | Accuracy: 0.992 | F1-score: 0.976
Test 14/449 | Loss: 0.04856 | Accuracy: 0.992 | F1-score: 0.995
Test 15/449 | Loss: 0.03221 | Accuracy: 0.984 | F1-score: 0.986
Test 16/449 | Loss: 0.00836 | Accuracy: 1.000 | F1-score: 1.000
Test 17/449 | Loss: 0.00918 | Accuracy: 1.000 | F1-score: 1.000
Test 18/449 | Loss: 0.02018 | Accuracy: 1.000 | F1-score: 1.000
Test 19/449 | Loss: 0.02185 | Accuracy: 0.992 | F1-score: 0.979
Test 20/449 | Loss: 0.00655 | Accuracy: 1.000 | F1-score: 1.000
Test 21/449 | Loss: 0.00960 | Accuracy: 1.000 | F1-score: 1.000
Test 22/449 | Loss: 0.00961 | Accuracy: 1.000 | F1-score: 1.000
Test 23/449 | Loss: 0.04795 | Accuracy: 0.977 | F1-score: 0.954
Test 24/449 | Loss: 0.02893 | Accuracy: 0.992 | F1-score: 0.974
Test 25/449 | Loss: 0.02203 | Accuracy: 0.992 | F1-score: 0.991
Test 26/449 | Loss: 0.02206 | Accuracy: 0.992 | F1-score: 0.988
Test 27/449 | Loss: 0.07179 | Accuracy: 0.977 | F1-score: 0.967
Test 28/449 | Loss: 0.01617 | Accuracy: 0.992 | F1-score: 0.975
Test 29/449 | Loss: 0.00511 | Accuracy: 1.000 | F1-score: 1.000
Test 30/449 | Loss: 0.01056 | Accuracy: 0.992 | F1-score: 0.976
Test 31/449 | Loss: 0.02051 | Accuracy: 0.992 | F1-score: 0.970
Test 32/449 | Loss: 0.07037 | Accuracy: 0.977 | F1-score: 0.959
Test 33/449 | Loss: 0.02371 | Accuracy: 0.992 | F1-score: 0.964
Test 34/449 | Loss: 0.03982 | Accuracy: 0.984 | F1-score: 0.966
Test 35/449 | Loss: 0.04909 | Accuracy: 0.984 | F1-score: 0.954
Test 36/449 | Loss: 0.08372 | Accuracy: 0.977 | F1-score: 0.983
Test 37/449 | Loss: 0.02566 | Accuracy: 0.992 | F1-score: 0.992
Test 38/449 | Loss: 0.00767 | Accuracy: 1.000 | F1-score: 1.000
Test 39/449 | Loss: 0.03634 | Accuracy: 0.992 | F1-score: 0.962
Test 40/449 | Loss: 0.12969 | Accuracy: 0.984 | F1-score: 0.944
Test 41/449 | Loss: 0.15765 | Accuracy: 0.977 | F1-score: 0.920
Test 42/449 | Loss: 0.02721 | Accuracy: 0.984 | F1-score: 0.986
Test 43/449 | Loss: 0.03920 | Accuracy: 0.984 | F1-score: 0.985
Test 44/449 | Loss: 0.03471 | Accuracy: 0.984 | F1-score: 0.980
Test 45/449 | Loss: 0.03784 | Accuracy: 0.992 | F1-score: 0.963
Test 46/449 | Loss: 0.00208 | Accuracy: 1.000 | F1-score: 1.000
Test 47/449 | Loss: 0.14611 | Accuracy: 0.984 | F1-score: 0.962
Test 48/449 | Loss: 0.04426 | Accuracy: 0.984 | F1-score: 0.951
Test 49/449 | Loss: 0.05442 | Accuracy: 0.984 | F1-score: 0.956
Test 50/449 | Loss: 0.04225 | Accuracy: 0.992 | F1-score: 0.974
Test 51/449 | Loss: 0.12783 | Accuracy: 0.961 | F1-score: 0.903
Test 52/449 | Loss: 0.04291 | Accuracy: 0.984 | F1-score: 0.957
Test 53/449 | Loss: 0.00873 | Accuracy: 1.000 | F1-score: 1.000
Test 54/449 | Loss: 0.07989 | Accuracy: 0.984 | F1-score: 0.937
Test 55/449 | Loss: 0.03088 | Accuracy: 0.992 | F1-score: 0.996
Test 56/449 | Loss: 0.04965 | Accuracy: 0.992 | F1-score: 0.994
Test 57/449 | Loss: 0.04076 | Accuracy: 0.977 | F1-score: 0.931
Test 58/449 | Loss: 0.02350 | Accuracy: 0.984 | F1-score: 0.974
Test 59/449 | Loss: 0.00441 | Accuracy: 1.000 | F1-score: 1.000
Test 60/449 | Loss: 0.03830 | Accuracy: 0.992 | F1-score: 0.976
Test 61/449 | Loss: 0.06731 | Accuracy: 0.984 | F1-score: 0.944
Test 62/449 | Loss: 0.08497 | Accuracy: 0.977 | F1-score: 0.927
Test 63/449 | Loss: 0.04247 | Accuracy: 0.992 | F1-score: 0.975
Test 64/449 | Loss: 0.04331 | Accuracy: 0.992 | F1-score: 0.973
Test 65/449 | Loss: 0.13958 | Accuracy: 0.977 | F1-score: 0.923
Test 66/449 | Loss: 0.01242 | Accuracy: 1.000 | F1-score: 1.000
Test 67/449 | Loss: 0.01492 | Accuracy: 0.992 | F1-score: 0.991
Test 68/449 | Loss: 0.00315 | Accuracy: 1.000 | F1-score: 1.000
Test 69/449 | Loss: 0.07747 | Accuracy: 0.977 | F1-score: 0.985
Test 70/449 | Loss: 0.02134 | Accuracy: 1.000 | F1-score: 1.000
Test 71/449 | Loss: 0.07736 | Accuracy: 0.969 | F1-score: 0.930
Test 72/449 | Loss: 0.01221 | Accuracy: 0.992 | F1-score: 0.973
Test 73/449 | Loss: 0.09168 | Accuracy: 0.969 | F1-score: 0.929
Test 74/449 | Loss: 0.01739 | Accuracy: 0.984 | F1-score: 0.938
Test 75/449 | Loss: 0.02313 | Accuracy: 0.992 | F1-score: 0.974
Test 76/449 | Loss: 0.06518 | Accuracy: 0.969 | F1-score: 0.928
Test 77/449 | Loss: 0.02976 | Accuracy: 0.992 | F1-score: 0.976
Test 78/449 | Loss: 0.01844 | Accuracy: 1.000 | F1-score: 1.000
Test 79/449 | Loss: 0.06726 | Accuracy: 0.984 | F1-score: 0.958
Test 80/449 | Loss: 0.01023 | Accuracy: 1.000 | F1-score: 1.000
Test 81/449 | Loss: 0.05470 | Accuracy: 0.992 | F1-score: 0.990
Test 82/449 | Loss: 0.03871 | Accuracy: 0.984 | F1-score: 0.971
Test 83/449 | Loss: 0.00517 | Accuracy: 1.000 | F1-score: 1.000
Test 84/449 | Loss: 0.02457 | Accuracy: 0.992 | F1-score: 0.989
Test 85/449 | Loss: 0.02887 | Accuracy: 0.992 | F1-score: 0.987
Test 86/449 | Loss: 0.01265 | Accuracy: 0.992 | F1-score: 0.966
Test 87/449 | Loss: 0.00988 | Accuracy: 1.000 | F1-score: 1.000
Test 88/449 | Loss: 0.06678 | Accuracy: 0.969 | F1-score: 0.941
Test 89/449 | Loss: 0.10026 | Accuracy: 0.977 | F1-score: 0.925
Test 90/449 | Loss: 0.01330 | Accuracy: 1.000 | F1-score: 1.000
Test 91/449 | Loss: 0.05058 | Accuracy: 0.992 | F1-score: 0.976
Test 92/449 | Loss: 0.06105 | Accuracy: 0.984 | F1-score: 0.963
Test 93/449 | Loss: 0.00117 | Accuracy: 1.000 | F1-score: 1.000
Test 94/449 | Loss: 0.01100 | Accuracy: 0.992 | F1-score: 0.987
Test 95/449 | Loss: 0.07161 | Accuracy: 0.984 | F1-score: 0.956
Test 96/449 | Loss: 0.04579 | Accuracy: 0.984 | F1-score: 0.972
Test 97/449 | Loss: 0.06241 | Accuracy: 0.992 | F1-score: 0.958
Test 98/449 | Loss: 0.02769 | Accuracy: 1.000 | F1-score: 1.000
Test 99/449 | Loss: 0.19307 | Accuracy: 0.977 | F1-score: 0.935
Test 100/449 | Loss: 0.05037 | Accuracy: 0.984 | F1-score: 0.989
Test 101/449 | Loss: 0.07988 | Accuracy: 0.977 | F1-score: 0.934
Test 102/449 | Loss: 0.00824 | Accuracy: 1.000 | F1-score: 1.000
Test 103/449 | Loss: 0.12381 | Accuracy: 0.977 | F1-score: 0.918
Test 104/449 | Loss: 0.03333 | Accuracy: 0.992 | F1-score: 0.994
Test 105/449 | Loss: 0.01542 | Accuracy: 1.000 | F1-score: 1.000
Test 106/449 | Loss: 0.00832 | Accuracy: 1.000 | F1-score: 1.000
Test 107/449 | Loss: 0.01921 | Accuracy: 1.000 | F1-score: 1.000
Test 108/449 | Loss: 0.12054 | Accuracy: 0.961 | F1-score: 0.892
Test 109/449 | Loss: 0.06639 | Accuracy: 0.984 | F1-score: 0.974
Test 110/449 | Loss: 0.02214 | Accuracy: 0.992 | F1-score: 0.992
Test 111/449 | Loss: 0.03508 | Accuracy: 0.992 | F1-score: 0.992
Test 112/449 | Loss: 0.11156 | Accuracy: 0.977 | F1-score: 0.931
Test 113/449 | Loss: 0.05955 | Accuracy: 0.984 | F1-score: 0.979
Test 114/449 | Loss: 0.06771 | Accuracy: 0.984 | F1-score: 0.952
Test 115/449 | Loss: 0.01054 | Accuracy: 1.000 | F1-score: 1.000
Test 116/449 | Loss: 0.08220 | Accuracy: 0.969 | F1-score: 0.897
Test 117/449 | Loss: 0.04319 | Accuracy: 0.984 | F1-score: 0.929
Test 118/449 | Loss: 0.01530 | Accuracy: 0.992 | F1-score: 0.976
Test 119/449 | Loss: 0.02135 | Accuracy: 0.992 | F1-score: 0.987
Test 120/449 | Loss: 0.06655 | Accuracy: 0.977 | F1-score: 0.915
Test 121/449 | Loss: 0.00136 | Accuracy: 1.000 | F1-score: 1.000
Test 122/449 | Loss: 0.02345 | Accuracy: 0.984 | F1-score: 0.975
Test 123/449 | Loss: 0.18603 | Accuracy: 0.961 | F1-score: 0.892
Test 124/449 | Loss: 0.10138 | Accuracy: 0.961 | F1-score: 0.900
Test 125/449 | Loss: 0.07369 | Accuracy: 0.969 | F1-score: 0.974
Test 126/449 | Loss: 0.01859 | Accuracy: 0.992 | F1-score: 0.992
Test 127/449 | Loss: 0.04355 | Accuracy: 0.992 | F1-score: 0.976
Test 128/449 | Loss: 0.00694 | Accuracy: 1.000 | F1-score: 1.000
Test 129/449 | Loss: 0.01515 | Accuracy: 0.992 | F1-score: 0.974
Test 130/449 | Loss: 0.01117 | Accuracy: 1.000 | F1-score: 1.000
Test 131/449 | Loss: 0.00456 | Accuracy: 1.000 | F1-score: 1.000
Test 132/449 | Loss: 0.02834 | Accuracy: 0.992 | F1-score: 0.977
Test 133/449 | Loss: 0.02706 | Accuracy: 0.992 | F1-score: 0.962
Test 134/449 | Loss: 0.01396 | Accuracy: 0.992 | F1-score: 0.996
Test 135/449 | Loss: 0.08767 | Accuracy: 0.984 | F1-score: 0.981
Test 136/449 | Loss: 0.03227 | Accuracy: 0.977 | F1-score: 0.935
Test 137/449 | Loss: 0.03891 | Accuracy: 0.977 | F1-score: 0.938
Test 138/449 | Loss: 0.01688 | Accuracy: 1.000 | F1-score: 1.000
Test 139/449 | Loss: 0.04916 | Accuracy: 0.984 | F1-score: 0.971
Test 140/449 | Loss: 0.02090 | Accuracy: 0.992 | F1-score: 0.962
Test 141/449 | Loss: 0.01228 | Accuracy: 0.992 | F1-score: 0.992
Test 142/449 | Loss: 0.02071 | Accuracy: 0.992 | F1-score: 0.990
Test 143/449 | Loss: 0.02370 | Accuracy: 0.992 | F1-score: 0.995
Test 144/449 | Loss: 0.06080 | Accuracy: 0.977 | F1-score: 0.944
Test 145/449 | Loss: 0.11212 | Accuracy: 0.977 | F1-score: 0.958
Test 146/449 | Loss: 0.01582 | Accuracy: 0.992 | F1-score: 0.989
Test 147/449 | Loss: 0.04670 | Accuracy: 0.984 | F1-score: 0.984
Test 148/449 | Loss: 0.05098 | Accuracy: 0.984 | F1-score: 0.969
Test 149/449 | Loss: 0.01281 | Accuracy: 1.000 | F1-score: 1.000
Test 150/449 | Loss: 0.01235 | Accuracy: 1.000 | F1-score: 1.000
Test 151/449 | Loss: 0.04928 | Accuracy: 0.992 | F1-score: 0.995
Test 152/449 | Loss: 0.04289 | Accuracy: 0.984 | F1-score: 0.989
Test 153/449 | Loss: 0.03454 | Accuracy: 0.977 | F1-score: 0.954
Test 154/449 | Loss: 0.02093 | Accuracy: 1.000 | F1-score: 1.000
Test 155/449 | Loss: 0.01639 | Accuracy: 0.992 | F1-score: 0.974
Test 156/449 | Loss: 0.02128 | Accuracy: 0.992 | F1-score: 0.987
Test 157/449 | Loss: 0.14450 | Accuracy: 0.953 | F1-score: 0.912
Test 158/449 | Loss: 0.05677 | Accuracy: 0.984 | F1-score: 0.953
Test 159/449 | Loss: 0.05884 | Accuracy: 0.984 | F1-score: 0.940
Test 160/449 | Loss: 0.19250 | Accuracy: 0.953 | F1-score: 0.928
Test 161/449 | Loss: 0.06456 | Accuracy: 0.977 | F1-score: 0.968
Test 162/449 | Loss: 0.02461 | Accuracy: 0.992 | F1-score: 0.977
Test 163/449 | Loss: 0.08250 | Accuracy: 0.977 | F1-score: 0.962
Test 164/449 | Loss: 0.04877 | Accuracy: 0.977 | F1-score: 0.947
Test 165/449 | Loss: 0.03647 | Accuracy: 0.992 | F1-score: 0.959
Test 166/449 | Loss: 0.03491 | Accuracy: 0.977 | F1-score: 0.956
Test 167/449 | Loss: 0.03120 | Accuracy: 0.984 | F1-score: 0.956
Test 168/449 | Loss: 0.04413 | Accuracy: 0.992 | F1-score: 0.994
Test 169/449 | Loss: 0.01744 | Accuracy: 0.992 | F1-score: 0.976
Test 170/449 | Loss: 0.01835 | Accuracy: 0.992 | F1-score: 0.976
Test 171/449 | Loss: 0.02836 | Accuracy: 0.984 | F1-score: 0.954
Test 172/449 | Loss: 0.01083 | Accuracy: 1.000 | F1-score: 1.000
Test 173/449 | Loss: 0.03295 | Accuracy: 0.984 | F1-score: 0.976
Test 174/449 | Loss: 0.09287 | Accuracy: 0.984 | F1-score: 0.972
Test 175/449 | Loss: 0.06510 | Accuracy: 0.977 | F1-score: 0.971
Test 176/449 | Loss: 0.02196 | Accuracy: 0.992 | F1-score: 0.977
Test 177/449 | Loss: 0.05161 | Accuracy: 0.984 | F1-score: 0.922
Test 178/449 | Loss: 0.06788 | Accuracy: 0.984 | F1-score: 0.942
Test 179/449 | Loss: 0.01357 | Accuracy: 0.992 | F1-score: 0.989
Test 180/449 | Loss: 0.01599 | Accuracy: 0.992 | F1-score: 0.964
Test 181/449 | Loss: 0.00706 | Accuracy: 1.000 | F1-score: 1.000
Test 182/449 | Loss: 0.00543 | Accuracy: 1.000 | F1-score: 1.000
Test 183/449 | Loss: 0.01022 | Accuracy: 1.000 | F1-score: 1.000
Test 184/449 | Loss: 0.02947 | Accuracy: 0.992 | F1-score: 0.963
Test 185/449 | Loss: 0.01229 | Accuracy: 1.000 | F1-score: 1.000
Test 186/449 | Loss: 0.02036 | Accuracy: 0.992 | F1-score: 0.962
Test 187/449 | Loss: 0.03798 | Accuracy: 0.984 | F1-score: 0.975
Test 188/449 | Loss: 0.01215 | Accuracy: 1.000 | F1-score: 1.000
Test 189/449 | Loss: 0.04997 | Accuracy: 0.977 | F1-score: 0.946
Test 190/449 | Loss: 0.00732 | Accuracy: 1.000 | F1-score: 1.000
Test 191/449 | Loss: 0.01520 | Accuracy: 1.000 | F1-score: 1.000
Test 192/449 | Loss: 0.01823 | Accuracy: 0.992 | F1-score: 0.990
Test 193/449 | Loss: 0.12048 | Accuracy: 0.969 | F1-score: 0.933
Test 194/449 | Loss: 0.06061 | Accuracy: 0.984 | F1-score: 0.984
Test 195/449 | Loss: 0.06037 | Accuracy: 0.984 | F1-score: 0.969
Test 196/449 | Loss: 0.01751 | Accuracy: 1.000 | F1-score: 1.000
Test 197/449 | Loss: 0.04386 | Accuracy: 0.992 | F1-score: 0.973
Test 198/449 | Loss: 0.06961 | Accuracy: 0.992 | F1-score: 0.979
Test 199/449 | Loss: 0.07668 | Accuracy: 0.977 | F1-score: 0.942
Test 200/449 | Loss: 0.05380 | Accuracy: 0.977 | F1-score: 0.979
Test 201/449 | Loss: 0.03505 | Accuracy: 0.977 | F1-score: 0.948
Test 202/449 | Loss: 0.00676 | Accuracy: 1.000 | F1-score: 1.000
Test 203/449 | Loss: 0.02920 | Accuracy: 0.992 | F1-score: 0.992
Test 204/449 | Loss: 0.04066 | Accuracy: 0.984 | F1-score: 0.987
Test 205/449 | Loss: 0.01139 | Accuracy: 1.000 | F1-score: 1.000
Test 206/449 | Loss: 0.01551 | Accuracy: 1.000 | F1-score: 1.000
Test 207/449 | Loss: 0.04053 | Accuracy: 0.992 | F1-score: 0.996
Test 208/449 | Loss: 0.00451 | Accuracy: 1.000 | F1-score: 1.000
Test 209/449 | Loss: 0.00759 | Accuracy: 1.000 | F1-score: 1.000
Test 210/449 | Loss: 0.02671 | Accuracy: 0.992 | F1-score: 0.994
Test 211/449 | Loss: 0.00509 | Accuracy: 1.000 | F1-score: 1.000
Test 212/449 | Loss: 0.02871 | Accuracy: 0.984 | F1-score: 0.964
Test 213/449 | Loss: 0.01437 | Accuracy: 1.000 | F1-score: 1.000
Test 214/449 | Loss: 0.01614 | Accuracy: 0.992 | F1-score: 0.980
Test 215/449 | Loss: 0.04942 | Accuracy: 0.984 | F1-score: 0.931
Test 216/449 | Loss: 0.03356 | Accuracy: 0.992 | F1-score: 0.962
Test 217/449 | Loss: 0.03063 | Accuracy: 0.992 | F1-score: 0.979
Test 218/449 | Loss: 0.04252 | Accuracy: 0.992 | F1-score: 0.977
Test 219/449 | Loss: 0.00834 | Accuracy: 0.992 | F1-score: 0.980
Test 220/449 | Loss: 0.03266 | Accuracy: 0.992 | F1-score: 0.963
Test 221/449 | Loss: 0.05818 | Accuracy: 0.984 | F1-score: 0.967
Test 222/449 | Loss: 0.01353 | Accuracy: 0.992 | F1-score: 0.994
Test 223/449 | Loss: 0.02536 | Accuracy: 1.000 | F1-score: 1.000
Test 224/449 | Loss: 0.01467 | Accuracy: 1.000 | F1-score: 1.000
Test 225/449 | Loss: 0.09908 | Accuracy: 0.977 | F1-score: 0.936
Test 226/449 | Loss: 0.03130 | Accuracy: 0.992 | F1-score: 0.989
Test 227/449 | Loss: 0.00410 | Accuracy: 1.000 | F1-score: 1.000
Test 228/449 | Loss: 0.06706 | Accuracy: 0.961 | F1-score: 0.918
Test 229/449 | Loss: 0.05317 | Accuracy: 0.992 | F1-score: 0.994
Test 230/449 | Loss: 0.12638 | Accuracy: 0.969 | F1-score: 0.909
Test 231/449 | Loss: 0.01416 | Accuracy: 0.992 | F1-score: 0.979
Test 232/449 | Loss: 0.06337 | Accuracy: 0.977 | F1-score: 0.945
Test 233/449 | Loss: 0.04855 | Accuracy: 0.984 | F1-score: 0.956
Test 234/449 | Loss: 0.02802 | Accuracy: 0.984 | F1-score: 0.987
Test 235/449 | Loss: 0.01956 | Accuracy: 0.992 | F1-score: 0.981
Test 236/449 | Loss: 0.02172 | Accuracy: 0.984 | F1-score: 0.978
Test 237/449 | Loss: 0.08003 | Accuracy: 0.969 | F1-score: 0.926
Test 238/449 | Loss: 0.00956 | Accuracy: 0.992 | F1-score: 0.973
Test 239/449 | Loss: 0.05063 | Accuracy: 0.992 | F1-score: 0.978
Test 240/449 | Loss: 0.02781 | Accuracy: 0.992 | F1-score: 0.965
Test 241/449 | Loss: 0.01667 | Accuracy: 0.992 | F1-score: 0.990
Test 242/449 | Loss: 0.01597 | Accuracy: 0.992 | F1-score: 0.992
Test 243/449 | Loss: 0.02291 | Accuracy: 0.992 | F1-score: 0.975
Test 244/449 | Loss: 0.05675 | Accuracy: 0.984 | F1-score: 0.990
Test 245/449 | Loss: 0.02072 | Accuracy: 1.000 | F1-score: 1.000
Test 246/449 | Loss: 0.09887 | Accuracy: 0.969 | F1-score: 0.924
Test 247/449 | Loss: 0.10752 | Accuracy: 0.969 | F1-score: 0.956
Test 248/449 | Loss: 0.03137 | Accuracy: 0.992 | F1-score: 0.979
Test 249/449 | Loss: 0.00831 | Accuracy: 1.000 | F1-score: 1.000
Test 250/449 | Loss: 0.10426 | Accuracy: 0.984 | F1-score: 0.957
Test 251/449 | Loss: 0.03192 | Accuracy: 0.992 | F1-score: 0.996
Test 252/449 | Loss: 0.04151 | Accuracy: 0.984 | F1-score: 0.970
Test 253/449 | Loss: 0.03275 | Accuracy: 0.992 | F1-score: 0.977
Test 254/449 | Loss: 0.02968 | Accuracy: 0.992 | F1-score: 0.991
Test 255/449 | Loss: 0.03650 | Accuracy: 0.992 | F1-score: 0.996
Test 256/449 | Loss: 0.02911 | Accuracy: 0.984 | F1-score: 0.988
Test 257/449 | Loss: 0.05941 | Accuracy: 0.977 | F1-score: 0.934
Test 258/449 | Loss: 0.09366 | Accuracy: 0.961 | F1-score: 0.921
Test 259/449 | Loss: 0.03044 | Accuracy: 0.992 | F1-score: 0.977
Test 260/449 | Loss: 0.01035 | Accuracy: 0.992 | F1-score: 0.980
Test 261/449 | Loss: 0.02374 | Accuracy: 0.992 | F1-score: 0.994
Test 262/449 | Loss: 0.03778 | Accuracy: 0.992 | F1-score: 0.995
Test 263/449 | Loss: 0.09729 | Accuracy: 0.977 | F1-score: 0.973
Test 264/449 | Loss: 0.08091 | Accuracy: 0.977 | F1-score: 0.915
Test 265/449 | Loss: 0.08562 | Accuracy: 0.953 | F1-score: 0.902
Test 266/449 | Loss: 0.04894 | Accuracy: 0.977 | F1-score: 0.912
Test 267/449 | Loss: 0.01867 | Accuracy: 1.000 | F1-score: 1.000
Test 268/449 | Loss: 0.07691 | Accuracy: 0.992 | F1-score: 0.995
Test 269/449 | Loss: 0.08830 | Accuracy: 0.977 | F1-score: 0.941
Test 270/449 | Loss: 0.07201 | Accuracy: 0.984 | F1-score: 0.987
Test 271/449 | Loss: 0.00717 | Accuracy: 1.000 | F1-score: 1.000
Test 272/449 | Loss: 0.05049 | Accuracy: 0.984 | F1-score: 0.968
Test 273/449 | Loss: 0.03186 | Accuracy: 0.984 | F1-score: 0.958
Test 274/449 | Loss: 0.01699 | Accuracy: 0.992 | F1-score: 0.964
Test 275/449 | Loss: 0.01275 | Accuracy: 1.000 | F1-score: 1.000
Test 276/449 | Loss: 0.02207 | Accuracy: 1.000 | F1-score: 1.000
Test 277/449 | Loss: 0.08454 | Accuracy: 0.977 | F1-score: 0.966
Test 278/449 | Loss: 0.00894 | Accuracy: 1.000 | F1-score: 1.000
Test 279/449 | Loss: 0.01161 | Accuracy: 1.000 | F1-score: 1.000
Test 280/449 | Loss: 0.02235 | Accuracy: 0.992 | F1-score: 0.990
Test 281/449 | Loss: 0.03644 | Accuracy: 0.992 | F1-score: 0.981
Test 282/449 | Loss: 0.18898 | Accuracy: 0.992 | F1-score: 0.995
Test 283/449 | Loss: 0.03600 | Accuracy: 0.992 | F1-score: 0.964
Test 284/449 | Loss: 0.08526 | Accuracy: 0.977 | F1-score: 0.971
Test 285/449 | Loss: 0.00921 | Accuracy: 1.000 | F1-score: 1.000
Test 286/449 | Loss: 0.05476 | Accuracy: 0.984 | F1-score: 0.971
Test 287/449 | Loss: 0.00511 | Accuracy: 1.000 | F1-score: 1.000
Test 288/449 | Loss: 0.02289 | Accuracy: 0.992 | F1-score: 0.962
Test 289/449 | Loss: 0.04529 | Accuracy: 0.977 | F1-score: 0.949
Test 290/449 | Loss: 0.02212 | Accuracy: 0.984 | F1-score: 0.954
Test 291/449 | Loss: 0.01895 | Accuracy: 0.992 | F1-score: 0.987
Test 292/449 | Loss: 0.00514 | Accuracy: 1.000 | F1-score: 1.000
Test 293/449 | Loss: 0.03202 | Accuracy: 0.992 | F1-score: 0.993
Test 294/449 | Loss: 0.01549 | Accuracy: 0.992 | F1-score: 0.977
Test 295/449 | Loss: 0.03453 | Accuracy: 0.984 | F1-score: 0.969
Test 296/449 | Loss: 0.15601 | Accuracy: 0.977 | F1-score: 0.920
Test 297/449 | Loss: 0.04482 | Accuracy: 0.992 | F1-score: 0.993
Test 298/449 | Loss: 0.06604 | Accuracy: 0.992 | F1-score: 0.979
Test 299/449 | Loss: 0.07665 | Accuracy: 0.969 | F1-score: 0.929
Test 300/449 | Loss: 0.05973 | Accuracy: 0.992 | F1-score: 0.964
Test 301/449 | Loss: 0.02090 | Accuracy: 0.984 | F1-score: 0.948
Test 302/449 | Loss: 0.00895 | Accuracy: 1.000 | F1-score: 1.000
Test 303/449 | Loss: 0.00544 | Accuracy: 1.000 | F1-score: 1.000
Test 304/449 | Loss: 0.05009 | Accuracy: 0.984 | F1-score: 0.953
Test 305/449 | Loss: 0.00882 | Accuracy: 1.000 | F1-score: 1.000
Test 306/449 | Loss: 0.06006 | Accuracy: 0.984 | F1-score: 0.978
Test 307/449 | Loss: 0.03565 | Accuracy: 0.992 | F1-score: 0.975
Test 308/449 | Loss: 0.02156 | Accuracy: 1.000 | F1-score: 1.000
Test 309/449 | Loss: 0.05107 | Accuracy: 0.992 | F1-score: 0.995
Test 310/449 | Loss: 0.05888 | Accuracy: 0.984 | F1-score: 0.988
Test 311/449 | Loss: 0.00741 | Accuracy: 1.000 | F1-score: 1.000
Test 312/449 | Loss: 0.00472 | Accuracy: 1.000 | F1-score: 1.000
Test 313/449 | Loss: 0.00596 | Accuracy: 1.000 | F1-score: 1.000
Test 314/449 | Loss: 0.03497 | Accuracy: 0.992 | F1-score: 0.962
Test 315/449 | Loss: 0.02380 | Accuracy: 0.992 | F1-score: 0.976
Test 316/449 | Loss: 0.12880 | Accuracy: 0.961 | F1-score: 0.940
Test 317/449 | Loss: 0.14661 | Accuracy: 0.969 | F1-score: 0.946
Test 318/449 | Loss: 0.01131 | Accuracy: 1.000 | F1-score: 1.000
Test 319/449 | Loss: 0.04268 | Accuracy: 0.977 | F1-score: 0.955
Test 320/449 | Loss: 0.02472 | Accuracy: 0.992 | F1-score: 0.976
Test 321/449 | Loss: 0.06139 | Accuracy: 0.984 | F1-score: 0.992
Test 322/449 | Loss: 0.01430 | Accuracy: 0.992 | F1-score: 0.975
Test 323/449 | Loss: 0.02446 | Accuracy: 0.992 | F1-score: 0.995
Test 324/449 | Loss: 0.01845 | Accuracy: 0.992 | F1-score: 0.994
Test 325/449 | Loss: 0.01326 | Accuracy: 1.000 | F1-score: 1.000
Test 326/449 | Loss: 0.00865 | Accuracy: 1.000 | F1-score: 1.000
Test 327/449 | Loss: 0.09050 | Accuracy: 0.977 | F1-score: 0.938
Test 328/449 | Loss: 0.06992 | Accuracy: 0.977 | F1-score: 0.924
Test 329/449 | Loss: 0.05763 | Accuracy: 0.984 | F1-score: 0.955
Test 330/449 | Loss: 0.02860 | Accuracy: 0.992 | F1-score: 0.990
Test 331/449 | Loss: 0.06582 | Accuracy: 0.977 | F1-score: 0.950
Test 332/449 | Loss: 0.01282 | Accuracy: 1.000 | F1-score: 1.000
Test 333/449 | Loss: 0.11414 | Accuracy: 0.977 | F1-score: 0.965
Test 334/449 | Loss: 0.05580 | Accuracy: 0.984 | F1-score: 0.937
Test 335/449 | Loss: 0.02097 | Accuracy: 0.984 | F1-score: 0.987
Test 336/449 | Loss: 0.05589 | Accuracy: 0.977 | F1-score: 0.951
Test 337/449 | Loss: 0.01770 | Accuracy: 0.992 | F1-score: 0.974
Test 338/449 | Loss: 0.02682 | Accuracy: 0.992 | F1-score: 0.962
Test 339/449 | Loss: 0.03303 | Accuracy: 0.984 | F1-score: 0.939
Test 340/449 | Loss: 0.04852 | Accuracy: 0.977 | F1-score: 0.956
Test 341/449 | Loss: 0.06514 | Accuracy: 0.984 | F1-score: 0.987
Test 342/449 | Loss: 0.04492 | Accuracy: 0.984 | F1-score: 0.976
Test 343/449 | Loss: 0.10774 | Accuracy: 0.984 | F1-score: 0.971
Test 344/449 | Loss: 0.04316 | Accuracy: 0.984 | F1-score: 0.972
Test 345/449 | Loss: 0.02749 | Accuracy: 0.984 | F1-score: 0.938
Test 346/449 | Loss: 0.07649 | Accuracy: 0.977 | F1-score: 0.967
Test 347/449 | Loss: 0.01525 | Accuracy: 0.992 | F1-score: 0.978
Test 348/449 | Loss: 0.04065 | Accuracy: 0.984 | F1-score: 0.973
Test 349/449 | Loss: 0.09100 | Accuracy: 0.977 | F1-score: 0.922
Test 350/449 | Loss: 0.04116 | Accuracy: 0.984 | F1-score: 0.990
Test 351/449 | Loss: 0.01204 | Accuracy: 1.000 | F1-score: 1.000
Test 352/449 | Loss: 0.00249 | Accuracy: 1.000 | F1-score: 1.000
Test 353/449 | Loss: 0.02875 | Accuracy: 0.984 | F1-score: 0.967
Test 354/449 | Loss: 0.03693 | Accuracy: 0.984 | F1-score: 0.955
Test 355/449 | Loss: 0.02803 | Accuracy: 0.992 | F1-score: 0.976
Test 356/449 | Loss: 0.02545 | Accuracy: 0.984 | F1-score: 0.952
Test 357/449 | Loss: 0.01538 | Accuracy: 0.992 | F1-score: 0.960
Test 358/449 | Loss: 0.02325 | Accuracy: 0.992 | F1-score: 0.977
Test 359/449 | Loss: 0.00297 | Accuracy: 1.000 | F1-score: 1.000
Test 360/449 | Loss: 0.07971 | Accuracy: 0.969 | F1-score: 0.918
Test 361/449 | Loss: 0.07131 | Accuracy: 0.977 | F1-score: 0.947
Test 362/449 | Loss: 0.02461 | Accuracy: 0.992 | F1-score: 0.989
Test 363/449 | Loss: 0.06243 | Accuracy: 0.984 | F1-score: 0.973
Test 364/449 | Loss: 0.01114 | Accuracy: 0.992 | F1-score: 0.995
Test 365/449 | Loss: 0.01270 | Accuracy: 0.992 | F1-score: 0.979
Test 366/449 | Loss: 0.00807 | Accuracy: 1.000 | F1-score: 1.000
Test 367/449 | Loss: 0.08883 | Accuracy: 0.969 | F1-score: 0.952
Test 368/449 | Loss: 0.02320 | Accuracy: 1.000 | F1-score: 1.000
Test 369/449 | Loss: 0.05854 | Accuracy: 0.984 | F1-score: 0.957
Test 370/449 | Loss: 0.15898 | Accuracy: 0.969 | F1-score: 0.933
Test 371/449 | Loss: 0.01736 | Accuracy: 0.992 | F1-score: 0.996
Test 372/449 | Loss: 0.00948 | Accuracy: 1.000 | F1-score: 1.000
Test 373/449 | Loss: 0.02943 | Accuracy: 0.992 | F1-score: 0.991
Test 374/449 | Loss: 0.01575 | Accuracy: 1.000 | F1-score: 1.000
Test 375/449 | Loss: 0.00828 | Accuracy: 1.000 | F1-score: 1.000
Test 376/449 | Loss: 0.07614 | Accuracy: 0.984 | F1-score: 0.950
Test 377/449 | Loss: 0.02504 | Accuracy: 0.984 | F1-score: 0.958
Test 378/449 | Loss: 0.07377 | Accuracy: 0.984 | F1-score: 0.949
Test 379/449 | Loss: 0.04179 | Accuracy: 0.984 | F1-score: 0.963
Test 380/449 | Loss: 0.03553 | Accuracy: 0.992 | F1-score: 0.977
Test 381/449 | Loss: 0.07135 | Accuracy: 0.984 | F1-score: 0.942
Test 382/449 | Loss: 0.03205 | Accuracy: 0.992 | F1-score: 0.967
Test 383/449 | Loss: 0.08571 | Accuracy: 0.977 | F1-score: 0.967
Test 384/449 | Loss: 0.02985 | Accuracy: 0.992 | F1-score: 0.993
Test 385/449 | Loss: 0.04549 | Accuracy: 0.992 | F1-score: 0.980
Test 386/449 | Loss: 0.05608 | Accuracy: 0.984 | F1-score: 0.953
Test 387/449 | Loss: 0.03292 | Accuracy: 0.992 | F1-score: 0.990
Test 388/449 | Loss: 0.04990 | Accuracy: 0.992 | F1-score: 0.973
Test 389/449 | Loss: 0.04509 | Accuracy: 0.984 | F1-score: 0.988
Test 390/449 | Loss: 0.07582 | Accuracy: 0.984 | F1-score: 0.988
Test 391/449 | Loss: 0.02797 | Accuracy: 0.992 | F1-score: 0.991
Test 392/449 | Loss: 0.02147 | Accuracy: 0.992 | F1-score: 0.957
Test 393/449 | Loss: 0.00657 | Accuracy: 1.000 | F1-score: 1.000
Test 394/449 | Loss: 0.06848 | Accuracy: 0.992 | F1-score: 0.976
Test 395/449 | Loss: 0.07822 | Accuracy: 0.969 | F1-score: 0.945
Test 396/449 | Loss: 0.02827 | Accuracy: 0.992 | F1-score: 0.995
Test 397/449 | Loss: 0.02228 | Accuracy: 0.992 | F1-score: 0.975
Test 398/449 | Loss: 0.05033 | Accuracy: 0.992 | F1-score: 0.963
Test 399/449 | Loss: 0.05693 | Accuracy: 0.969 | F1-score: 0.907
Test 400/449 | Loss: 0.04349 | Accuracy: 0.984 | F1-score: 0.971
Test 401/449 | Loss: 0.05444 | Accuracy: 0.992 | F1-score: 0.992
Test 402/449 | Loss: 0.05870 | Accuracy: 0.969 | F1-score: 0.938
Test 403/449 | Loss: 0.04251 | Accuracy: 0.984 | F1-score: 0.967
Test 404/449 | Loss: 0.02809 | Accuracy: 0.992 | F1-score: 0.993
Test 405/449 | Loss: 0.06600 | Accuracy: 0.984 | F1-score: 0.971
Test 406/449 | Loss: 0.04432 | Accuracy: 0.977 | F1-score: 0.970
Test 407/449 | Loss: 0.04278 | Accuracy: 0.992 | F1-score: 0.976
Test 408/449 | Loss: 0.03406 | Accuracy: 0.984 | F1-score: 0.969
Test 409/449 | Loss: 0.04345 | Accuracy: 0.992 | F1-score: 0.979
Test 410/449 | Loss: 0.11625 | Accuracy: 0.977 | F1-score: 0.956
Test 411/449 | Loss: 0.05138 | Accuracy: 0.992 | F1-score: 0.995
Test 412/449 | Loss: 0.01938 | Accuracy: 0.992 | F1-score: 0.994
Test 413/449 | Loss: 0.04502 | Accuracy: 0.984 | F1-score: 0.933
Test 414/449 | Loss: 0.02035 | Accuracy: 0.992 | F1-score: 0.991
Test 415/449 | Loss: 0.01217 | Accuracy: 1.000 | F1-score: 1.000
Test 416/449 | Loss: 0.03097 | Accuracy: 0.992 | F1-score: 0.990
Test 417/449 | Loss: 0.01382 | Accuracy: 1.000 | F1-score: 1.000
Test 418/449 | Loss: 0.04062 | Accuracy: 0.992 | F1-score: 0.981
Test 419/449 | Loss: 0.07540 | Accuracy: 0.969 | F1-score: 0.914
Test 420/449 | Loss: 0.04837 | Accuracy: 0.984 | F1-score: 0.956
Test 421/449 | Loss: 0.06119 | Accuracy: 0.969 | F1-score: 0.930
Test 422/449 | Loss: 0.00299 | Accuracy: 1.000 | F1-score: 1.000
Test 423/449 | Loss: 0.02573 | Accuracy: 0.992 | F1-score: 0.962
Test 424/449 | Loss: 0.01169 | Accuracy: 1.000 | F1-score: 1.000
Test 425/449 | Loss: 0.04634 | Accuracy: 0.984 | F1-score: 0.968
Test 426/449 | Loss: 0.01973 | Accuracy: 0.992 | F1-score: 0.980
Test 427/449 | Loss: 0.02054 | Accuracy: 0.992 | F1-score: 0.974
Test 428/449 | Loss: 0.06712 | Accuracy: 0.977 | F1-score: 0.952
Test 429/449 | Loss: 0.05746 | Accuracy: 0.977 | F1-score: 0.943
Test 430/449 | Loss: 0.02828 | Accuracy: 0.992 | F1-score: 0.977
Test 431/449 | Loss: 0.22647 | Accuracy: 0.969 | F1-score: 0.943
Test 432/449 | Loss: 0.03737 | Accuracy: 0.992 | F1-score: 0.982
Test 433/449 | Loss: 0.02888 | Accuracy: 0.992 | F1-score: 0.961
Test 434/449 | Loss: 0.01397 | Accuracy: 0.992 | F1-score: 0.991
Test 435/449 | Loss: 0.02677 | Accuracy: 0.984 | F1-score: 0.959
Test 436/449 | Loss: 0.03439 | Accuracy: 0.992 | F1-score: 0.979
Test 437/449 | Loss: 0.05907 | Accuracy: 0.984 | F1-score: 0.947
Test 438/449 | Loss: 0.02709 | Accuracy: 0.992 | F1-score: 0.990
Test 439/449 | Loss: 0.00617 | Accuracy: 1.000 | F1-score: 1.000
Test 440/449 | Loss: 0.02074 | Accuracy: 1.000 | F1-score: 1.000
Test 441/449 | Loss: 0.06488 | Accuracy: 0.992 | F1-score: 0.991
Test 442/449 | Loss: 0.00528 | Accuracy: 1.000 | F1-score: 1.000
Test 443/449 | Loss: 0.02039 | Accuracy: 0.992 | F1-score: 0.996
Test 444/449 | Loss: 0.02399 | Accuracy: 0.992 | F1-score: 0.996
Test 445/449 | Loss: 0.02547 | Accuracy: 0.992 | F1-score: 0.993
Test 446/449 | Loss: 0.01940 | Accuracy: 0.992 | F1-score: 0.990
Test 447/449 | Loss: 0.06967 | Accuracy: 0.984 | F1-score: 0.981
Test 448/449 | Loss: 0.08867 | Accuracy: 0.984 | F1-score: 0.970
Test | Loss: 0.04279 | Accuracy: 0.988 | F1-score: 0.973
Finished
```
## exp12
- Train on categories having more than 2800 images
# Notes
- Solutions for data imbalance
  - Sampling methods
    - Oversampling for the categories have a lot of images 
    - Undersampling for the categories have less images
  - Loss
    - Focal loss
  - Data augmentation
  - **Transfer learning**
  - Use traditional machine learning methods
- A discussion about TP=FP=FN=0:
  - https://github.com/scikit-learn/scikit-learn/discussions/28675