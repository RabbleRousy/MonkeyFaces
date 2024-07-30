# Train
## Train on Full Dataset
- `python .\train.py -nc 122 -es 20 -bs 128 -trp path_to_training_dataset --train --val --test`
## Train on Selected Dataset
- Obtain partial dataset: `python utils.py`
- `python .\train.py -nc number_of_category/directory -es 20 -bs 128 -trp path_to_partial_training_dataset --train`

# Predict
- Method 1: `python predict.py`
- Method 2: `python .\train.py -nc 122 -es 20 -bs 128 -tep path_to_test_dataset --test`

# Load with pre-trained model
- `python train.py -nc 122 -bs 128 -tep E:\datasets\monkeys\facedata_yamada\facedata_yamada\test_Magface --checkpoint E:\ws\MonkeyFace\Prototypes\Zhang\logs\exp5\weights\12_0.9711587778429074.pth.tar --train --test `

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