# Train
## Train on Full Dataset
- `python .\train.py -nc 122 -es 20 -bs 128 -trp path_to_training_dataset`
## Train on Selected Dataset
- Obtain partial dataset: `python utils.py`
- `python .\train.py -nc number_of_category/directory -es 20 -bs 128 -trp path_to_partial_training_dataset`

# Predict
- `python predict.py`

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
## exp7
- random seed: 42
- Train on selected dataset without normalization


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