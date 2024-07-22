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