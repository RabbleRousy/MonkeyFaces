# Train
## Train on Full Dataset
- `python .\train.py -nc 122 -es 20 -bs 128 -trp path_to_training_dataset`
## Train on Selected Dataset
- Obtain partial dataset: `python utils.py`
- `python .\train.py -nc 28 -es 20 -bs 128 -trp path_to_partial_training_dataset`

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
| | |
| --- | --- |
| Loss |  |
| Accuracy |  |
| F1-score |  |

## exp4
- Implement validation and testing process