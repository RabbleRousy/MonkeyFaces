# モデルを学習後に実行してください
import torch
from tqdm import tqdm

# モデルの評価
def evaluate_model(model, test_dataloader, class_names):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # テストデータに含まれるクラスのみで評価
    test_classes = sorted(set(all_labels))
    y_true = [label for label in all_labels if label in test_classes]
    y_pred = [pred for pred, label in zip(all_preds, all_labels) if label in test_classes]

    macro_f1 = f1_score(y_true, y_pred, average='macro')
    print(f'Macro F1 Score: {macro_f1:.4f}')