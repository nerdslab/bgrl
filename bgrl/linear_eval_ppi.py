import numpy as np
import torch
from sklearn import metrics


def ppi_train_linear_layer(num_classes, train_data, val_data, test_data, device):
    r"""
    Trains a linear layer on top of the representations. This function is specific to the PPI dataset,
    which has multiple labels.
    """
    def train(classifier, train_data, optimizer):
        classifier.train()

        x, label = train_data
        x, label = x.to(device), label.to(device)
        for step in range(100):
            # forward
            optimizer.zero_grad()
            pred_logits = classifier(x)

            # loss and backprop
            loss = criterion(pred_logits, label)
            loss.backward()
            optimizer.step()

    def test(classifier, data):
        classifier.eval()
        x, label = data
        label = label.cpu().numpy().squeeze()

        # feed to network and classifier
        with torch.no_grad():
            pred_logits = classifier(x.to(device))
            pred_class = (pred_logits > 0).float().cpu().numpy()

        return metrics.f1_score(label, pred_class, average='micro') if pred_class.sum() > 0 else 0

    num_feats = train_data[0].size(1)
    criterion = torch.nn.BCEWithLogitsLoss()

    # normalization
    mean, std = train_data[0].mean(0, keepdim=True), train_data[0].std(0, unbiased=False, keepdim=True)
    train_data[0] = (train_data[0] - mean) / std
    val_data[0] = (val_data[0] - mean) / std
    test_data[0] = (test_data[0] - mean) / std

    best_val_f1 = 0
    test_f1 = 0
    for weight_decay in 2.0 ** np.arange(-10, 11, 2):
        classifier = torch.nn.Linear(num_feats, num_classes).to(device)
        optimizer = torch.optim.AdamW(params=classifier.parameters(), lr=0.01, weight_decay=weight_decay)

        train(classifier, train_data, optimizer)
        val_f1 = test(classifier, val_data)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            test_f1 = test(classifier, test_data)

    return best_val_f1, test_f1
