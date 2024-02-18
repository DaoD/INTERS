import numpy as np
from sklearn.metrics import f1_score, accuracy_score

class FactVerification():
    def __init__(self, name, shot, setting):
        self._cluster = "fact_verification"
        self._name = name
        self._shot = shot
        self._setting = setting

    def get_path(self):
        return f"data/{self._setting}/{self._shot}/{self._cluster}_{self._name}.{self._shot}.test.jsonl"

    def compute_metrics(self, preds, labels):
        labels = np.asarray([x[0] for x in labels])
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='weighted')

        results = {
            "Acc": acc,
            "F1": f1
        }

        return results

    