from sklearn.metrics import f1_score, accuracy_score

class QueryIntentClassification():
    def __init__(self, name, shot, setting, multi_label=False):
        self._cluster = "query_intent_classification"
        self._name = name
        self._shot = shot
        self._setting = setting
        self._multi_label = multi_label

    def get_path(self):
        return f"data/{self._setting}/{self._shot}/{self._cluster}_{self._name}.{self._shot}.test.jsonl"

    def compute_metrics(self, preds, labels):
        # accuracy = np.sum(np.asarray(preds) == np.asarray(labels)) / len(preds)
        if self._multi_label:
            p = 0
            for pred, label in zip(preds, labels):
                if pred in label:
                    p += 1
            results = {
                "P@1": p / len(preds)
            }
        else:
            acc = accuracy_score(labels, preds)
            f1 = f1_score(labels, preds, average='weighted')
            results = {
                "Acc": acc,
                "F1": f1
            }

        return results
