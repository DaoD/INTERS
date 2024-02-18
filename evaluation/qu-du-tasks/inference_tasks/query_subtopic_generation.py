import numpy as np
import re

class QuerySubtopicGeneration():
    def __init__(self, name, shot, setting):
        self._cluster = "query_subtopic_generation"
        self._name = name
        self._shot = shot
        self._setting = setting

    def get_path(self):
        return f"data/{self._setting}/{self._shot}/{self._cluster}_{self._name}.{self._shot}.test.jsonl"

    def compute_metrics(self, preds, labels):
        def normalize_results(text):
            pattern = r"(?:\d\.|\[\d\]|\(\d\))\s*(.*?)\s*(?=\d\.|\[\d\]|\(\d\)|$)"
            extracted_text = re.findall(pattern, text)
            return extracted_text
        
        all_p, all_r, all_f1 = [], [], []
        for prediction, label in zip(preds, labels):
            label = normalize_results(label)
            extracted_prediction = normalize_results(prediction)
            if extracted_prediction == []:
                extracted_prediction = prediction
            TP = len(set(extracted_prediction).intersection(label))
            FP = len(set(extracted_prediction) - set(label))
            FN = len(set(label) - set(extracted_prediction))
            precision = TP / (TP + FP) if (TP + FP) != 0 else 0
            recall = TP / (TP + FN) if (TP + FN) != 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
            all_p.append(precision)
            all_r.append(recall)
            all_f1.append(f1)
        all_p = np.asarray(all_p)
        all_r = np.asarray(all_r)
        all_f1 = np.asarray(all_f1)
        results = {
            "precision": np.sum(all_p) / len(all_p),
            "recall": np.sum(all_r) / len(all_r),
            "f1": np.sum(all_f1) / len(all_f1),
        }
        return results