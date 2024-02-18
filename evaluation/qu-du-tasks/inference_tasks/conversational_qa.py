import numpy as np
import re

class ConversationalQA:
    def __init__(self, name, shot, setting):
        self._cluster = "conversational_qa"
        self._name = name
        self._shot = shot
        self._setting = setting

    def get_path(self):
        return f"data/{self._setting}/{self._shot}/{self._cluster}_{self._name}.{self._shot}.test.jsonl"

    def compute_metrics(self, preds, labels):
        def normalize_results(text):
            pattern = r"(\d+\.|\[\d+\]|\(\d+\))\s*(.*?)\s*(?=\d\.|\[\d\]|\(\d\)|$)"
            extracted_text = re.findall(pattern, text)
            return extracted_text

        all_acc = []
        for prediction, label in zip(preds, labels):
            extracted_answer = normalize_results(label)
            extracted_prediction = normalize_results(prediction)
            if extracted_answer == []:
                extracted_answer = label
                answer_dict = {0: extracted_answer} 
            else:
                answer_dict = {match[0]: match[1] for match in extracted_answer}
            if extracted_prediction == []:
                extracted_prediction = prediction
                extracted_dict = {0: extracted_prediction}
            else:
                extracted_dict = {match[0]: match[1] for match in extracted_prediction}
            
            acc = 0
            for k in extracted_dict.keys():
                if k not in answer_dict:
                    continue
                if extracted_dict[k] == answer_dict[k]:
                    acc += 1
            acc = acc / len(answer_dict.keys())
            all_acc.append(acc)
        all_acc = np.asarray(all_acc)
        results = {
            "Acc": np.mean(all_acc)
        }
        return results