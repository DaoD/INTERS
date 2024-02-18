from rouge_score import rouge_scorer
import numpy as np

class Summarization():
    def __init__(self, name, shot, setting):
        self._cluster = "summarization"
        self._name = name
        self._shot = shot
        self._setting = setting
        
    
    def get_path(self):
        return f"data/{self._setting}/{self._shot}/{self._cluster}_{self._name}.{self._shot}.test.jsonl"

    def compute_metrics(self, preds, labels):
        result_rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        all_rouge_l, all_rouge_1, all_rouge_2 = [], [], []
        for prediction, label in zip(preds, labels):
            scores = result_rouge_scorer.score(label, prediction)
            rouge1 = scores['rouge1'].fmeasure
            rouge2 = scores['rouge2'].fmeasure
            rougel = scores['rougeL'].fmeasure
            all_rouge_1.append(rouge1)
            all_rouge_2.append(rouge2)
            all_rouge_l.append(rougel)
        avg_rouge_l = np.mean(np.asarray(all_rouge_l))
        avg_rouge_1 = np.mean(np.asarray(all_rouge_1))
        avg_rouge_2 = np.mean(np.asarray(all_rouge_2))
        results = {
            "ROUGE-1": avg_rouge_1,
            "ROUGE-2": avg_rouge_2,
            "ROUGE-L": avg_rouge_l,
        }
        return results