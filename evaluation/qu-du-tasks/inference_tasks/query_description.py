from rouge_score import rouge_scorer
import numpy as np
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate import bleu_score

class QueryDescription():
    def __init__(self, name, shot, setting):
        self._cluster = "query_description"
        self._name = name
        self._shot = shot
        self._setting = setting
        
    
    def get_path(self):
        return f"data/{self._setting}/{self._shot}/{self._cluster}_{self._name}.{self._shot}.test.jsonl"

    def compute_metrics(self, preds, labels):
        result_rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        all_rouge_l = []
        split_labels = [[x.split()] for x in labels]
        split_preds = [x.split() for x in preds]
        bleu1 = corpus_bleu(split_labels, split_preds, weights=(1, 0, 0, 0), smoothing_function=bleu_score.SmoothingFunction().method4)
        bleu2 = corpus_bleu(split_labels, split_preds, weights=(0.5, 0.5, 0, 0), smoothing_function=bleu_score.SmoothingFunction().method4)
        for prediction, label in zip(preds, labels):
            scores = result_rouge_scorer.score(label, prediction)
            rougel = scores['rougeL'].fmeasure
            all_rouge_l.append(rougel)
        avg_rouge_l = np.mean(np.asarray(all_rouge_l))
        results = {
            "BLEU-1": bleu1,
            "BLEU-2": bleu2,
            "ROUGE-L": avg_rouge_l,
        }
        return results