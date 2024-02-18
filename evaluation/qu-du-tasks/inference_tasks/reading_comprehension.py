      
import re
import string
from collections import Counter
from typing import List
import numpy as np
from rouge_score import rouge_scorer
from sklearn.metrics import f1_score, accuracy_score
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate import bleu_score


class ReadingComprehension():
    def __init__(self, name, shot, setting):
        self._cluster = "reading_comprehension"
        self._name = name
        self._shot = shot
        self._setting = setting

    def get_path(self):
        return f"data/{self._setting}/{self._shot}/{self._cluster}_{self._name}.{self._shot}.test.jsonl"

    def compute_metrics(self, preds, labels):
        
        def normalize_text(s):

            def remove_articles(text):
                return re.sub(r'\b(a|an|the)\b', ' ', text)

            def white_space_fix(text):
                return ' '.join(text.split())

            def remove_punc(text):
                exclude = set(string.punctuation)
                return ''.join(ch for ch in text if ch not in exclude)

            def lower(text):
                return text.lower()

            return white_space_fix(remove_articles(remove_punc(lower(s))))


        def calc_exact_match(text: str, answers: List[str]) -> bool:
            """Check if prediction is exactly the same as any of the answers."""
            norm_text = normalize_text(text)
            norm_answers = [normalize_text(ans) for ans in answers]
            return max([(norm_text == norm_ans) for norm_ans in norm_answers])

        def calc_soft_exact_match(text: str, answers: List[str]) -> bool:
            norm_text = normalize_text(text)
            norm_answers = [normalize_text(ans) for ans in answers]
            return max([(norm_ans in norm_text) for norm_ans in norm_answers])

        def calc_unigram_f1(text: str, answers: List[str]) -> float:
            """Calculate unigram f1 score between the text and reference answers."""
            norm_pred = normalize_text(text)
            norm_answers = [normalize_text(ans) for ans in answers]
            common_tokens = [
                Counter(norm_pred) & Counter(norm_ans) for norm_ans in norm_answers
            ]
            num_same = [sum(common.values()) for common in common_tokens]

            score_list = []
            for i, num in enumerate(num_same):
                if num == 0:
                    score_list.append(0.0)
                else:
                    p = 1.0 * num / len(norm_pred)
                    r = 1.0 * num / len(norm_answers[i])
                    f1 = 2 * p * r / (p + r)
                    score_list.append(f1)
            return max(score_list)
        
        if self._name == "boolq":
            # accuracy = np.sum(np.asarray(preds) == np.asarray(labels)) / len(preds)
            labels = np.asarray([x[0] for x in labels])
            acc = accuracy_score(labels, preds)
            f1 = f1_score(labels, preds, average='weighted')

            results = {
                "Acc": acc,
                "F1": f1
            }
            
        elif self._name == "webglm_qa":
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
        else:
            em_scores = [calc_exact_match(pred, label.split(" <=SEP=> ")) for pred, label in zip(preds, labels)]
            f1_scores = [calc_unigram_f1(pred, label.split(" <=SEP=> ")) for pred, label in zip(preds, labels)]
            soft_em_scors = [calc_soft_exact_match(pred, label.split(" <=SEP=> ")) for pred, label in zip(preds, labels)]

            results = {
                # "EM": sum(em_scores) / len(em_scores),
                "F1": sum(f1_scores) / len(f1_scores),
                # "Soft-EM": sum(soft_em_scors) / len(soft_em_scors),
            }

        return results