import os
import json
import logging
import inspect
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Tuple
from .modeling import RerankResult
from .utils import makedirs, split_file_dir_name_ext, normalize_text

logger = logging.getLogger(__name__)


class Metric:
    """Class for computing metrics and some post-processings."""
    @classmethod
    def get_metric_fn(cls, metric_names, **kwds):
        assert isinstance(metric_names, list) or isinstance(metric_names, tuple), "You must pass metric_names in a list or tuple!"
        all_metrics = {}
        # get all methods
        all_implemented_fns = [x[0] for x in inspect.getmembers(cls, predicate=inspect.isfunction) if not x[0].startswith("_")]

        def compute_metrics(*args, **kwargs):
            for metric_name in metric_names:
                # call corresponding method
                if metric_name in all_implemented_fns:
                    metric_fn = getattr(cls, metric_name)
                    metric = metric_fn(**kwds)(*args, **kwargs)
                    # NOTE: some metric_fn are only used for post-processing and saving results, which return None by default
                    if metric is not None:
                        all_metrics.update(metric)
                else:
                    raise NotImplementedError(f"Metric {metric_name} not implemented!")
            return all_metrics
        return compute_metrics

    @staticmethod
    def _get_save_path(eval_data, output_dir=None, field="result", save_name=None):
        """
        if output_dir is None:
            -> {eval_data_dir}/{eval_data_name}.{field}.{save_name}.{eval_data_ext}
        else:
            -> {output_dir}/{eval_data_name}.{field}.{save_name}.{eval_data_ext}
        """
        eval_data_dir, eval_data_name, eval_data_ext = split_file_dir_name_ext(eval_data)
        if output_dir is None:
            output_dir = eval_data_dir
        fields = [eval_data_name, field]
        if save_name is not None:
            fields.append(save_name)
        save_path = os.path.join(output_dir, ".".join(fields) + eval_data_ext)
        makedirs(save_path)
        return save_path

    @staticmethod
    def _save_generation_result(result, path, eval_data=None):        
        if eval_data is not None:
            items = {}
            with open(eval_data, encoding="utf-8") as f:
                for i, line in enumerate(f):
                    item = json.loads(line)
                    if "query_id" in eval_data:
                        index = item["query_id"]
                    else:
                        index = i
                    items[index] = item
        with open(path, "w") as f:
            for index, pred in result.items():
                res = {"query_id": index}
                if eval_data is not None:
                    item = items[index]
                    res.update(item)
                res["pred"] = pred
                f.write(json.dumps(res, ensure_ascii=False) + "\n")

    @staticmethod
    def _save_rerank_result(results: Dict[int, List[RerankResult]], path, eval_data=None):        
        if eval_data is not None:
            items = {}
            with open(eval_data, encoding="utf-8") as f:
                for i, line in enumerate(f):
                    full_item = json.loads(line)
                    # we only save the index and score of positive samples
                    item = {
                        "pos_index": full_item["pos_index"],
                        "pos_score": full_item["pos_score"],
                    }
                    if "query_id" in eval_data:
                        index = full_item["query_id"]
                    else:
                        index = i
                    items[index] = item
        
        with open(path, "w") as f:
            for index, preds in results.items():
                item = {
                    "query_id": index,
                    "doc_index": [x.doc_id for x in preds],
                    "doc_score": [x.doc_score for x in preds],
                }
                if eval_data is not None:
                    data_item = items[index]
                    item.update(data_item)
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    @staticmethod
    def _prepare_label_for_generation(eval_data):
        labels = {}
        with open(eval_data) as f:
            for i, line in enumerate(f):
                item = json.loads(line)
                if "query_id" in item:
                    index = item["query_id"]
                else:
                    index = i
                label = item["completion"]
                labels[index] = label
        return labels

    @staticmethod
    def _prepare_label_for_retrieval(eval_data):
        labels = {}
        with open(eval_data) as f:
            for i, line in enumerate(f):
                item = json.loads(line)
                if "query_id" in item:
                    query_id = item["query_id"]
                else:
                    query_id = i
                # save positive indices and their scores (for computing ndcg)
                labels[query_id] = (item["pos_index"], item["pos_score"])
        return labels
    
    @staticmethod
    def _prepare_pred_for_retrieval(preds:List[RerankResult]) -> List[RerankResult]:
        valid_preds = [x for x in preds if x.doc_id > -1]
        return valid_preds

    @staticmethod
    def mrr(eval_data=None, cutoffs=[10], **kwds):
        if eval_data is not None:
            data_labels = Metric._prepare_label_for_retrieval(eval_data)
        
        def compute_metric(results:Dict[int, List[RerankResult]], labels:[Dict[int, Tuple[List[int], List[int]]]]=None, **kwargs):
            if labels is None:
                labels = data_labels
            
            mrrs = np.zeros(len(cutoffs))
            counts = 0

            for query_id, preds in results.items():
                pos_indices, pos_scores = labels[query_id]
                # remove irrelevant documents
                pos_indices = [pos_index for i, pos_index in enumerate(pos_indices) if pos_scores[i] > 0]
                if len(pos_indices) == 0:
                    continue

                preds = Metric._prepare_pred_for_retrieval(preds)
                jump = False
                counts += 1

                for i, pred in enumerate(preds, 1):
                    if pred.doc_id in pos_indices:
                        for k, cutoff in enumerate(cutoffs):
                            if i <= cutoff:
                                mrrs[k] += 1 / i
                        jump = True
                    if jump:
                        break

            mrrs /= counts

            metric = {}
            for i, cutoff in enumerate(cutoffs):
                mrr = mrrs[i]
                metric[f"mrr@{cutoff}"] = mrr

            return metric
        return compute_metric

    @staticmethod
    def recall(eval_data=None, cutoffs=[10], **kwds):
        if eval_data is not None:
            data_labels = Metric._prepare_label_for_retrieval(eval_data)
        
        def compute_metric(results:Dict[int, List[RerankResult]], labels:[Dict[int, Tuple[List[int], List[int]]]]=None, **kwargs):
            if labels is None:
                labels = data_labels

            recalls = np.zeros(len(cutoffs))
            counts = 0

            for query_id, preds in results.items():
                pos_indices, pos_scores = labels[query_id]
                # remove irrelevant documents
                pos_indices = [pos_index for i, pos_index in enumerate(pos_indices) if pos_scores[i] > 0]
                if len(pos_indices) == 0:
                    continue

                preds = Metric._prepare_pred_for_retrieval(preds)
                preds_indices = [x.doc_id for x in preds]
                counts += 1

                for k, cutoff in enumerate(cutoffs):
                    recall = np.intersect1d(pos_indices, preds_indices[:cutoff])
                    recalls[k] += len(recall) / len(pos_indices)

            recalls /= counts

            metric = {}
            for i, cutoff in enumerate(cutoffs):
                recall = recalls[i]
                metric[f"recall@{cutoff}"] = recall

            return metric
        return compute_metric
    
    @staticmethod
    def ndcg(eval_data=None, cutoffs=[10], **kwds):
        if eval_data is not None:
            data_labels = Metric._prepare_label_for_retrieval(eval_data)

        def compute_metric(results:Dict[int, List[RerankResult]], labels:[Dict[int, Tuple[List[int], List[int]]]]=None, **kwargs):
            if labels is None:
                labels = data_labels

            ndcgs = np.zeros(len(cutoffs))
            counts = 0

            for query_id, preds in results.items():
                pos_indices, pos_scores = labels[query_id]
                preds = Metric._prepare_pred_for_retrieval(preds)

                pos_indices_to_scores = {k: v for k, v in zip(pos_indices, pos_scores)}
                if len(pos_indices_to_scores) == 0:
                    continue

                dcg = np.zeros(len(cutoffs))
                idcg = np.zeros(len(cutoffs))
                counts += 1

                for i, pred in enumerate(preds, 1):
                    if pred.doc_id in pos_indices:
                        for k, cutoff in enumerate(cutoffs):
                            if i <= cutoff:
                                # get the relevance score of the pred
                                dcg[k] += (2 ** pos_indices_to_scores[pred.doc_id] - 1) / np.log2(i + 1)

                # descendingly sort positives to acquire the ideal ranking
                ideal_ranking = sorted(pos_scores, reverse=True)
                for j, y in enumerate(ideal_ranking, 1):
                    for k, cutoff in enumerate(cutoffs):
                        if j <= cutoff:
                            idcg[k] += (2 ** y - 1) / np.log2(j + 1)

                ndcgs += dcg / idcg

            ndcgs /= counts

            metric = {}
            for i, cutoff in enumerate(cutoffs):
                ndcg = ndcgs[i]
                metric[f"ndcg@{cutoff}"] = ndcg
            return metric
        return compute_metric
