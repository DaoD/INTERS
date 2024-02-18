import sys
import pytz
import torch
import pathlib
import json
import string
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from collections import defaultdict
from typing import List, Any, Dict


def makedirs(path):
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return path


def split_file_dir_name_ext(path):
    """Return the directory, name, and extension of a given file."""
    p = pathlib.Path(path)
    assert p.is_file()
    return p.parent, p.stem, p.suffix


def get_max_length_in_nested_lists(lst):
    if len(lst) and isinstance(lst[0], list):
        lengths = []
        for elem in lst:
            length = get_max_length_in_nested_lists(elem)
            lengths.append(length)
        max_length = max(lengths)
        return max_length
    else:
        return len(lst)


def pad_nested_lists(lst, max_length, padding_value, padding_side="right"):
    if isinstance(lst, list) and len(lst) and isinstance(lst[0], list):
        masks = []
        for i, elem in enumerate(lst):
            lst[i], mask = pad_nested_lists(elem, max_length, padding_value, padding_side)
            masks.append(mask)
        return lst, masks
    elif isinstance(lst, list):
        if padding_side == "right":
            mask = [1] * len(lst) + [0] * (max_length - len(lst))
            lst = lst + [padding_value for _ in range(max_length - len(lst))]
            return lst, mask
        else:
            mask = [0] * (max_length - len(lst)) + [1] * len(lst)
            lst = [padding_value for _ in range(max_length - len(lst))] + lst
            return lst, mask
    else:
        raise NotImplementedError(f"Unrecognized type {lst}")

def mask_nested_lists(lst, mask_target, mask_value=0):
    if isinstance(lst[0], list):
        for i, elem in enumerate(lst):
            lst[i] = mask_nested_lists(elem, mask_target, mask_value)
        return lst
    else:
        return [x if x != mask_target else mask_value for x in lst]

def are_elements_of_same_length(lst: List):
    if not isinstance(lst[0], list):
        return False

    length = len(lst[0])
    return all(len(x) == length if isinstance(x, list) else False for x in lst)


def normalize_text(text, ignore_case=True, ignore_punctuation=True, ignore_space=True, ignore_number=False):
    if isinstance(text, str):
        text = [text]
        unpack = True
    else:
        unpack = False
    if ignore_case:
        text = np.char.lower(text)
    if ignore_punctuation:
        repl_table = string.punctuation.maketrans("", "", string.punctuation)
        text = np.char.translate(text, table=repl_table)
    if ignore_number:
        repl_table = string.digits.maketrans("", "", string.digits)
        text = np.char.translate(text, table=repl_table)
    if ignore_space:
        for i, words in enumerate(np.char.split(text)):
            text[i] = " ".join(words)
    if isinstance(text, np.ndarray):
        text = text.tolist()
    if unpack:
        text = text[0]
    return text


class DatasetProcessFn:
    """Wrapper for any user-defined process function for huggingface datasets.

    1. Process batched examples by looping the process function over them;
    2. Gather returned examples if any data augmentation happens with augment=True;
    3. Pass indices of examples inside the process function with _index keywords if they exist.

    The wrapped function should take in any needed columns and return a dict with 1 or more samples.
    """
    def __init__(self, augment=False):
        self.augment = augment

    def __call__(self, _process_fn):
        def process(*args):
            sample_or_batch_sample = args[0]
            if len(args) == 1:
                pass
            elif len(args) == 2:
                indices = args[1]
                # detach the slice so that _index will not be set in the original data
                sample_or_batch_sample = sample_or_batch_sample.copy()
                sample_or_batch_sample["_index"] = indices
            else:
                raise NotImplementedError(f"Found more than 2 arguments {args}!")

            keys = list(sample_or_batch_sample.keys())
            func_args = [sample_or_batch_sample[k] for k in keys]
            
            # FIXME: if all values in one sample are of the same length, this would fail
            if are_elements_of_same_length(func_args):
                outputs = defaultdict(list)
                for arg in zip(*func_args):
                    # get each element in a batch
                    kwargs = {keys[j]: arg[j] for j in range(len(arg))}
                    output = _process_fn(**kwargs)
                    if output is not None:
                        for k, v in output.items():
                            if self.augment:
                                outputs[k].extend(v)
                            else:
                                outputs[k].append(v)
            else:
                outputs = _process_fn(**sample_or_batch_sample)
                if outputs is None:
                    raise ValueError(f"Found None returned from process_fn. Make sure you set 'batched=True' when trying to augment/distract samples in the datasets!")
            return dict(outputs)
        return process


@dataclass
class DefaultDataCollator:
    """
    Data collator that can:
    1. Dynamically pad all inputs received. The inputs must be dict of lists.
    2. Add position_ids based on attention_mask if required.
    """
    tokenizer: Any = None
    attention_padding_value: int = 0
    label_padding_value: int = -100
    add_position_ids: bool = False

    def __call__(self, batch_elem: List) -> Dict[str, Any]:
        first_elem = batch_elem[0]
        return_batch = {}
        
        for key, value in first_elem.items():
            # HACK: any key containing attention_mask must be attention_mask
            # important to assign different pad token for different types of inputs
            if "attention_mask" in key:
                pad_token_id = self.attention_padding_value
            elif "label" in key:
                pad_token_id = self.label_padding_value
            else:
                pad_token_id = self.tokenizer.pad_token_id

            batch_value = [elem[key] for elem in batch_elem]
            # pad all lists and nested lists
            if isinstance(value, list):
                max_length = get_max_length_in_nested_lists(batch_value)
                batch_value, _ = pad_nested_lists(batch_value, max_length, pad_token_id, self.tokenizer.padding_side)

            try:
                return_batch[key] = torch.tensor(batch_value)
            except:
                # handle strings and None
                return_batch[key] = batch_value

            if "attention_mask" in key and self.add_position_ids:
                value = return_batch[key]
                position_ids = value.cumsum(-1) - 1
                position_ids = position_ids.masked_fill(value == 0, 0)
                return_batch[key.replace("attention_mask", "position_ids")] = position_ids
        return return_batch


class FileLogger:
    def __init__(self, log_file) -> None:
        self.log_file = log_file
    
    def log(self, metrics, **kwargs):
        with open(self.log_file, "a+") as f:
            # get current time
            tz = pytz.timezone('Asia/Shanghai')
            time = f"{'Time': <10}: {json.dumps(datetime.now(tz).strftime('%Y-%m-%d, %H:%M:%S'), ensure_ascii=False)}\n"
            print(time)
            command = f"{'Command': <10}: {json.dumps(' '.join(sys.argv), ensure_ascii=False)}\n"
            print(command)
            metrics = f"{'Metrics': <10}: {json.dumps(metrics, ensure_ascii=False)}\n"
            msg = time + command

            for key, value in kwargs.items():
                x = f"{key: <10}: {json.dumps(value, ensure_ascii=False)}\n"
                print(x)
                msg += x
            msg += metrics
            print(metrics)
            f.write(str(msg) + "\n")
