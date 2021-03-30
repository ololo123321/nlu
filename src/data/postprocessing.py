from typing import List
import numpy as np
from .base import SpanExtended


def get_valid_spans(logits: np.ndarray, is_flat_ner: bool) -> List[SpanExtended]:
    """
    https://arxiv.org/abs/2005.07150

    :param logits: np.array of shape [num_tokens, num_tokens, num_labels]
    :param is_flat_ner:
    :return:
    """
    labels = logits.argmax(-1)  # [num_tokens, num_tokens]
    candidates = []
    for start, end in zip(*np.where(labels != 0)):
        if start <= end:
            label = labels[start, end]
            score = logits[start, end, label]
            span = SpanExtended(start=start, end=end, label=label, score=score)
            candidates.append(span)

    candidates = sorted(candidates, reverse=True, key=lambda x: x.score)
    res = []
    for candidate in candidates:
        for approved in res:
            if (candidate.start < approved.start <= candidate.end < approved.end) \
                    or (approved.start < candidate.start <= approved.end < candidate.end):
                # for both nested and flat ner no clash is allowed
                break
            if is_flat_ner and \
                    (candidate.start <= approved.start <= approved.end <= candidate.end
                     or approved.start <= candidate.start <= candidate.end <= approved.end):
                # for flat ner nested mentions are not allowed
                break
        else:
            res.append(candidate)
    return res
