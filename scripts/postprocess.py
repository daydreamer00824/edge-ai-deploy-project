import numpy as np

def softmax(results : np.ndarray, ) -> np.ndarray:
    logits = results - np.max(results, axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=1, keepdims=True)

def get_topk(results : np.ndarray, k : int = 5):
    probs = softmax(results)

    topk_indices = np.argsort(probs, axis=1)[:, -k:][:, ::-1]
    topk_scores = np.take_along_axis(probs, topk_indices, axis=1)

    return topk_indices, topk_scores