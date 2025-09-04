from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


def correlation_decoder(
    train_rates: np.ndarray,
    train_labels: Sequence[int],
    test_rates: np.ndarray,
    *,
    return_scores: bool = False,
) -> Tuple[np.ndarray, np.ndarray | None]:
    """Decode stimulus identity by correlation to class-average patterns.

    Parameters
    ----------
    train_rates
        Training data of shape ``(n_train, n_features)``.
    train_labels
        Integer labels (e.g., image index) for the training data.
    test_rates
        Test data of shape ``(n_test, n_features)``.
    return_scores
        If *True*, also return the correlation matrix of shape
        ``(n_test, n_classes)``.

    Returns
    -------
    pred : ndarray
        Predicted label for each test sample.
    scores : ndarray | None
        Correlation scores if *return_scores* is *True*, otherwise *None*.
    """
    train_rates = np.asarray(train_rates, dtype=np.float32)
    test_rates = np.asarray(test_rates, dtype=np.float32)
    train_labels = np.asarray(train_labels, dtype=int)

    classes = np.unique(train_labels)
    n_classes = len(classes)
    n_features = train_rates.shape[1]

    # Compute class templates (average response)
    templates = np.zeros((n_classes, n_features), dtype=np.float32)
    for i, cls in enumerate(classes):
        templates[i] = train_rates[train_labels == cls].mean(axis=0)

    # Normalise templates and test vectors for cosine similarity (=corr if mean=0)
    # subtract mean to convert to correlation-like measure
    templates -= templates.mean(axis=1, keepdims=True)
    test_centered = test_rates - test_rates.mean(axis=1, keepdims=True)

    # Normalise by L2 norm
    templates_norm = templates / (np.linalg.norm(templates, axis=1, keepdims=True) + 1e-12)
    test_norm = test_centered / (np.linalg.norm(test_centered, axis=1, keepdims=True) + 1e-12)

    scores = test_norm @ templates_norm.T  # shape (n_test, n_classes)
    pred_cls_indices = scores.argmax(axis=1)
    pred = classes[pred_cls_indices]
    return (pred, scores) if return_scores else (pred, None)


def accuracy(pred: Sequence[int], true: Sequence[int]) -> float:
    """Compute classification accuracy."""
    pred = np.asarray(pred)
    true = np.asarray(true)
    return float((pred == true).mean())


# -----------------------------------------------------------------------------
# Multinomial logistic-regression decoder using scikit-learn
# -----------------------------------------------------------------------------

def logistic_decoder(
    train_rates: np.ndarray,
    train_labels: Sequence[int],
    test_rates: np.ndarray,
    *,
    max_iter: int = 3000,
    n_jobs: int | None = None,
):
    """Multinomial logistic-regression decoder (softmax).

    Parameters
    ----------
    train_rates, train_labels, test_rates
        As in :func:`correlation_decoder`.
    max_iter
        Maximum iterations for optimiser.
    n_jobs
        Parallel jobs for scikit-learn.  *None* = default (1).

    Returns
    -------
    pred : ndarray
        Predicted labels for *test_rates*.
    clf : sklearn.linear_model.LogisticRegression
        Trained classifier instance (returned in case caller wants to inspect
        coefficients).
    """

    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError as e:
        raise ImportError("scikit-learn is required for logistic_decoder") from e

    clf = LogisticRegression(
        penalty="l2",
        solver="lbfgs",            # faster for dense, low-dim problems
        tol=1e-3,
        max_iter=max_iter,
        n_jobs=n_jobs,
    )

    clf.fit(train_rates, train_labels)
    pred = clf.predict(test_rates)
    return pred, clf 