from sklearn.model_selection import KFold
from sklearn.metrics import get_scorer
from sklearn.base import clone
import numpy as np
from joblib import Parallel, delayed

def _default_scorer(estimator, X, y):
    estimator.score(X, y)

@delayed
def _fit_score_predict(estimator, X, y, scoring, predict_method, train_idx, validate_idx, fit_params):
    estimator.fit(X[train_idx], y[train_idx], **fit_params)
    score = scoring(estimator, X[validate_idx], y[validate_idx])
    predictions = getattr(estimator, predict_method)(X[validate_idx])
    return score, predictions, validate_idx

def cross_val_score_predict(estimator, X, y, groups=None, scoring=None, predict_method=None, cv=5, n_jobs=None, fit_params=None):
    """
    A combination of sklearn's cross_val_score and cross_val_predict that only cross-validates once. Like
    cross_val_score, the scores are computed separately for each validation fold, which avoids the
    pitfalls of calculating a single score on the return value of cross_val_predict.
    """
    if scoring is None:
        scoring = _default_scorer
    else:
        scoring = get_scorer(scoring)
    if predict_method is None:
        predict_method = 'decision_function' if hasattr(estimator, 'decision_function') else 'predict_proba'
    if isinstance(cv, int):
        cv = KFold(cv)
    if fit_params is None:
        fit_params = {}
    
    parallel = Parallel(n_jobs)
    results = parallel(
        _fit_score_predict(clone(estimator), X, y, scoring, predict_method, train_idx, validate_idx, fit_params)
        for train_idx, validate_idx in cv.split(X, y, groups))
    
    scores = np.array([s for s, _, _ in results])

    first_fold_prediction_shape = results[0][1].shape
    if len(first_fold_prediction_shape) == 1:
        prediction_shape = y.shape
    else:
        num_classes = first_fold_prediction_shape[1:]
        prediction_shape = y.shape + num_classes
    predictions = np.zeros(prediction_shape)
    for _, fold_predictions, validate_idx in results:
        predictions[validate_idx] = fold_predictions

    return scores, predictions
