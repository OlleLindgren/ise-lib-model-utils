__version__ = version = 'v0.0.3'

from typing import Tuple
import numpy as np

from persistence import DependencySpecType

# Sklearn

# Pipeline
from sklearn.pipeline import Pipeline

# Dummy
from sklearn.dummy import DummyRegressor

# Feature selection
from sklearn.feature_selection import (
    SelectKBest, 
    # SelectPercentile,         # equiavlent to SelectKBest
    # GenericUnivariateSelect   # equiavlent to SelectKBest
) # Supports callable
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.feature_selection import RFE, SequentialFeatureSelector
from sklearn.feature_selection import SelectFromModel

# Preprocessing
from sklearn.preprocessing import StandardScaler, Normalizer, QuantileTransformer

# Regression
from sklearn.linear_model import (
    ElasticNet, 
    SGDRegressor
)

from sklearn.ensemble import (
    RandomForestRegressor, 
    ExtraTreesRegressor, 
    HistGradientBoostingRegressor,
    BaggingRegressor, 
    VotingRegressor, 
    AdaBoostRegressor,
)
from sklearn.neighbors import KNeighborsRegressor
# from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.cross_decomposition import PLSRegression
# from sklearn.svm import LinearSVR

def fit_model(
    model: Pipeline,
    X_train: np.array,
    y_train: np.array) -> dict:
    """Fit a sklearn model"""

    # Ensure y_train is positive so MSLE can be used
    if LOSS=="MSLE":
        y_train[y_train<1e-8] = 1e-8

    model.fit(X_train, y_train)

def eval_model(
    model: Pipeline,
    dummy: Pipeline,
    X_test: np.array,
    y_test: np.array
    ) -> dict:

    # Evaluate model
    
    from sklearn.metrics import mean_squared_error, mean_squared_log_error, confusion_matrix
    
    # Ensure y_test is positive so MSLE can be used
    if LOSS=="MSLE":
        y_test[y_test<1e-8] = 1e-8

    prediction = model.predict(X_test)
    prediction[prediction<1e-8] = 1e-8
    msle = mean_squared_log_error(prediction.flatten() , y_test.flatten())
    mse = mean_squared_error(prediction.flatten() , y_test.flatten())
    rmse = np.sqrt(mse)
    crl = np.corrcoef(prediction.flatten(), y_test.flatten())[0, 1] or 0.0
    
    dm_prediction = dummy.predict(X_test)
    dm_prediction[dm_prediction<1e-8] = 1e-8
    dm_msle = mean_squared_log_error(dm_prediction.flatten() , y_test.flatten())
    dm_mse = mean_squared_error(dm_prediction.flatten() , y_test.flatten())
    dm_rmse = np.sqrt(dm_mse)
    dm_crl = np.corrcoef(y_test, dm_prediction)[0, 1] or 0.0

    # Correlation between predicted increase and actual increase
    edge_over_dummy = np.corrcoef(
        prediction.flatten() - dm_prediction.flatten(), 
        y_test.flatten() - dm_prediction.flatten()
    )[0, 1]

    # Probability of being closer to response than dummy
    p_beat_dummy = np.count_nonzero(
        (prediction.flatten() - y_test.flatten())**2
        <
        (dm_prediction.flatten() - y_test.flatten())**2
    ) / len(y_test.flatten())

    # Probability of predicting "in the right direction", i.e. up if up, down if down
    p_right_direction = np.count_nonzero(
        (prediction.flatten() - dm_prediction.flatten())
        *
        (y_test.flatten() - dm_prediction.flatten())
        >
        0
    ) / len(y_test.flatten())

    # Confusion matrix, where positive=increase
    increases = y_test.flatten() > dm_prediction.flatten()
    pred_increases = prediction.flatten() > dm_prediction.flatten()
    tn, fp, fn, tp = map(int, confusion_matrix(increases, pred_increases).ravel())

    try:
        f1 = tp / (tp + (fp + fn) / 2)
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        dor = (tp / fn) / (fp / tn)
    except ZeroDivisionError:
        f1 = 0.0
        sensitivity = 0.0
        specificity = 0.0
        precision = 0.0
        recall = 0.0
        dor = 0.0

    # Build loss dictionary
    eval_metrics = {
        "mse": (mse, dm_mse), 
        "msle": (msle, dm_msle), 
        "rmse": (rmse, dm_rmse), 
        "crl": (crl, dm_crl),
        "edge_over_dummy": edge_over_dummy,
        "p_beat_dummy": p_beat_dummy,
        "p_right_direction": p_right_direction,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "f1": f1,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "recall": recall,
        "dor": dor,
        "dummy_type": 'SmartDummy' if isinstance(dummy, SmartDummy) else f'{dummy.__class__.__name__}({dummy.strategy=})',
        "__version__": __version__
    }

    return eval_metrics

# Model generators

def random_model(X_spec: DependencySpecType, y_spec: DependencySpecType) -> Pipeline:
    # Generate a random model that fits with some X -> y DependencySpecs.
    # This is intended to be continuously re-run hundreds of times,
    # where some other process filters out the "good" models.

    # Number of processors to use
    from multiprocessing import cpu_count
    n_processors_fit = min(1, cpu_count()-1)

    assert isinstance(X_spec, DependencySpecType)
    p = len(X_spec)
    min_features = max(3, int(p * 0.05))
    max_features = p

    def get_estimator(with_fi: bool = False):
        choices = [
            ElasticNet(
                alpha=np.random.random(),
                l1_ratio=np.random.random()
            ), 
            SGDRegressor(
                loss=np.random.choice(['squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive']),
                penalty=np.random.choice(['l2', 'l1', 'elasticnet']),
                alpha=np.random.lognormal(-4.0, 4.0)
            ), 
            # GaussianProcessRegressor(), 
            # PLSRegression()
        ]
        if not with_fi:
            choices.extend([
                KNeighborsRegressor(
                    metric=np.random.choice(['euclidean', 'minkowski']),
                    n_neighbors=np.random.randint(10, 100),
                ), 
            ])
        return np.random.choice(choices)

    def get_preprocessor():
        return np.random.choice([
            StandardScaler(), 
            Normalizer(), 
            QuantileTransformer(),
            None
        ])

    def get_feature_selector(allow_recursive: bool):
        fast_choices = [
            SelectKBest(
                score_func=np.random.choice([f_regression, mutual_info_regression]),
                k=np.random.randint(min_features, max_features+1)
            ), 
            SelectFromModel(
                estimator=get_estimator(with_fi=True),
                threshold=f"{1+np.random.random():.3f}*mean",
                max_features=max_features
            )
        ]
        recursive_choices = [
            RFE(
                estimator=get_estimator(with_fi=True),
                n_features_to_select=np.random.randint(min_features, max_features+1),
                step=.1
            ), 
            SequentialFeatureSelector(
                estimator=get_estimator(with_fi=True),
                n_features_to_select=np.random.randint(min_features, max_features+1),
                direction=np.random.choice(['forward', 'backward']),
                n_jobs=n_processors_fit
            )
        ]
        return np.random.choice(
            fast_choices + recursive_choices if allow_recursive
            else fast_choices)

    def ensemble_estimator():
        n_estimators = np.random.randint(10, 200)
        return np.random.choice([
            BaggingRegressor(
                base_estimator=get_estimator(), 
                n_estimators=n_estimators,
                n_jobs=n_processors_fit
            ),
            VotingRegressor(
                estimators=[(f"estimator_{i}", get_estimator()) 
                            for i in range(n_estimators)],
                n_jobs=n_processors_fit
            ),
            AdaBoostRegressor(
                base_estimator=get_estimator(),
                n_estimators=n_estimators
            ),
            RandomForestRegressor(
                n_estimators=n_estimators, 
                min_samples_split=20, 
                min_samples_leaf=10,
                n_jobs=n_processors_fit,
            ),
            ExtraTreesRegressor(
                n_estimators=n_estimators, 
                min_samples_split=20, 
                min_samples_leaf=10,
                n_jobs=n_processors_fit,
            ),
            HistGradientBoostingRegressor(
                loss=np.random.choice(['least_squares', 'least_absolute_deviation', 'poisson'])
            )
        ])

    return Pipeline([
        ("feature_selection", get_feature_selector(allow_recursive=False)),
        ("preprocessor", get_preprocessor()),
        ("model", ensemble_estimator()),
    ])

def dummy(strategy: str='median') -> Pipeline:
    # Generate a dummy model that fits with some X -> y DependencySpecs.

    return DummyRegressor(strategy=strategy)

from sklearn.base import BaseEstimator

class SmartDummy(BaseEstimator):

    X_spec: DependencySpecType
    y_spec: DependencySpecType

    y_shape: Tuple

    def __init__(self, X_spec: DependencySpecType, y_spec: DependencySpecType) -> None:
        """SmartDummy returns the col in X_spec that best matches y_spec"""
        super().__init__()

        if not isinstance(X_spec, DependencySpecType):
            raise TypeError(f"X_spec must be DependencySpecType, not {type(X_spec)}")
        if not isinstance(y_spec, DependencySpecType):
            raise TypeError(f"y_spec must be DependencySpecType, not {type(y_spec)}")

        self.index = self.get_index(X_spec, y_spec)

        self.X_spec = X_spec
        self.y_spec = y_spec

    @staticmethod
    def get_index(X_spec: DependencySpecType, y_spec: DependencySpecType):
        """Get index of y_spec from X_spec"""

        if len(y_spec) > 1:
            raise NotImplementedError(f"y_spec of type with length {len(y_spec)} is not implemented")
        # It is impossible to instantiate any DependencySpecType with len < 1

        for i, dep in enumerate(X_spec):
            if dep==y_spec.dependencies[0]:
                return i

        raise ValueError("X_spec does not contain y_spec; Cannot initialize SmartDummy")

    def fit(self, X, y) -> None:
        self.y_shape = y.shape
        return

    def predict(self, X):
        result = X[:, self.index]
        return result if result.shape==self.y_shape else result.ravel()
