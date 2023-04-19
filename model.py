"""
This module is used to demonstrate how we designed our ensemble classifier.
It is worth noting that the parameters of the model are obtained by preliminary tuning using grid search,
which does not achieve the overall optimization in order to balance different frequency subbands.
We suggest using optuna for further tuning if necessary.
"""

from lightgbm import LGBMClassifier
from mlxtend.classifier import StackingCVClassifier
from sklearn.ensemble import GradientBoostingClassifier as GBM
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import LinearSVC

# Base learners
base_SVM = LinearSVC(max_iter=10000)
base_LR = LogisticRegressionCV(solver='liblinear', class_weight='balanced', cv=6, max_iter=10000,
                               n_jobs=-1)
base_GBM = GBM(learning_rate=0.05, n_estimators=240, max_depth=11, max_features=15,
               min_samples_split=550, min_samples_leaf=100, subsample=0.75,
               random_state=2019)
base_LGBM = LGBMClassifier(objective='binary', boosting_type='gbdt', metrics='binary', random_state=2020,
                           learning_rate=0.05, n_estimators=120, max_depth=7, num_leaves=20,
                           max_bin=95, min_split_gain=0.4, min_child_samples=45, min_child_weight=0.0005,
                           bagging_fraction=0.8, feature_fraction=0.8, bagging_freq=15)

# Ensemble classifier & mate learner
ensemble_classifier = StackingCVClassifier(classifiers=[base_SVM, base_GBM, base_LGBM, base_LR],
                                           meta_classifier=LogisticRegressionCV(solver='liblinear',
                                                                                class_weight='balanced', cv=6,
                                                                                max_iter=10000,
                                                                                n_jobs=-1),
                                           cv=4, use_features_in_secondary='True', n_jobs=-1)
