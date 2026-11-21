"""The thirteen classical classifiers and the majority-class reference (fixed hyperparameters)."""
from ml_common import SEED
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.dummy import DummyClassifier
from sklearn.calibration import CalibratedClassifierCV
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
def models():
    return {
     'LightGBM':lambda:LGBMClassifier(n_estimators=600,learning_rate=0.05,num_leaves=31,subsample=0.8,subsample_freq=1,colsample_bytree=0.8,class_weight='balanced',random_state=SEED,n_jobs=2,verbose=-1),
     'XGBoost':lambda:XGBClassifier(n_estimators=600,learning_rate=0.05,max_depth=6,subsample=0.8,colsample_bytree=0.8,random_state=SEED,n_jobs=2,tree_method='hist'),
     'CatBoost':lambda:CatBoostClassifier(iterations=800,learning_rate=0.08,depth=6,auto_class_weights='Balanced',random_seed=SEED,verbose=0,thread_count=2),
     'Extra Trees':lambda:ExtraTreesClassifier(500,class_weight='balanced',random_state=SEED,n_jobs=2),
     'Random Forest':lambda:RandomForestClassifier(500,class_weight='balanced',random_state=SEED,n_jobs=2),
     'KNN':lambda:KNeighborsClassifier(5,weights='distance'),
     'SVM (RBF)':lambda:SVC(C=10,gamma='scale',class_weight='balanced',probability=True,random_state=SEED),
     'Logistic regression':lambda:LogisticRegression(C=1.0,max_iter=2000,class_weight='balanced'),
     'LDA':lambda:LinearDiscriminantAnalysis(),
     'SVM (linear)':lambda:CalibratedClassifierCV(LinearSVC(C=1.0,class_weight='balanced',random_state=SEED),cv=3),
     'Decision tree':lambda:DecisionTreeClassifier(class_weight='balanced',random_state=SEED),
     'AdaBoost':lambda:AdaBoostClassifier(n_estimators=300,random_state=SEED),
     'Naive Bayes':lambda:GaussianNB(),
     'Dummy (majority)':lambda:DummyClassifier(strategy='most_frequent'),
    }
