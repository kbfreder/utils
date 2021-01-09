import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.metrics import make_scorer, classification_report, accuracy_score, confusion_matrix, roc_auc_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_score, recall_score, f1_score


seed = 19

def assess_model(preprocessor, model, X, y, n=5):
    '''Stratified k-fold cross-validation, returns ALL THE THINGS:
    precision, recall, f1-score, confusion matrix, aucs'''
    '''***Think about rewriting this as cross_validate function'''
    pipe =  Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    cv = StratifiedKFold(n_splits=n)

    precs = []
    recalls = []
    f1s = []
    conf_mat = []
    aucs = []


    for train, test in cv.split(X, y):
        y_pred = pipe.fit(X.loc[train], y[train]).predict(X.loc[test])
        y_proba = pipe.fit(X.loc[train], y[train]).predict_proba(X.loc[test])[:,1]
        precs.append(precision_score(y[test],y_pred, average=None))
            # y_pred = model.predict(xtest)
        recalls.append(recall_score(y[test], y_pred, average=None))
        f1s.append(f1_score(y[test], y_pred, average=None))
        conf_mat.append(confusion_matrix(y[test], y_pred))
        aucs.append(roc_auc_score(y[test],y_proba))

    twos = [precs, recalls, f1s]
    data1 = [[s[i][j] for j in range(2) for i in range(3)] for s in twos]
    df1 = pd.DataFrame(data1,
             columns=['Precision-0', 'Recall-0 (Specificty)','F1score-0','Precision-1',
             'Recall-1 (Sensitivity)','F1score-1']).mean()

    data2 = [[s[i][j] for j in range(2) for i in range(2)] for s in conf_mat]
    df2 = pd.DataFrame(data2, columns=['TN','FN','FP','TP']).mean()

    df3 = pd.DataFrame(aucs, columns=['AUC']).mean()

    return df1.append(df2.append(df3))

def false_neg_scorer(y_true, y_pred):
    '''FNR = 1 - TPR = FN / (FN + TP)'''
    conf_mat_nums = confusion_matrix(y_true, y_pred).ravel()
    return 1 - (conf_mat_nums[1] / conf_mat_nums[1] + conf_mat_nums[3])

def specificity_scorer(y_true, y_pred):
    '''Specificity = TN / (TN + FP)'''
    conf_mat_nums = confusion_matrix(y_true, y_pred).ravel()
    return conf_mat_nums[0] / (conf_mat_nums[0] + conf_mat_nums[2])

def precision_neg_class(y_true, y_pred):
    '''Precision-0 = TN / (TN + FN)'''
    conf_mat_nums = confusion_matrix(y_true, y_pred).ravel()
    return conf_mat_nums[0] / (conf_mat_nums[0] + conf_mat_nums[1])

def conf_mat_avg_cv(preprocessor, model, X, y, n=5):
    '''Returns "average" confusion matrix'''
    pipe =  Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    cv = StratifiedKFold(n_splits=n)
    conf_mat = []

    for train, test in cv.split(X, y):
        y_pred = pipe.fit(X.loc[train], y[train]).predict(X.loc[test])
        conf_mat.append(confusion_matrix(y[test], y_pred))

    df_data = [[s[i][j] for j in range(2) for i in range(2)] for s in conf_mat]
    return (pd.DataFrame(df_data,
                         columns=['TN','FN','FP','TP'])).mean()

def assess_preproc_model_auc(preprocessor, model, X, y, n=5):
    pipe =  Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    cv = StratifiedKFold(n_splits=n, random_state=seed)
    scores = cross_validate(pipe, X, y, cv=cv, scoring='roc_auc', n_jobs=-1,
                            return_train_score=False)
    return np.mean(scores['test_score'])

def assess_pipe_auc(pipe, X, y, n=5):
    cv = StratifiedKFold(n_splits=n, random_state=seed)
    scores = cross_validate(pipe, X, y, cv=cv, scoring='roc_auc', n_jobs=-1,
                            return_train_score=False)
    return np.mean(scores['test_score'])

def get_avg_roc_curve(preprocessor, model, X, y, n=5):
    pipe =  Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    cv = StratifiedKFold(n_splits=n)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    for train, test in cv.split(X, y):
        y_proba = pipe.fit(X.loc[train], y[train]).predict_proba(X.loc[test])
        # Compute ROC curve and area the curve
        fpr, tpr, _ = roc_curve(y[test], y_proba[:, 1])
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    return [mean_fpr, mean_tpr, mean_auc, std_auc]

def test_roc_auc(preproc, model, X_train, y_train, X_test, y_test):
    pipe = Pipeline(steps=[('preprocessor', preproc), ('classifier', model)])
    pipe.fit(X_train, y_train)

    y_proba = pipe.predict_proba(X_test)

    fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
    roc_auc = auc(fpr, tpr)

    return [fpr, tpr, roc_auc]

def assess_model_with_resamp(preprocessor, model, res, X, y, n=5):
    '''Tests model with preprocessing steps, and 'res' resampler
    using Stratified k-fold cross-validation'''

    pipe =  Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    cv = StratifiedKFold(n_splits=n, random_state=seed)
    scores = []
    for train, test in cv.split(X, y):
        X_res, y_res = res.fit_resample(X[train], y[train])
        pipe.fit(X_res, y_res)
        y_proba = pipe.predict_proba(X[test])[:,1]
        scores.append(roc_auc_score(y[test], y_proba))

    return np.mean(scores), np.std(scores)

def assess_all_models_with_resamp(preprocessor, model_lib, res, X, y, n=5):
    '''Example:
    model_lib = {'lr':LogisticRegression(solver='liblinear'),
                 'gb': GradientBoostingClassifier(),
                 'nb': GaussianNB()}
    '''

    cv = StratifiedKFold(n_splits=n, random_state=seed)
    scores = defaultdict(list)

    for train, test in cv.split(X, y):
        X_res, y_res = res.fit_resample(X.loc[train], y.loc[train])
        for mod_str, model in model_lib.items():
            pipe =  Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
            pipe.fit(X_res, y_res)
            y_proba = pipe.predict_proba(X.loc[test])[:,1]
            scores[mod_str].append(roc_auc_score(y.loc[test], y_proba))

    for k,v in scores.items():
        scores[k] = sum(v) / n

    return scores

def get_avg_roc_curve(preprocessor, model, X, y, n=5):
    pipe =  Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    cv = StratifiedKFold(n_splits=n)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    for train, test in cv.split(X, y):
        y_proba = pipe.fit(X.loc[train], y[train]).predict_proba(X.loc[test])
        # Compute ROC curve and area the curve
        fpr, tpr, _ = roc_curve(y[test], y_proba[:, 1])
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    return [mean_fpr, mean_tpr, mean_auc, std_auc]

def get_avg_npv_curve(preprocessor, model, X, y, n=5):
    pipe =  Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    cv = StratifiedKFold(n_splits=n)

    fnrs, tnrs, npvs = [], [], []
#     mean_fnr = np.linspace(0, 1, 100)

    for train, test in cv.split(X, y):
        y_proba = pipe.fit(X.loc[train], y[train]).predict_proba(X.loc[test])
        fnr, tnr, npv = [], [], []

        for cutoff in cut:
            pred = np.array(y_proba[:,1] > cutoff)
            surv = y[test]
            fpos = pred * (1 - surv)
            tpos = pred * surv
            fneg = (1 - pred) * surv
            tneg = (1 - pred) * (1 - surv)

            fnr.append(np.sum(fneg) / (np.sum(fneg) + np.sum(tpos)))
            tnr.append(np.sum(tneg) / (np.sum(tneg) + np.sum(fpos)))
            npv.append(np.sum(tneg / (np.sum(tneg) + np.sum(fneg))))

        tnrs.append(tnr)
        fnrs.append(fnr)
        npvs.append(npv)

    mean_tnr = np.mean(tnrs, axis=0)
    mean_fnr = np.mean(fnrs, axis=0)
    mean_npv = np.mean(npvs, axis=0)
    npv_auc = auc(mean_fnr, mean_tnr)
    return [mean_fnr, mean_tnr, npv_auc, mean_npv]
