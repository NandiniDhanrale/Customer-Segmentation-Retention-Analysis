"""Leakage-aware chronological model evaluation."""
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

FEATURES = ['recency','frequency','monetary','tenure','mean_order_value']

def train_and_evaluate(rows, horizon_days=90):
    if rows.empty or rows.snapshot.nunique() < 8:
        return pd.DataFrame(), None, 'Not enough distinct historical snapshots for leakage-aware evaluation.'
    dates = sorted(pd.to_datetime(rows.snapshot.unique()))
    n = len(dates)
    val_start, test_start = dates[int(n*.6)], dates[int(n*.8)]
    gap = pd.Timedelta(days=horizon_days)
    # Label windows for training must end before validation starts;
    # label windows for validation must end before the test period starts.
    train = rows[rows.snapshot < val_start-gap]
    val = rows[(rows.snapshot >= val_start) & (rows.snapshot < test_start-gap)]
    test = rows[rows.snapshot >= test_start]
    if any(len(x)<20 or x.inactive_90d.nunique()<2 for x in (train,val,test)):
        return pd.DataFrame(), None, 'Insufficient samples or class diversity after 90-day temporal embargo.'
    models = {
      'Majority baseline': DummyClassifier(strategy='most_frequent'),
      'Logistic regression': make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000,class_weight='balanced')),
      'Histogram gradient boosting': HistGradientBoostingClassifier(random_state=42,max_iter=100,max_leaf_nodes=15)
    }
    trained, val_scores = {}, {}
    for name, model in models.items():
        model.fit(train[FEATURES],train.inactive_90d)
        trained[name] = model
        val_scores[name] = average_precision_score(val.inactive_90d,model.predict_proba(val[FEATURES])[:,list(model.classes_).index(1)])
    winner = max(('Logistic regression','Histogram gradient boosting'),key=lambda k:val_scores[k])
    results = []
    for name, model in trained.items():
        score = model.predict_proba(test[FEATURES])[:,list(model.classes_).index(1)]
        predicted = (score>=.5).astype(int)
        y = test.inactive_90d
        results.append(dict(model=name,pr_auc=average_precision_score(y,score),
          precision=precision_score(y,predicted,zero_division=0),recall=recall_score(y,predicted,zero_division=0),
          f1=f1_score(y,predicted,zero_division=0),roc_auc=roc_auc_score(y,score),
          tn_fp_fn_tp=str(confusion_matrix(y,predicted,labels=[0,1]).ravel().tolist())))
    return pd.DataFrame(results), trained[winner], f'Selected {winner} by validation PR-AUC; test window is held out with a {horizon_days}-day embargo.'
