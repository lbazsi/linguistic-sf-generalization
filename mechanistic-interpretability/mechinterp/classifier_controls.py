"""CPU-only grouped-split lexical/length controls for probe interpretation.

Same semantic-ID train/development/test groups as the activation classifier.
Reports accuracy/AUC, not claims of value endorsement or feature abstraction.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from .common import FEATURES, load_config, run_dir, write_json
from .experiments import probe_dataset

def evaluate_binary(y, scores):
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score
    y=np.asarray(y,dtype=int)
    if not len(y) or len(set(y))!=2:raise ValueError("Need both classes to audit a classifier")
    p=np.asarray(scores,dtype=float)
    return {"n":len(y),"balanced_accuracy":float(balanced_accuracy_score(y,p>=0)),
            "roc_auc":float(roc_auc_score(y,p)),"class_prevalence":float(y.mean())}

def controls(rows,seed=42):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    parts={p:[r for r in rows if r["partition"]==p] for p in ("train","development","test")}
    if min(map(len,parts.values()))<2:raise ValueError("Dataset splits incomplete")
    train,development,test=[parts[p] for p in ("train","development","test")]
    y=np.array([r["label"] for r in train],dtype=int)
    y_dev=np.array([r["label"] for r in development],dtype=int)
    y_test=np.array([r["label"] for r in test],dtype=int)
    if any(set(a)!={0,1} for a in (y,y_dev,y_test)):
        raise ValueError("Both classes required in every split")
    getlength=lambda part:np.asarray([[np.log1p(len(r["text"])),
                                      np.log1p(len(r["text"].split()))] for r in part])
    length=make_pipeline(StandardScaler(),LogisticRegression(C=1.0,max_iter=1000,random_state=seed))
    length.fit(getlength(train),y)
    lexical=make_pipeline(TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True,min_df=2,
                                          max_features=25000),
                          LogisticRegression(C=1.0,max_iter=1000,random_state=seed))
    lexical.fit([r["text"] for r in train],y)
    result={}
    for name,model,represent in (("length",length,getlength),
                                  ("lexical_tfidf",lexical,lambda part:[r["text"] for r in part])):
        result[name]={
            "development":evaluate_binary(y_dev,model.decision_function(represent(development))),
            "test":evaluate_binary(y_test,model.decision_function(represent(test)))}
    result["constant_baseline"]={"balanced_accuracy":0.5,"roc_auc":0.5}
    return result

def run(cfg,family):
    result={}
    for task,features in (("concept",[None]),("linguistic",FEATURES)):
        for feature in features:
            # Lexical controls are estimated on base-corpus text; no GPU needed.
            rows=probe_dataset(cfg,task,feature=feature)
            result[f"{task}:{feature or 'all'}"]=controls(rows,int(cfg["seed"]))
    out=run_dir(cfg,family,"linear-probing")/"classifier_control_audit.json"
    write_json(out,{"family":family,"revision":2,"controls":result,
                    "interpretation":"High activation-classifier AUC must be compared with lexical and length baselines; concept detection is not welfare endorsement."})
    return out

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",choices=("gemma2","qwen3","llama31","mistral"),required=True)
    p.add_argument("--config",default="config/pipeline.yaml")
    args=p.parse_args()
    print(run(load_config(args.config),args.model))
if __name__=="__main__":main()
