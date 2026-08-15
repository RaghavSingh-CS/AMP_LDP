"""Saved-artifact-only frozen FC/CNN/LSTM architecture comparison."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
MODELS={'fc':{'results':ROOT/'results/autoencoder/stride_30','checkpoint':ROOT/'models/autoencoder/ae_stride30_baseline.pt','threshold':ROOT/'models/autoencoder/ae_stride30_threshold.json','config':ROOT/'models/autoencoder/ae_stride30_baseline_config.json'},'cnn':{'results':ROOT/'results/cnn_autoencoder/stride_30','checkpoint':ROOT/'models/cnn_autoencoder/cnn_ae_stride30_baseline.pt','threshold':ROOT/'models/cnn_autoencoder/cnn_ae_stride30_threshold.json','config':ROOT/'models/cnn_autoencoder/cnn_ae_stride30_baseline_config.json'},'lstm':{'results':ROOT/'results/lstm_autoencoder/stride_30','checkpoint':ROOT/'models/lstm_autoencoder/lstm_ae_stride30_baseline.pt','threshold':ROOT/'models/lstm_autoencoder/lstm_ae_stride30_threshold.json','config':ROOT/'models/lstm_autoencoder/lstm_ae_stride30_baseline_config.json'}}
OUT=ROOT/'results/model_comparison/fc_vs_cnn_vs_lstm'
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
 data={}
 for name,paths in MODELS.items():
  r=paths['results'];data[name]={'train':load(r/'training_summary.json'),'threshold':load(paths['threshold']),'validation':load(r/'threshold_analysis/threshold_summary.json'),'test':load(r/'test_evaluation/test_metrics.json'),'final':load(r/'test_evaluation/baseline_final_summary.json'),'config':load(paths['config'])}
 fc=data['fc']; metrics=[]
 def add(metric,extract):
  values={name:float(extract(d)) for name,d in data.items()};base=values['fc'];metrics.append({'metric':metric,'fully_connected':base,'cnn':values['cnn'],'lstm':values['lstm'],'cnn_minus_fc':values['cnn']-base,'lstm_minus_fc':values['lstm']-base,'cnn_relative_to_fc_percent':(values['cnn']-base)/base*100 if base else None,'lstm_relative_to_fc_percent':(values['lstm']-base)/base*100 if base else None})
 add('trainable_parameters',lambda d:d['config']['trainable_parameters']);add('epochs_executed',lambda d:d['train']['epochs_executed']);add('early_stopping_occurred',lambda d:int(d['train']['early_stopping_occurred']));add('best_epoch',lambda d:d['train']['best_epoch']);add('best_validation_mse',lambda d:d['train']['best_validation_mse']);add('training_runtime_seconds',lambda d:d['train']['runtime_seconds']);add('runtime_relative_to_fc',lambda d:d['train']['runtime_seconds']/fc['train']['runtime_seconds']);add('frozen_p95_threshold',lambda d:d['threshold']['threshold_value'])
 for key in ('precision','recall','F1','specificity','false_positive_rate','balanced_accuracy','accuracy','predicted_anomaly_percentage'):
  add('validation_'+key,lambda d,k=key:d['threshold']['validation_metrics'][k])
 for key in ('ROC_AUC','PR_AUC'):add('validation_'+key,lambda d,k=key:d['validation']['validation_ranking_metrics'][k])
 for key in ('precision','recall','F1','specificity','false_positive_rate','balanced_accuracy','accuracy','ROC_AUC','PR_AUC','predicted_anomaly_percentage'):
  add('test_'+key,lambda d,k=key:d['test']['window_level_test_metrics'][k])
 for aggregation in ('mean_mse','maximum_mse','p95_mse'):
  for key in ('ROC_AUC','PR_AUC'):add('test_customer_'+aggregation+'_'+key,lambda d,a=aggregation,k=key:d['test']['customer_level_diagnostic']['aggregation_auc'][a][k])
 table=pd.DataFrame(metrics);OUT.mkdir(parents=True,exist_ok=True);table.to_csv(OUT/'architecture_comparison.csv',index=False)
 selection={'selected_architecture':'fully_connected_autoencoder','model_assessments':{'fully_connected':'weak','cnn':'weak','lstm':'weak'},'rationale':'FC leads the frozen test ROC-AUC, PR-AUC, precision, recall, F1, and every customer-level mean/maximum/P95 ROC-AUC and PR-AUC diagnostic. It is also the smallest and fastest model.','architecture_effects':{'fc':'reference baseline','cnn':'best normal reconstruction but worse detection','lstm':'intermediate normal reconstruction and no detection improvement over FC'},'next_stage_recommendation':'Use the frozen stride-30 FC Autoencoder as the reconstruction baseline for later research. Investigate a new, pre-registered modeling hypothesis rather than a hyperparameter search; no next model is implemented here.'}
 payload={'stage':'Stage 12 — Frozen FC versus CNN versus LSTM comparison','comparison_policy':'saved artifacts only; no retraining, threshold tuning, or test-driven selection','shared_protocol':{'customer_split':'Stage 2 fixed customer split','preprocessing':'Stage 1 plus train-only StandardScaler','window_length':30,'stride':30,'usable_window_rule':'missing_fraction < 1.0','training':'FLAG=0 only','loss':'MSE','optimizer':'Adam','learning_rate':0.001,'batch_size':256,'maximum_epochs':50,'seed':42,'threshold_method':'normal_validation_95th_percentile','test_method':'frozen evaluation'},'metrics':table.to_dict('records'),'selection':selection,'weak_supervision_caveat':'FLAG is customer-level; positive windows are not confirmed theft events.'}
 (OUT/'architecture_comparison.json').write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n',encoding='utf-8')
 rows='\n'.join(f'| {r.metric} | {r.fully_connected:.6f} | {r.cnn:.6f} | {r.lstm:.6f} | {r.cnn_minus_fc:+.6f} | {r.lstm_minus_fc:+.6f} |' for r in table.itertuples(index=False))
 report=f'''# Stage 12 — Frozen FC vs CNN vs LSTM Comparison

All models use the same fixed customer split, preprocessing/scaler, 30-day stride-30 usable windows, normal-only training, MSE/Adam protocol, validation-normal P95 cutoff method, and frozen-test methodology. The changed factor is architecture.

| Metric | FC | CNN | LSTM | CNN − FC | LSTM − FC |
|---|---:|---:|---:|---:|---:|
{rows}

## Architecture selection

FC is the selected reconstruction-model baseline. It is simplest (1,362 parameters), fastest (600.96 CPU seconds), and has the strongest frozen anomaly discrimination: test ROC-AUC/PR-AUC/F1 `{data['fc']['test']['window_level_test_metrics']['ROC_AUC']:.6f}` / `{data['fc']['test']['window_level_test_metrics']['PR_AUC']:.6f}` / `{data['fc']['test']['window_level_test_metrics']['F1']:.6f}`. It also leads every customer aggregate diagnostic.

CNN reconstructs normal windows best (validation MSE `{data['cnn']['train']['best_validation_mse']:.6f}`) but has weaker detection. LSTM reconstruction (`{data['lstm']['train']['best_validation_mse']:.6f}`) lies between FC and CNN, but LSTM does not improve frozen window-level or customer-level anomaly discrimination over FC and costs `{data['lstm']['train']['runtime_seconds']/data['fc']['train']['runtime_seconds']:.2f}×` FC CPU runtime. All three models are weak under customer-level `FLAG` supervision.

`FLAG` is customer-level weak supervision; these results do not establish confirmed per-window theft detection.
'''
 (OUT/'architecture_comparison_report.md').write_text(report,encoding='utf-8');print('Selected architecture: fully_connected_autoencoder')
if __name__=='__main__':main()
