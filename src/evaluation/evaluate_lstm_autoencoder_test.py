"""One-time frozen test evaluation for the Stage 11 LSTM Autoencoder."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from src.data.window_dataset import ProcessedSplitData, WindowDataset
from src.evaluation import evaluate_autoencoder_test as shared
from src.models.lstm_autoencoder import LSTMAutoencoder, architecture_config, count_trainable_parameters

ROOT=Path(__file__).resolve().parents[2]; MODEL_DIR=ROOT/'models'/'lstm_autoencoder'; RESULTS=ROOT/'results'/'lstm_autoencoder'/'stride_30'; OUT=RESULTS/'test_evaluation'
CKPT=MODEL_DIR/'lstm_ae_stride30_baseline.pt'; THRESHOLD=MODEL_DIR/'lstm_ae_stride30_threshold.json'; FILTER=ROOT/'data/windows/stride_30/usable/filter_summary.json'; MANIFEST=FILTER.parent/'test_windows.parquet'; THRESHOLD_SUMMARY=RESULTS/'threshold_analysis/threshold_summary.json'
ERRORS=OUT/'test_reconstruction_errors.parquet'; METRICS=OUT/'test_metrics.json'; REPORT=OUT/'test_report.md'; CONFUSION=OUT/'confusion_matrix.png'; DIST=OUT/'test_error_distribution.png'; COMPARISON=OUT/'validation_vs_test.csv'; CUSTOMER=OUT/'customer_level_test_diagnostics.csv'; FRACTIONS=OUT/'customer_anomaly_fraction_summary.csv'; MISSING=OUT/'missingness_test_summary.csv'; TOP=OUT/'top_test_error_cases.csv'; FINAL=OUT/'baseline_final_summary.json'
THRESHOLD_OUTPUTS=(THRESHOLD,RESULTS/'threshold_analysis/threshold_metrics.csv',THRESHOLD_SUMMARY,RESULTS/'threshold_analysis/threshold_report.md',RESULTS/'threshold_analysis/precision_recall_curve.png',RESULTS/'threshold_analysis/threshold_tradeoff.png',RESULTS/'threshold_analysis/customer_level_diagnostics.csv',RESULTS/'threshold_analysis/missingness_group_summary.csv')

def sha(p:Path)->str:
 d=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):d.update(b)
 return d.hexdigest()
def load(p:Path)->dict:
 x=json.loads(p.read_text(encoding='utf-8'))
 if not isinstance(x,dict):raise TypeError(p)
 return x
def protected()->dict[str,str]:
 t=load(THRESHOLD); f=load(FILTER); paths=[ROOT/x for x in t['lstm_training_input_sha256']]; paths.extend(THRESHOLD_OUTPUTS); paths.extend(ROOT/x for x in f['integrity']['protected_sha256']); paths.extend(FILTER.parent/f'{s}_windows.parquet' for s in ('train','val','test')); paths.extend((FILTER,MANIFEST)); unique=tuple(dict.fromkeys(p.resolve() for p in paths)); missing=[str(p) for p in unique if not p.is_file()]
 if missing:raise FileNotFoundError(missing)
 return {p.relative_to(ROOT).as_posix():sha(p) for p in unique}
def frozen(device):
 t=load(THRESHOLD); value=float(t['threshold_value'])
 if not np.isfinite(value) or value<0 or t['threshold_selection_method']!='normal_validation_95th_percentile' or t['anomaly_rule']!='reconstruction_mse > threshold_value' or t['test_data_accessed'] is not False:raise ValueError('Invalid frozen LSTM threshold')
 if Path(t['model_checkpoint'])!=Path('models/lstm_autoencoder/lstm_ae_stride30_baseline.pt'):raise ValueError('Wrong checkpoint reference')
 c=torch.load(CKPT,map_location=device,weights_only=True)
 if c['architecture_configuration']!=architecture_config() or c['window_length']!=30 or c['stride']!=30 or c['latent_dim']!=8:raise ValueError('Invalid LSTM checkpoint')
 m=LSTMAutoencoder().to(device);m.load_state_dict(c['model_state_dict'],strict=True);m.eval()
 if count_trainable_parameters(m)!=3177:raise ValueError('Parameter mismatch')
 return m,c,value,t
def datasets():
 store=ProcessedSplitData.load('test'); normal=WindowDataset('test',0,stride=30,store=store,include_metadata=True); theft=WindowDataset('test',1,stride=30,store=store,include_metadata=True); expected=int(load(FILTER)['splits']['test']['retained_windows'])
 if len(normal)+len(theft)!=expected:raise ValueError('Test window count mismatch')
 return normal,theft
@torch.inference_mode()
def score(model,dataset,batch,device):
 values=[]
 for flat in DataLoader(dataset,batch_size=batch,shuffle=False,num_workers=0):
  inputs=flat.to(device=device,dtype=torch.float32).unsqueeze(-1); out=model(inputs); mse=torch.mean((out-inputs)**2,dim=(1,2))
  if not torch.isfinite(mse).all() or torch.any(mse<0):raise FloatingPointError('Invalid test MSE')
  values.append(mse.cpu().numpy())
 scores=np.concatenate(values).astype(np.float32,copy=False); meta=dataset.metadata
 if len(scores)!=len(meta):raise RuntimeError('Score metadata misalignment')
 meta['reconstruction_mse']=scores;return meta
def compute(model,normal,theft,batch,device):
 frame=pd.concat([score(model,normal,batch,device),score(model,theft,batch,device)],ignore_index=True); dates=pd.read_parquet(MANIFEST,columns=['window_id','start_date','end_date']); frame=frame.merge(dates,on='window_id',how='left',validate='one_to_one')
 if frame[['start_date','end_date']].isna().any().any():raise ValueError('Date alignment failure')
 return frame
def configure():
 shared.CONFUSION_PATH=CONFUSION;shared.ERROR_DISTRIBUTION_PATH=DIST;shared.THRESHOLD_PATH=THRESHOLD;shared.VALIDATION_THRESHOLD_SUMMARY_PATH=THRESHOLD_SUMMARY
def report(threshold,metrics,comparison,generalization,customer,missingness):
 rows='\n'.join(f'| {r.metric} | {r.validation:.6f} | {r.test:.6f} | {r.absolute_difference:.6f} |' for r in comparison.itertuples(index=False))
 return f'''# Stage 12 — Frozen LSTM Test Evaluation

The fixed normal-validation P95 threshold `{threshold:.12g}` was applied once using `reconstruction_mse > threshold`. No retraining, threshold adjustment, or test-driven selection occurred.

| Metric | Value |
|---|---:|
| TP / TN / FP / FN | {metrics['TP']:,} / {metrics['TN']:,} / {metrics['FP']:,} / {metrics['FN']:,} |
| Precision | {metrics['precision']:.6f} |
| Recall | {metrics['recall']:.6f} |
| F1 | {metrics['F1']:.6f} |
| Specificity | {metrics['specificity']:.6f} |
| FPR | {metrics['false_positive_rate']:.6f} |
| ROC-AUC / PR-AUC | {metrics['ROC_AUC']:.6f} / {metrics['PR_AUC']:.6f} |

Validation-to-test classification: **{generalization}**.

| Metric | Validation | Test | Absolute difference |
|---|---:|---:|---:|
{rows}

Customer P95 ROC-AUC/PR-AUC: `{customer['aggregation_auc']['p95_mse']['ROC_AUC']:.6f}` / `{customer['aggregation_auc']['p95_mse']['PR_AUC']:.6f}`. Overall MSE/missing-fraction Spearman: `{missingness['overall']['spearman']:.6f}`. `FLAG` is customer-level weak supervision, not confirmed per-window theft.
'''
def save(frame,threshold,t,c,before):
 configure(); frame=shared.apply_threshold(frame,threshold); metrics=shared.calculate_window_metrics(frame,threshold); dist=shared.summarize_error_distributions(frame,threshold); customer_table,fractions,customer=shared.calculate_customer_diagnostics(frame); missing_table,missing=shared.analyze_missingness(frame); comparison,general=shared.compare_validation_test(metrics); cases=shared.top_error_cases(frame)
 shared.save_parquet(frame,ERRORS);shared.save_csv(comparison,COMPARISON);shared.save_csv(customer_table,CUSTOMER);shared.save_csv(fractions,FRACTIONS);shared.save_csv(missing_table,MISSING);shared.save_csv(cases,TOP);shared.save_confusion_matrix(metrics);shared.save_error_distribution(frame,threshold)
 payload={'stage':'Stage 12 — Frozen LSTM Test Evaluation','frozen_threshold':threshold,'threshold_source':THRESHOLD.relative_to(ROOT).as_posix(),'anomaly_rule':'reconstruction_mse > frozen_threshold','window_level_test_metrics':metrics,'error_distributions':dist,'validation_vs_test':{'classification':general,'metrics':comparison.to_dict('records')},'customer_level_diagnostic':customer,'missingness_diagnostic':missing,'weak_supervision_note':'FLAG is customer-level; positive windows are not confirmed theft events.','retraining_performed':False,'threshold_modified':False,'test_driven_tuning_performed':False,'protected_pretest_sha256':before}
 shared.save_json(payload,METRICS)
 final={'artifact_role':'final stride-30 LSTM Autoencoder performance record','model':{'architecture':architecture_config(),'trainable_parameters':3177,'checkpoint':CKPT.relative_to(ROOT).as_posix(),'checkpoint_sha256':before[CKPT.relative_to(ROOT).as_posix()],'best_epoch':c['best_epoch'],'best_validation_mse':c['best_validation_loss']},'threshold':{'value':threshold,'selection_method':t['threshold_selection_method'],'artifact':THRESHOLD.relative_to(ROOT).as_posix(),'artifact_sha256':before[THRESHOLD.relative_to(ROOT).as_posix()],'rule':'reconstruction_mse > threshold'},'test_sample_counts':{'normal_windows':int(frame.FLAG.eq(0).sum()),'theft_labelled_windows':int(frame.FLAG.eq(1).sum()),'total_windows':len(frame)},'test_metrics':{k:metrics[k] for k in ('TP','TN','FP','FN','precision','recall','F1','specificity','false_positive_rate','balanced_accuracy','accuracy','ROC_AUC','PR_AUC','predicted_anomaly_percentage')},'validation_test_generalization':general,'weak_supervision_caveat':'FLAG is customer-level; window metrics are not confirmed theft-event metrics.','frozen_evaluation_guarantees':{'checkpoint_unchanged':True,'threshold_unchanged':True,'retraining_performed':False,'test_driven_tuning_performed':False}}
 shared.save_json(final,FINAL);shared.save_text(report(threshold,metrics,comparison,general,customer,missing),REPORT);return payload
def main():
 parser=argparse.ArgumentParser(description='Evaluate frozen LSTM on test once.');parser.add_argument('--batch-size',type=int,default=256);args=parser.parse_args()
 if args.batch_size<=0:raise ValueError('batch-size must be positive')
 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');before=protected();model,checkpoint,threshold,artifact=frozen(device);normal,theft=datasets();frame=compute(model,normal,theft,args.batch_size,device);payload=save(frame,threshold,artifact,checkpoint,before)
 if before!=protected():raise RuntimeError('Protected pre-test artifact changed')
 m=payload['window_level_test_metrics'];print(f"LSTM test windows: {len(frame):,}");print(f"ROC-AUC: {m['ROC_AUC']:.6f}; PR-AUC: {m['PR_AUC']:.6f}; F1: {m['F1']:.6f}");print('Checkpoint and threshold unchanged: True')
if __name__=='__main__':main()
