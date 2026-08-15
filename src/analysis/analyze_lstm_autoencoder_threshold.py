"""Stage 12 LSTM validation-only analysis and P95 threshold freezing."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from src.analysis import analyze_autoencoder_threshold as shared
from src.models.lstm_autoencoder import LSTMAutoencoder, architecture_config, count_trainable_parameters

ROOT=Path(__file__).resolve().parents[2]
MODEL_DIR=ROOT/'models'/'lstm_autoencoder'
RESULTS=ROOT/'results'/'lstm_autoencoder'/'stride_30'
OUT=RESULTS/'threshold_analysis'
CKPT=MODEL_DIR/'lstm_ae_stride30_baseline.pt'
CONFIG=MODEL_DIR/'lstm_ae_stride30_baseline_config.json'
THRESHOLD=MODEL_DIR/'lstm_ae_stride30_threshold.json'
HISTORY=RESULTS/'training_history.csv'
SUMMARY=RESULTS/'training_summary.json'
ERRORS=RESULTS/'val_reconstruction_errors.parquet'
TRAINING=(CKPT,CONFIG,HISTORY,SUMMARY,RESULTS/'training_curve.png',ERRORS,RESULTS/'validation_error_distribution.png',ROOT/'src/models/lstm_autoencoder.py',ROOT/'src/training/train_lstm_autoencoder.py')

def sha(path: Path)->str:
    d=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): d.update(b)
    return d.hexdigest()
def load(path: Path)->dict:
    value=json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value,dict): raise TypeError(f'Expected JSON object: {path}')
    return value
def hashes()->dict[str,str]:
    missing=[str(x) for x in TRAINING if not x.is_file()]
    if missing: raise FileNotFoundError(missing)
    return {x.relative_to(ROOT).as_posix():sha(x) for x in TRAINING}
def verify()->dict:
    cfg,summary=load(CONFIG),load(SUMMARY)
    history=pd.read_csv(HISTORY)
    checkpoint=torch.load(CKPT,map_location='cpu',weights_only=True)
    model=LSTMAutoencoder(); model.load_state_dict(checkpoint['model_state_dict'],strict=True)
    if cfg['model']!=architecture_config() or checkpoint['architecture_configuration']!=architecture_config(): raise ValueError('LSTM architecture mismatch')
    if count_trainable_parameters(model)!=3177 or checkpoint['input_size']!=1 or checkpoint['sequence_length']!=30 or checkpoint['latent_dim']!=8: raise ValueError('LSTM shape/parameter mismatch')
    if checkpoint['window_length']!=30 or checkpoint['stride']!=30: raise ValueError('LSTM window configuration mismatch')
    if cfg['training_population']!='usable stride-30 train windows with FLAG == 0 only' or cfg['early_stopping_population']!='usable stride-30 validation windows with FLAG == 0 only': raise ValueError('LSTM normal-only policy mismatch')
    if cfg['test_set_policy']!='not loaded or used during LSTM training' or summary['test_set_used'] or summary['anomaly_threshold_selected']: raise ValueError('LSTM training boundary violated')
    if list(history.columns)!=['epoch','train_mse','val_normal_mse'] or len(history)!=50 or int(summary['epochs_executed'])!=50: raise ValueError('LSTM history mismatch')
    row=history.loc[history.epoch.eq(int(checkpoint['best_epoch']))]
    if len(row)!=1 or not np.isclose(float(row.iloc[0].val_normal_mse),float(checkpoint['best_validation_loss']),rtol=0,atol=1e-12): raise ValueError('Best checkpoint mismatch')
    if int(checkpoint['best_epoch'])!=50 or bool(summary['early_stopping_occurred']): raise ValueError('Unexpected Stage 11 epoch policy outcome')
    return {'device':summary['device'],'training_normal_windows':int(summary['training_normal_window_count']),'validation_normal_windows':int(summary['validation_normal_window_count']),'validation_theft_windows':int(summary['validation_theft_window_count']),'epochs_executed':50,'best_epoch':50,'best_validation_mse':float(checkpoint['best_validation_loss']),'final_train_mse':float(history.iloc[-1].train_mse),'runtime_seconds':float(summary['runtime_seconds']),'trainable_parameters':3177}
def errors()->pd.DataFrame:
    frame=pd.read_parquet(ERRORS,engine='pyarrow'); expected=['window_id','CONS_NO','FLAG','reconstruction_mse','missing_fraction']
    if list(frame.columns)!=expected or not frame.window_id.is_unique or set(frame.FLAG.unique())!={0,1}: raise ValueError('LSTM validation error schema mismatch')
    values=frame.reconstruction_mse.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values<0): raise ValueError('Invalid LSTM validation errors')
    return frame
def report(v,d,m,s,auc,c,miss)->str:
    n,t=d['FLAG_0_normal'],d['FLAG_1_theft_labelled']; ex=d['theft_exceeding_normal_percentiles']
    return f'''# Stage 12 — LSTM Validation Threshold Analysis

The manually trained LSTM uses `(B,30,1)`, latent dimension 8, and 3,177 parameters. It executed the predefined 50 epochs; epoch 50 retained the best checkpoint at normal-validation MSE `{v['best_validation_mse']:.12f}`. The maximum was not extended. Test data was not accessed.

| Population | Count | Mean | Median | Std | Min | P25 | P50 | P75 | P90 | P95 | P99 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Normal | {n['count']:,} | {n['mean']:.8f} | {n['median']:.8f} | {n['standard_deviation']:.8f} | {n['minimum']:.8f} | {n['p25']:.8f} | {n['p50']:.8f} | {n['p75']:.8f} | {n['p90']:.8f} | {n['p95']:.8f} | {n['p99']:.8f} | {n['maximum']:.8f} |
| Theft-labelled | {t['count']:,} | {t['mean']:.8f} | {t['median']:.8f} | {t['standard_deviation']:.8f} | {t['minimum']:.8f} | {t['p25']:.8f} | {t['p50']:.8f} | {t['p75']:.8f} | {t['p90']:.8f} | {t['p95']:.8f} | {t['p99']:.8f} | {t['maximum']:.8f} |

Theft/normal median and mean ratios: `{d['theft_to_normal_median_ratio']:.4f}` / `{d['theft_to_normal_mean_ratio']:.4f}`. Theft-labelled windows above normal P90/P95/P97.5/P99: `{100*ex['normal_p90']['theft_proportion_exceeding']:.2f}%` / `{100*ex['normal_p95']['theft_proportion_exceeding']:.2f}%` / `{100*ex['normal_p97.5']['theft_proportion_exceeding']:.2f}%` / `{100*ex['normal_p99']['theft_proportion_exceeding']:.2f}%`.

Validation ROC-AUC `{auc['ROC_AUC']:.6f}`; PR-AUC `{auc['PR_AUC']:.6f}`. `FLAG` is customer-level weak supervision.

## Candidate thresholds

{shared.format_metrics_table(m)}

The frozen threshold is normal-validation P95: `{s['threshold']:.12g}`. Best-F1 and best-balanced-accuracy candidates are diagnostic only.
'''
def main()->None:
    before=hashes(); v=verify(); frame=errors()
    if len(frame)!=v['validation_normal_windows']+v['validation_theft_windows']: raise ValueError('Validation count mismatch')
    d=shared.summarize_error_distributions(frame); p=shared.evaluate_percentile_thresholds(frame); sup=shared.evaluate_supervised_thresholds(frame); metrics=pd.concat([p,sup],ignore_index=True); selected,rationale=shared.select_baseline_threshold(metrics)
    labels=frame.FLAG.to_numpy(dtype=np.int8); scores=frame.reconstruction_mse.to_numpy(dtype=np.float64); auc={'ROC_AUC':float(roc_auc_score(labels,scores)),'PR_AUC':float(average_precision_score(labels,scores)),'theft_label_prevalence':float(np.mean(labels))}
    customers,customer=shared.analyze_customer_level_scores(frame); missing,miss=shared.analyze_missingness_relationship(frame)
    sm={k:(int(selected[k]) if k in {'TP','TN','FP','FN','predicted_anomaly_count'} else float(selected[k])) for k in ('threshold','TP','TN','FP','FN','precision','recall','F1','specificity','false_positive_rate','balanced_accuracy','accuracy','predicted_anomaly_count','predicted_anomaly_percentage')}
    artifact={'stage':'Stage 12 — LSTM Validation-Based Anomaly Threshold','model_checkpoint':CKPT.relative_to(ROOT).as_posix(),'stride':30,'window_length':30,'threshold_value':sm['threshold'],'threshold_selection_method':'normal_validation_95th_percentile','anomaly_rule':'reconstruction_mse > threshold_value','validation_metrics':sm,'validation_sample_counts':{'normal_windows':int(frame.FLAG.eq(0).sum()),'theft_labelled_windows':int(frame.FLAG.eq(1).sum()),'total_windows':len(frame)},'model_context':{'architecture':architecture_config(),'trainable_parameters':3177,'latent_dimension':8,'loss':'MSELoss','optimizer':'Adam','learning_rate':0.001,'best_epoch':50,'best_validation_mse':v['best_validation_mse']},'selection_rationale':rationale,'weak_supervision_note':'FLAG is customer-level; positive windows are not confirmed theft events.','test_data_accessed':False,'lstm_training_input_sha256':before}
    OUT.mkdir(parents=True,exist_ok=True); shared.save_json(artifact,THRESHOLD); shared.save_csv(metrics,OUT/'threshold_metrics.csv'); shared.save_csv(customers,OUT/'customer_level_diagnostics.csv'); shared.save_csv(missing,OUT/'missingness_group_summary.csv'); shared.save_json({'stage':'Stage 12 — LSTM Validation Threshold','training_verification':v,'validation_error_distributions':d,'validation_ranking_metrics':auc,'candidate_thresholds':metrics.replace({np.nan:None}).to_dict('records'),'selected_threshold':artifact,'customer_level_diagnostic':customer,'missingness_diagnostic':miss,'test_data_accessed':False,'lstm_training_artifacts_unchanged':True},OUT/'threshold_summary.json'); shared.save_text(report(v,d,metrics,selected,auc,customer,miss),OUT/'threshold_report.md')
    shared.ANALYSIS_DIR=OUT; shared.PR_CURVE_PATH=OUT/'precision_recall_curve.png'; shared.TRADEOFF_PATH=OUT/'threshold_tradeoff.png'; shared.save_precision_recall_curve(frame,metrics,selected); shared.save_threshold_tradeoff(frame,selected)
    if before!=hashes(): raise RuntimeError('LSTM training artifacts changed')
    print(f"Validation windows: {len(frame):,}"); print(f"ROC-AUC: {auc['ROC_AUC']:.6f}; PR-AUC: {auc['PR_AUC']:.6f}"); print(f"Frozen P95 threshold: {sm['threshold']:.12g}"); print('Test data accessed: False')
if __name__=='__main__': main()
