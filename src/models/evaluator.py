"""
Model Evaluation, Accuracy, and FPR / FNR Benchmarking Engine
Computes comprehensive metrics: Accuracy, Precision, Recall, F1-Score,
False Positive Rate (FPR), False Negative Rate (FNR), Confusion Matrices,
and Threshold Trade-Off Curves for all AML detection models.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)

class ModelEvaluator:
    def __init__(self, detector_suite):
        self.detector = detector_suite

    def compute_single_model_metrics(self, y_true, y_pred, y_proba=None, model_name="Model"):
        """
        Computes all standard classification metrics with explicit focus on
        Accuracy, False Positive Rate (FPR), and False Negative Rate (FNR).
        """
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        
        total = tn + fp + fn + tp
        accuracy = (tp + tn) / total if total > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Explicit FPR and FNR
        # FPR: Fraction of legitimate transactions falsely flagged (Analyst Alert Fatigue)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        # FNR: Fraction of illicit money laundering transactions missed (Regulatory Exposure Risk)
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        
        auc = 0.5
        if y_proba is not None and len(np.unique(y_true)) > 1:
            try:
                auc = float(roc_auc_score(y_true, y_proba))
            except Exception:
                auc = 0.5
                
        return {
            'model_name': model_name,
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'specificity': float(specificity),
            'f1_score': float(f1),
            'false_positive_rate': float(fpr),
            'false_negative_rate': float(fnr),
            'roc_auc': float(auc),
            'confusion_matrix': {
                'tn': int(tn),
                'fp': int(fp),
                'fn': int(fn),
                'tp': int(tp)
            },
            'sample_size': int(total)
        }

    def evaluate_all_models(self, test_df_with_features, ground_truth_col='is_sar_reported', threshold=0.45):
        """
        Runs comprehensive benchmark comparing:
        1. Rule-Based Compliance Engine
        2. Supervised Random Forest Classifier
        3. Unsupervised Isolation Forest
        4. Calibrated Hybrid Ensemble
        """
        y_true = test_df_with_features[ground_truth_col].astype(int).values
        
        # 1. Rule Engine
        rule_preds, rule_proba, _ = self.detector.predict_rule_based(test_df_with_features)
        metrics_rule = self.compute_single_model_metrics(
            y_true, rule_preds, rule_proba, model_name="1. Rule-Based Compliance Engine"
        )
        
        # 2. Random Forest
        rf_preds, rf_proba = self.detector.predict_random_forest(test_df_with_features, threshold=threshold)
        metrics_rf = self.compute_single_model_metrics(
            y_true, rf_preds, rf_proba, model_name="2. Supervised Random Forest"
        )
        
        # 3. Isolation Forest
        iso_preds, iso_proba = self.detector.predict_isolation_forest(test_df_with_features, threshold=threshold)
        metrics_iso = self.compute_single_model_metrics(
            y_true, iso_preds, iso_proba, model_name="3. Unsupervised Isolation Forest"
        )
        
        # 4. Calibrated Hybrid Ensemble
        ens_preds, ens_proba, _, _ = self.detector.predict_hybrid_ensemble(test_df_with_features, threshold=threshold)
        metrics_ens = self.compute_single_model_metrics(
            y_true, ens_preds, ens_proba, model_name="4. Calibrated Hybrid Ensemble"
        )
        
        all_metrics = [metrics_rule, metrics_rf, metrics_iso, metrics_ens]
        
        # Summary comparison table
        summary_table = pd.DataFrame([{
            'Model': m['model_name'],
            'Accuracy': f"{m['accuracy'] * 100:.2f}%",
            'False Positive Rate (FPR)': f"{m['false_positive_rate'] * 100:.2f}%",
            'False Negative Rate (FNR)': f"{m['false_negative_rate'] * 100:.2f}%",
            'Precision': f"{m['precision'] * 100:.2f}%",
            'Recall (TPR)': f"{m['recall'] * 100:.2f}%",
            'F1-Score': f"{m['f1_score'] * 100:.2f}%",
            'ROC-AUC': f"{m['roc_auc']:.4f}",
            'True Positives (TP)': m['confusion_matrix']['tp'],
            'False Positives (FP)': m['confusion_matrix']['fp'],
            'True Negatives (TN)': m['confusion_matrix']['tn'],
            'False Negatives (FN)': m['confusion_matrix']['fn']
        } for m in all_metrics])
        
        # Generate ROC curves
        roc_data = {}
        for name, prob in [
            ("Rule Engine", rule_proba),
            ("Random Forest", rf_proba),
            ("Isolation Forest", iso_proba),
            ("Hybrid Ensemble", ens_proba)
        ]:
            if len(np.unique(y_true)) > 1:
                fpr_curve, tpr_curve, thresholds = roc_curve(y_true, prob)
                roc_data[name] = {
                    'fpr': fpr_curve.tolist(),
                    'tpr': tpr_curve.tolist(),
                    'auc': float(roc_auc_score(y_true, prob))
                }
                
        return {
            'metrics_list': all_metrics,
            'summary_table': summary_table,
            'roc_data': roc_data,
            'probas': {
                'Rule Engine': rule_proba,
                'Random Forest': rf_proba,
                'Isolation Forest': iso_proba,
                'Hybrid Ensemble': ens_proba
            },
            'y_true': y_true
        }

    def compute_threshold_tradeoff(self, test_df_with_features, model_type='ensemble', steps=19):
        """
        Sweeps decision threshold from 0.05 to 0.95 to show how shifting the bar
        trades off False Positive Rate (analyst workload) vs False Negative Rate (compliance risk).
        """
        y_true = test_df_with_features['is_sar_reported'].astype(int).values
        
        if model_type == 'ensemble':
            _, proba, _, _ = self.detector.predict_hybrid_ensemble(test_df_with_features)
        elif model_type == 'random_forest':
            _, proba = self.detector.predict_random_forest(test_df_with_features)
        elif model_type == 'rule_based':
            _, proba, _ = self.detector.predict_rule_based(test_df_with_features)
        else:
            _, proba = self.detector.predict_isolation_forest(test_df_with_features)
            
        thresholds = np.linspace(0.05, 0.95, steps)
        rows = []
        for th in thresholds:
            pred = (proba >= th).astype(int)
            cm = confusion_matrix(y_true, pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            total = tn + fp + fn + tp
            
            acc = (tp + tn) / total
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
            
            rows.append({
                'threshold': round(float(th), 2),
                'accuracy': float(acc),
                'fpr': float(fpr),
                'fnr': float(fnr),
                'precision': float(prec),
                'recall': float(rec),
                'f1': float(f1),
                'fp_count': int(fp),
                'fn_count': int(fn)
            })
            
        return pd.DataFrame(rows)
