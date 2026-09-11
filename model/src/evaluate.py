from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


class ModelEvaluator:
    @staticmethod
    def evaluate_all(y_true, y_pred_prob, threshold=0.5):
        y_true = np.asarray(y_true).astype(int)
        y_pred = (y_pred_prob >= threshold).astype(int)
        unique_classes = np.unique(y_true)

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fdr = fp / (tp + fp) if (tp + fp) > 0 else 0.0

        metrics = {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Log-Loss (Cross-Entropy)": log_loss(y_true, y_pred_prob, labels=[0, 1]),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall (Detection Rate / TPR)": recall_score(y_true, y_pred, zero_division=0),
            "F1-Score": f1_score(y_true, y_pred, zero_division=0),
            "False Positive Rate (FPR)": fpr,
            "False Discovery Rate (FDR)": fdr,
            "True Positives (TP)": int(tp),
            "True Negatives (TN)": int(tn),
            "False Positives (FP)": int(fp),
            "False Negatives (FN)": int(fn),
        }

        if len(unique_classes) > 1:
            metrics["ROC-AUC"] = roc_auc_score(y_true, y_pred_prob)
        else:
            metrics["ROC-AUC"] = None

        print("\n" + "=" * 45)
        print("         FULL TEST DIAGNOSTIC REPORT         ")
        print("=" * 45)
        for metric_name, value in metrics.items():
            if isinstance(value, float):
                print(f"{metric_name:<30}: {value:.6f}")
            else:
                print(f"{metric_name:<30}: {value}")
        print("=" * 45 + "\n")

        return metrics, y_pred

    @staticmethod
    def plot_all_diagnostics(y_true, y_pred_prob, output_dir="reports", suffix=""):
        """Generates and exports Confusion Matrix, ROC Curve, and PR Curve."""
        y_true = np.asarray(y_true).astype(int)
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # 1. Confusion Matrix Plot
        y_pred = (y_pred_prob >= 0.5).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

        plt.figure(figsize=(6, 5))
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix {suffix}")
        plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ["Benign (0)", "Malware (1)"])
        plt.yticks(tick_marks, ["Benign (0)", "Malware (1)"])

        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(
                    j,
                    i,
                    format(cm[i, j], "d"),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black",
                )

        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        cm_file = out_path / f"confusion_matrix{suffix}.png"
        plt.savefig(cm_file, dpi=300)
        plt.close()
        print(f"Saved: {cm_file}")

        # 2. ROC and Precision-Recall Curves (requires both classes)
        if len(np.unique(y_true)) > 1:
            # ROC Curve
            fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
            auc_score = roc_auc_score(y_true, y_pred_prob)

            plt.figure(figsize=(6, 5))
            plt.plot(fpr, tpr, label=f"ROC (AUC = {auc_score:.4f})", color="darkorange", lw=2)
            plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate (FPR)")
            plt.ylabel("True Positive Rate (TPR)")
            plt.title(f"ROC Curve {suffix}")
            plt.legend(loc="lower right")
            plt.tight_layout()
            roc_file = out_path / f"roc_curve{suffix}.png"
            plt.savefig(roc_file, dpi=300)
            plt.close()
            print(f"Saved: {roc_file}")

            # Precision-Recall Curve
            prec, rec, _ = precision_recall_curve(y_true, y_pred_prob)
            plt.figure(figsize=(6, 5))
            plt.plot(rec, prec, color="green", lw=2, label="PR Curve")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title(f"Precision-Recall Curve {suffix}")
            plt.legend(loc="lower left")
            plt.tight_layout()
            pr_file = out_path / f"pr_curve{suffix}.png"
            plt.savefig(pr_file, dpi=300)
            plt.close()
            print(f"Saved: {pr_file}")