import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
from itertools import cycle
import os
import config as cfg
import logging

logger = logging.getLogger(__name__)

def set_style():
    sns.set_context("paper", font_scale=1.4)
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['figure.dpi'] = 300

def plot_training_history(history, model_name):
    set_style()
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    
    logger.info(f"Plotting training history for {model_name}. Epochs: {len(epochs)}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.plot(epochs, acc, 'b-', label='Training Acc', linewidth=2)
    ax1.plot(epochs, val_acc, 'r--', label='Validation Acc', linewidth=2)
    ax1.set_title(f'{model_name} - Accuracy')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, loss, 'b-', label='Training Loss', linewidth=2)
    ax2.plot(epochs, val_loss, 'r--', label='Validation Loss', linewidth=2)
    ax2.set_title(f'{model_name} - Loss')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(cfg.RESULTS_PATH, f'{model_name}_training_history.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved training history plot to {save_path}")

def plot_confusion_matrix(y_true, y_pred, classes, model_name):
    set_style()
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    logger.info(f"Plotting confusion matrix for {model_name}. Data points: {len(y_true)}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes, square=True, ax=ax1)
    ax1.set_title(f'{model_name} - Confusion Matrix (Count)')
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predicted Label')
    
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='YlOrRd', 
                xticklabels=classes, yticklabels=classes, square=True, ax=ax2)
    ax2.set_title(f'{model_name} - Normalized Confusion Matrix')
    ax2.set_ylabel('True Label')
    ax2.set_xlabel('Predicted Label')
    
    plt.tight_layout()
    save_path = os.path.join(cfg.RESULTS_PATH, f'{model_name}_confusion_matrix.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved confusion matrix to {save_path}")

def plot_roc_curves(y_true, y_scores, classes, model_name):
    set_style()
    y_true_bin = label_binarize(y_true, classes=range(len(classes)))
    n_classes = len(classes)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_scores.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.figure(figsize=(12, 9))
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC (AUC = {0:0.3f})'.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple', 'brown', 'pink'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='{0} (AUC = {1:0.3f})'.format(classes[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} - ROC Curves')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(cfg.RESULTS_PATH, f'{model_name}_roc_curves.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved ROC curves to {save_path}")

def plot_model_comparison(metrics_df):
    set_style()
    
    df_melted = metrics_df.melt(id_vars='Model', value_vars=['Accuracy', 'Precision', 'Recall', 'F1'], 
                                var_name='Metric', value_name='Score')

    plt.figure(figsize=(14, 7))
    sns.barplot(x='Model', y='Score', hue='Metric', data=df_melted, palette='viridis')
    plt.title('Model Comparison - Performance Metrics', fontsize=16, fontweight='bold')
    plt.ylim(0, 1.0)
    plt.ylabel('Score', fontsize=12)
    plt.xlabel('Model', fontsize=12)
    plt.xticks(rotation=15)
    plt.legend(title='Metric', fontsize=10)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    save_path = os.path.join(cfg.RESULTS_PATH, 'model_comparison.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved model comparison plot to {save_path}")

def plot_tsne(model, x_data, y_data, classes, model_name):
    from sklearn.manifold import TSNE
    from tensorflow.keras.models import Model

    try:
        feature_extractor = Model(inputs=model.inputs, outputs=model.layers[-2].output)
        features = feature_extractor.predict(x_data, verbose=0)
        logger.info(f"Extracted features for t-SNE: {features.shape}")
        
        if features.shape[0] > 2000:
            indices = np.random.choice(features.shape[0], 2000, replace=False)
            features = features[indices]
            y_data = y_data[indices]
        
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
        tsne_results = tsne.fit_transform(features)
        
        plt.figure(figsize=(12, 9))
        scatter = plt.scatter(
            tsne_results[:, 0], tsne_results[:, 1],
            c=y_data,
            cmap='tab10',
            alpha=0.7,
            s=50,
            edgecolors='black',
            linewidth=0.5
        )
        
        cbar = plt.colorbar(scatter, ticks=range(len(classes)))
        cbar.set_label('Emotion Class', rotation=270, labelpad=20)
        cbar.ax.set_yticklabels(classes)
        
        plt.title(f'{model_name} - t-SNE Projection of Features', fontsize=16, fontweight='bold')
        plt.xlabel('t-SNE Component 1', fontsize=12)
        plt.ylabel('t-SNE Component 2', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        save_path = os.path.join(cfg.RESULTS_PATH, f'{model_name}_tsne.png')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved t-SNE plot to {save_path}")
    except Exception as e:
        logger.error(f"Could not generate t-SNE plot for {model_name}: {e}")

def plot_all_training_comparison(all_histories):
    set_style()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    
    for idx, (model_name, history) in enumerate(all_histories.items()):
        epochs = range(1, len(history['accuracy']) + 1)
        color = colors[idx % len(colors)]
        
        ax1.plot(epochs, history['val_accuracy'], label=f'{model_name}', 
                linewidth=2, color=color, alpha=0.8)
        ax2.plot(epochs, history['val_loss'], label=f'{model_name}', 
                linewidth=2, color=color, alpha=0.8)
    
    ax1.set_title('Validation Accuracy - All Models', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Validation Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_title('Validation Loss - All Models', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Validation Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(cfg.RESULTS_PATH, 'all_models_training_comparison.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved all models training comparison to {save_path}")

def plot_metrics_radar(metrics_df):
    set_style()
    
    categories = ['Accuracy', 'Precision', 'Recall', 'F1']
    num_vars = len(categories)
    
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    
    for idx, row in metrics_df.iterrows():
        values = row[categories].tolist()
        values += values[:1]
        
        color = colors[idx % len(colors)]
        ax.plot(angles, values, 'o-', linewidth=2, label=row['Model'], color=color)
        ax.fill(angles, values, alpha=0.15, color=color)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=12)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=10)
    ax.set_title('Model Performance Radar Chart', size=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)
    
    plt.tight_layout()
    save_path = os.path.join(cfg.RESULTS_PATH, 'metrics_radar_chart.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved metrics radar chart to {save_path}")

def plot_class_performance(y_true, y_pred, classes):
    set_style()
    
    from sklearn.metrics import classification_report
    
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    
    metrics_data = []
    for class_name in classes:
        if class_name in report:
            metrics_data.append({
                'Class': class_name,
                'Precision': report[class_name]['precision'],
                'Recall': report[class_name]['recall'],
                'F1-Score': report[class_name]['f1-score'],
                'Support': report[class_name]['support']
            })
    
    df = pd.DataFrame(metrics_data)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    
    x = np.arange(len(classes))
    width = 0.25
    
    ax1.bar(x - width, df['Precision'], width, label='Precision', alpha=0.8)
    ax1.bar(x, df['Recall'], width, label='Recall', alpha=0.8)
    ax1.bar(x + width, df['F1-Score'], width, label='F1-Score', alpha=0.8)
    
    ax1.set_xlabel('Emotion Class', fontsize=12)
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(classes, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, 1.1)
    
    ax2.bar(classes, df['Support'], color='steelblue', alpha=0.8)
    ax2.set_xlabel('Emotion Class', fontsize=12)
    ax2.set_ylabel('Number of Samples', fontsize=12)
    ax2.set_title('Class Distribution in Test Set', fontsize=14, fontweight='bold')
    ax2.set_xticklabels(classes, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_path = os.path.join(cfg.RESULTS_PATH, 'class_performance.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved class performance plot to {save_path}")

def plot_precision_recall_curves(y_true, y_scores, classes, model_name):
    set_style()
    
    y_true_bin = label_binarize(y_true, classes=range(len(classes)))
    n_classes = len(classes)
    
    precision = dict()
    recall = dict()
    avg_precision = dict()
    
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_scores[:, i])
        avg_precision[i] = average_precision_score(y_true_bin[:, i], y_scores[:, i])
    
    precision["micro"], recall["micro"], _ = precision_recall_curve(
        y_true_bin.ravel(), y_scores.ravel()
    )
    avg_precision["micro"] = average_precision_score(y_true_bin, y_scores, average="micro")
    
    plt.figure(figsize=(12, 9))
    
    plt.plot(recall["micro"], precision["micro"],
             label='micro-average (AP = {0:0.3f})'.format(avg_precision["micro"]),
             color='deeppink', linestyle=':', linewidth=4)
    
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple', 'brown', 'pink'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(recall[i], precision[i], color=color, lw=2,
                 label='{0} (AP = {1:0.3f})'.format(classes[i], avg_precision[i]))
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{model_name} - Precision-Recall Curves')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(cfg.RESULTS_PATH, f'{model_name}_precision_recall.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved precision-recall curves to {save_path}")