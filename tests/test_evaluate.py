import pytest
import numpy as np
import pandas as pd
from src.evaluate import evaluate_model, plot_confusion_matrix, plot_roc_curve_custom, plot_pr_curve_custom

@pytest.fixture
def mock_predictions():
    y_true = np.array([0, 1, 0, 1, 0, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 0, 1, 1, 1, 0, 1])
    y_prob = np.array([0.1, 0.9, 0.2, 0.4, 0.3, 0.8, 0.7, 0.85, 0.15, 0.95])
    return y_true, y_pred, y_prob

def test_evaluate_model(mock_predictions):
    y_true, y_pred, y_prob = mock_predictions
    metrics = evaluate_model(y_true, y_pred, y_prob)
    
    assert 'Accuracy' in metrics
    assert 'ROC-AUC' in metrics
    assert metrics['Accuracy'] == 0.8
    assert metrics['Recall'] == 0.8

def test_evaluate_model_no_prob(mock_predictions):
    y_true, y_pred, _ = mock_predictions
    metrics = evaluate_model(y_true, y_pred)
    
    assert 'Accuracy' in metrics
    assert 'ROC-AUC' not in metrics

def test_plots(mock_predictions):
    y_true, y_pred, y_prob = mock_predictions
    fig1 = plot_confusion_matrix(y_true, y_pred)
    fig2 = plot_roc_curve_custom(y_true, y_prob)
    fig3 = plot_pr_curve_custom(y_true, y_prob)
    
    assert fig1 is not None
    assert fig2 is not None
    assert fig3 is not None
