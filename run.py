from src.logistic_regression_model import logistic_regression_model as lrm
from src.random_forest_model import random_forest_model as rfm
from src.neural_net_model import neural_net_model as nnm
from src.plot_results import plot_confusion_matrix, plot_classification_report

# Logistic Regression
acc, cm, cr = lrm('data/logistic_regression_data.csv')
print(f'Logistic Regression Accuracy: {acc}')
plot_confusion_matrix(cm, 'Logistic Regression')
plot_classification_report(cr, 'Logistic Regression')

# Random Forest
acc, cm, cr = rfm('data/random_forest_data.csv')
print(f'Random Forest Accuracy: {acc}')
plot_confusion_matrix(cm, 'Random Forest')
plot_classification_report(cr, 'Random Forest')

# Neural Network
acc, cm, cr = nnm('data/neural_net_data.csv')
print(f'Neural Network Accuracy: {acc}')
plot_confusion_matrix(cm, 'Neural Network')
plot_classification_report(cr, 'Neural Network')
