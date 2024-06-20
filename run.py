from src.logistic_regression_model import logistic_regression_model
from src.random_forest_model import random_forest_model
from src.neural_net_model import neural_net_model
from src.plot_results import plot_confusion_matrix, plot_classification_report

# Logistic Regression
acc, cm, cr = logistic_regression_model('data/logistic_regression_data.csv')
print(f'Logistic Regression Accuracy: {acc}')
plot_confusion_matrix(cm, 'Logistic Regression')
plot_classification_report(cr, 'Logistic Regression')

# Random Forest
acc, cm, cr = random_forest_model('data/random_forest_data.csv')
print(f'Random Forest Accuracy: {acc}')
plot_confusion_matrix(cm, 'Random Forest')
plot_classification_report(cr, 'Random Forest')

# Neural Network
acc, cm, cr = neural_net_model('data/neural_net_data.csv')
print(f'Neural Network Accuracy: {acc}')
plot_confusion_matrix(cm, 'Neural Network')
plot_classification_report(cr, 'Neural Network')
