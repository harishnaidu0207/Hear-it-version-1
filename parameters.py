from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor

classification_grid_parameters = {
    MLPClassifier():    {
        'hidden_layer_sizes': [(200,), (300,), (400,), (128, 128), (256, 256)],
        'alpha': [0.001, 0.005, 0.01],
        'batch_size': [128, 256, 512, 1024],
        'learning_rate': ['constant', 'adaptive'],
        'max_iter': [200, 300, 400, 500]
    }
}
