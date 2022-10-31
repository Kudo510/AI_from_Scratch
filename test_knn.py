from utils.data_preprocessing import min_max_scaler, normalize_scaler
from utils.model_selection import train_test_split
from utils.data_manipulation import accuracy_score
import numpy as np
from sklearn import datasets

from machine_learning import KNN



def main():
    data = datasets.load_iris()
    X = normalize_scaler(data.data)
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    clf = KNN(k=5)
    y_pred = clf.predict(X_test, X_train, y_train)
    
    accuracy = accuracy_score(y_test, y_pred)

    print ("Accuracy:", accuracy)

if __name__ == "__main__":
    main()