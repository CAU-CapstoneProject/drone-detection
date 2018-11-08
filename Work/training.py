from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_recall_fscore_support
import pandas as pd

def cross_val(model, X, Y, size=0.2, iteration=10):
    # Print setting
    print(model)
    print("Input Data: ", X.shape)
    print("Label Data: ", Y.shape)
    print("Ratio: ", size)
    print("Iteration: ", iteration, "\n")
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=size, shuffle=True)

    # Bugger to store a result
    result = pd.Series()
    pred = pd.Series()
    
    # Iteration Start
    for k in range(iteration):
        model = model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        pred.at[k] = y_pred
        result.at[k] = accuracy_score(y_test, y_pred)
        print(k, "is finished, Accuracy: ", result.at[k])
    
    # Print result
    p,r,f,s = precision_recall_fscore_support(y_test, pred.mean(), average='micro')
    print("\nF-Score:", round(f,3))
    print("Accuracy: ", result.mean())
    print(classification_report(y_test, pred.mean()))
    print(confusion_matrix(y_test, pred.mean()))

    # Return Accuracy mean
    return result.mean()