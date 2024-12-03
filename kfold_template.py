from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

def run_kfold(model, data, target, k=4, verbose=False):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    scores = []
    for train_idx, test_idx in kf.split(data):
        X_train, X_test = data[train_idx], data[test_idx]
        y_train, y_test = target[train_idx], target[test_idx]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        scores.append(accuracy)
        if verbose:
            print(f"Fold Accuracy: {accuracy}")
    return scores
