from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def logistic_regression(X_train, X_test, y_train, y_test, random_seed):
    lr_model = LogisticRegression(max_iter=10000, random_state=random_seed)
    lr_model.fit(X_train, y_train)

    y_pred = lr_model.predict(X_test)
    print(f'Logistic Regression ChatGPT Refusal Detection Accuracy Using Perturbed VAE Encodings Representing the Refusal Texts: {accuracy_score(y_test, y_pred)*100:.2f}%')
