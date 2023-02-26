import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
# Train the model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

diabetes = pd.read_csv('diabetes.csv')
diabetes.head()
# Separate features and labels
features = ['Pregnancies', 'PlasmaGlucose', 'DiastolicBloodPressure', 'TricepsThickness', 'SerumInsulin', 'BMI',
            'DiabetesPedigree', 'Age']
label = 'Diabetic'
X, y = diabetes[features].values, diabetes[label].values

for n in range(0, 4):
    print("Patient", str(n + 1), "\n  Features:", list(X[n]), "\n  Label:", y[n])

# Split data 70%-30% into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

print('Training cases: %d\nTest cases: %d' % (X_train.shape[0], X_test.shape[0]))

# Set regularization rate
reg = 0.01
# train a logistic regression model on the training set
model = LogisticRegression(C=1 / reg, solver="liblinear").fit(X_train, y_train)
print("Model:", model)
predictions = model.predict(X_test)
print('Predicted labels: ', predictions)
print('Actual labels:    ', y_test)
print('Accuracy: ', accuracy_score(y_test, predictions))
# classification report
print(classification_report(y_test, predictions))
print("Overall Precision:", precision_score(y_test, predictions))
print("Overall Recall:", recall_score(y_test, predictions))
# Print the confusion matrix
cm = confusion_matrix(y_test, predictions)
print("Confusion matrix", cm)
# Predict proba
y_scores = model.predict_proba(X_test)
print("y_scores", y_scores)
# calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_scores[:, 1])

# plot ROC curve
fig = plt.figure(figsize=(6, 6))
# Plot the diagonal 50% line
plt.plot([0, 1], [0, 1], 'k--')
# Plot the FPR and TPR achieved by our model
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
# Area under the curve
auc = roc_auc_score(y_test, y_scores[:, 1])
print('AUC: ' + str(auc))
