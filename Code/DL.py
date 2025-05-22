import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score, mean_squared_error, roc_curve, confusion_matrix, accuracy_score, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from sklearn.metrics import roc_auc_score,confusion_matrix,classification_report,accuracy_score
from tensorflow.keras.layers import Dense, SimpleRNN
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
def cal_accuracy(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred) * 100
    report = classification_report(y_test, y_pred)
    print("\nConfusion Matrix:\n", cm)
    print("\nAccuracy: {:.2f}%".format(accuracy))
    print("Report:\n", report)

def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    X = df[['time', 'Remaining_Energy']]
    y = df['NODE_ID'] 
    energy_threshold = 6050  
    y = (df['Remaining_Energy'] < energy_threshold).astype(int)
    df['Energy_Change'] = df['Remaining_Energy'].diff()
    energy_change_threshold = 0.7 
    malicious_nodes = (df['Energy_Change'].abs() <= energy_change_threshold)
    df.drop('malicious_node', axis=1, inplace=True)
    print("Potentially Malicious Nodes:")
    print(df[malicious_nodes])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test

def build_mlp_model(input_dim):
    model = Sequential()
    model.add(Dense(16, input_dim=input_dim, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_mlp_model(X_train, y_train):
    input_dim = X_train.shape[1]
    model = build_mlp_model(input_dim)

    history = model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

    return model, history

def build_decision_tree_model():
    return RandomForestClassifier(random_state=42)

def train_decision_tree_model(X_train, y_train):
    model = build_decision_tree_model()
    model.fit(X_train, y_train)
    return model

def build_rnn_model(input_dim):
    model = Sequential()
    model.add(SimpleRNN(16, input_shape=(input_dim, 1), activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
def build_naive_bayes_model():
    return GaussianNB()

def train_naive_bayes_model(X_train, y_train):
    model = build_naive_bayes_model()
    model.fit(X_train, y_train)
    return model

def train_rnn_model(X_train, y_train):
    input_dim = X_train.shape[1]
    X_train_rnn = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    model = build_rnn_model(input_dim)

    history = model.fit(X_train_rnn, y_train, epochs=10, batch_size=32, verbose=1)

    return model, history

def test_model(model, X_test, y_test, model_name):
    if model_name == 'rnn':
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{model_name.capitalize()} Accuracy: {accuracy * 100:.2f}%")
def predict_mlp(model, scaler, time, energy, threshold=0.2):
    input_data = np.array([[time, energy]])
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)
    prediction_binary = (prediction > threshold).astype(int)
    return prediction_binary

def plot_training_history(history, model_name):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.title(f'{model_name.capitalize()} Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.title(f'{model_name.capitalize()} Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.tight_layout()
    plt.show()
def linear_regression(X_train, X_test, y_train, y_test, threshold=0.6):
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_binary = (y_pred > threshold).astype(int)
    y_test_binary = (y_test > threshold).astype(int)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("Mean Squared Error:", mse)
    print("R-squared:", r2)
    cal_accuracy(y_test_binary, y_pred_binary)
    if X_test.shape[1] == 1:
        plt.scatter(X_test, y_test, color='black')
        plt.plot(X_test, y_pred, color='blue', linewidth=3)
        plt.xlabel('X')
        plt.ylabel('y')
        plt.title('Linear Regression')
        plt.show()
    print("Coefficients:", model.coef_)
    print("Intercept:", model.intercept_)
    print("Linear Regression")

    return mse, r2, model.coef_, model.intercept_

def main():
    file_path = 'C:\\Users\\mouni\\OneDrive\\Desktop\\Dataset\\90.csv' 
    X_train, X_test, y_train, y_test  = preprocess_data(file_path)
    heading = "LINEAR-REGRESSION"
    print("\n" + heading.center(100))
    mse, r2, coef, intercept = linear_regression(X_train, X_test, y_train, y_test)
    
    # MLP
    heading = "MLP"
    print("\n" + heading.center(100))
    X_train_mlp = X_train.copy()
    model_mlp, history_mlp = train_mlp_model(X_train_mlp, y_train)
    y_pred_mlp = model_mlp.predict(X_test)
    y_pred_mlp = (y_pred_mlp > 0.5).astype(int)
    test_model(model_mlp, X_test, y_test, 'mlp')
    cal_accuracy(y_test, y_pred_mlp)
    plot_training_history(history_mlp, 'mlp')
    heading = "Decision-tree"
    print("\n" + heading.center(100))
    
  # Decision Tree (Random Forest)
    X_train_dt = X_train.copy()
    model_dt = train_decision_tree_model(X_train_dt, y_train)
    y_pred_dt = model_dt.predict(X_test)
    y_pred_dt = (y_pred_dt > 1.9).astype(int)
    test_model(model_dt, X_test, y_test, 'decision tree')
    cal_accuracy(y_test, y_pred_dt)

  # RNN
    heading = "RNN"
    print("\n" + heading.center(100))
    X_train_rnn = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    model_rnn, history_rnn = train_rnn_model(X_train_rnn, y_train)
    y_pred_rnn = model_rnn.predict(np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1)))
    y_pred_rnn = (y_pred_rnn > 0.6).astype(int)
    test_model(model_rnn, X_test, y_test, 'rnn')
    cal_accuracy(y_test, y_pred_rnn)
    plot_training_history(history_rnn, 'rnn')
    
    # Naive Bayes
    heading = "Naive Bayes"
    print("\n" + heading.center(100))
    model_nb = train_naive_bayes_model(X_train, y_train)
    y_pred_nb = model_nb.predict(X_test)
    cal_accuracy(y_test, y_pred_nb)
    
  
    
   
    
if __name__ == "__main__":
    main()