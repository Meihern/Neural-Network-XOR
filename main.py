from sklearn.model_selection import train_test_split
from multilayerneuralnetwork import layers, networks, functions, losses
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

if __name__ == '__main__':

    rng = np.random.RandomState(0)
    X = rng.randn(300, 2)
    y = np.array(np.logical_xor(X[:, 0] > 0, X[:, 1] > 0), dtype=int)
    y = np.array([y]).T
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    hidden_layer = layers.Layer(dims=4, input_dims=2)
    output_layer = layers.Layer(dims=1, input_dims=4)

    model = networks.MultiLayerNN(hidden_layer, output_layer, loss_function=losses.squared_error)

    # print(model.predict(X_test))

    model.fit(X_train, y_train, epochs=10000)
    y_predicted = model.predict(X_test)

    fig = plt.figure(figsize=(10, 8))
    fig = plot_decision_regions(X=X_test, y=np.array(y_test.T.tolist()[0]), clf=model, legend=2)
    plt.title("XOR MLNN from scratch")
    plt.show()