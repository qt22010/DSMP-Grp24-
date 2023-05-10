import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Input, Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

def create_autoencoder(hidden_layers, input_dim, encoding_dim):
    input_layer = Input(shape=(input_dim,))
    encoder = Dense(hidden_layers[0], activation='relu')(input_layer)

    for hl in hidden_layers[1:]:
        encoder = Dense(hl, activation='relu')(encoder)

    encoder_output = Dense(encoding_dim, activation='relu')(encoder)

    decoder = Dense(hidden_layers[-1], activation='relu')(encoder_output)
    for hl in hidden_layers[-2::-1]:
        decoder = Dense(hl, activation='relu')(decoder)

    decoder_output = Dense(input_dim, activation='sigmoid')(decoder)

    autoencoder = Model(inputs=input_layer, outputs=decoder_output)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    return autoencoder

def main():
    # Load your dataset
    data = pd.read_csv("your_dataset.csv")
    X = data.values
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

    # Set the autoencoder's hyperparameters for grid search
    param_grid = {
        'hidden_layers': [
            (128, 64), (64, 32), (32, 16)
        ],
        'input_dim': [X_train.shape[1]],
        'encoding_dim': [16, 8, 4],
        'epochs': [50, 100],
        'batch_size': [32, 64]
    }

    autoencoder = KerasClassifier(build_fn=create_autoencoder, verbose=0)

    grid = GridSearchCV(estimator=autoencoder, param_grid=param_grid, cv=3)
    grid_result = grid.fit(X_train, X_train)

    # Display the optimal hyperparameters
    print("Best score: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

    # Train the autoencoder with the optimal hyperparameters on the whole training set
    best_params = grid_result.best_params_
    best_autoencoder = create_autoencoder(best_params['hidden_layers'], best_params['input_dim'], best_params['encoding_dim'])
    best_autoencoder.fit(X_train, X_train, epochs=best_params['epochs'], batch_size=best_params['batch_size'])

    # Evaluate the autoencoder on the test set
    loss = best_autoencoder.evaluate(X_test, X_test)
    print("Test loss: %f" % loss)

if __name__ == "__main__":
    main()