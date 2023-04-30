import numpy as np
# from utils import timer, load_config, get_current_dt, load_npz, f1_score
import utils as ut
import os
import log
import pandas as pd
import itertools
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import gradient_descent_v2
from keras.layers import BatchNormalization, Dropout
from keras import regularizers
from keras.preprocessing.image import ImageDataGenerator


logger = log.get_logger(__name__)

@ut.timer
def define_model(num_conv_layers=2, num_filters=32, filter_size=3, 
                 num_dense_layers=1, dense_units=128, 
                 learning_rate=0.001, momentum=0.9, activation='relu',
                 padding='same', use_batch_norm=False, use_dropout=False,
                 dropout_rate=0.2, use_early_stopping=True, patience=2,
                 l2_weight_decay=0.01):
    """
    Defines a CNN model with the specified number of convolutional and dense layers, number of filters, filter size,
    learning rate, and momentum for the optimizer. Options for batch normalization, dropout, and early stopping are
    included. Regularization using L2 weight decay can also be applied.
    """
    model = Sequential()
    # Add convolutional layers
    for i in range(num_conv_layers):
        if i == 0:
            model.add(Conv2D(num_filters, (filter_size, filter_size), activation=activation, 
                             kernel_initializer='he_uniform', padding=padding, input_shape=(32, 32, 3),
                             kernel_regularizer=regularizers.l2(l2_weight_decay)))
        else:
            model.add(Conv2D(num_filters, (filter_size, filter_size), activation=activation, 
                             kernel_initializer='he_uniform', padding=padding,
                             kernel_regularizer=regularizers.l2(l2_weight_decay)))
        if use_batch_norm:
            model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))
    # Flatten the output of the convolutional layers
    model.add(Flatten())
    # Add dense layers
    for i in range(num_dense_layers):
        model.add(Dense(dense_units, activation=activation, kernel_initializer='he_uniform',
                        kernel_regularizer=regularizers.l2(l2_weight_decay)))
        if use_batch_norm:
            model.add(BatchNormalization())
        if use_dropout:
            model.add(Dropout(dropout_rate))
    # Add output layer
    model.add(Dense(10, activation='softmax'))
    # Compile the model with the specified optimizer, learning rate, and momentum
    opt = gradient_descent_v2.SGD(learning_rate=learning_rate, momentum=momentum)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy', ut.f1_score])
    if use_early_stopping:
        es = EarlyStopping(monitor='val_loss', patience=patience)
        logger.info(f"Using EarlyStopping with patience={patience}.")
        logger.info(f"Model defined: {model.summary()}")
        return model, es
    else:
        logger.info(f"Model defined: {model.summary()}")
        return model

@ut.timer
def save_model(model_path, model, model_name, combination_num, datetime):
    """
    Saves the trained model to disk.
    """
    # Create directory for saving models if it does not exist
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    # Save model architecture and weights
    model.save(f"{model_path}{model_name}_{combination_num}_{datetime}.h5")
    logger(f"Model saved: {model_path}{model_name}_{combination_num}_{datetime}.h5")

@ut.timer
def save_history(history_path, history, model_name, combination_num, datetime):
    """
    Saves the training history to disk.
    """
    # Create directory for saving history if it does not exist
    if not os.path.exists(history_path):
        os.makedirs(history_path)
    
    # Save training history to csv
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(f"{history_path}{model_name}_{combination_num}_{datetime}_history.csv", index=False)
    logger.info(f"Training history saved: {history_path}{model_name}_{combination_num}_{datetime}_history.csv")

@ut.timer
def train(model, es, aug_gen, valX, valY, batch_size, epochs,
                model_path, model_name, combination_num, datetime):
    """
    Trains the CNN model on the training set and validates on the validation set.
    """
    # define the callback to save the weights
    checkpoint = ModelCheckpoint(f'{model_path}{model_name}_{combination_num}_{datetime}.h5', 
                                 monitor='val_accuracy', 
                                 save_best_only=True, 
                                 save_weights_only=True, 
                                 verbose=1)

    history = model.fit(aug_gen,
                        validation_data=(valX, valY),
                        callbacks=[es, checkpoint], 
                        batch_size=batch_size, 
                        epochs=epochs,
                        verbose=1)
    model.save(f"{model_path}{model_name}_{combination_num}_{datetime}.h5")    
    return history



@ut.timer
def save_plot(plot_filepath, fig, dataset, datetime):
    """
    Saves the plot to file.
    """
    # generate file name
    prefix = f"cifar10-classification-{dataset}-{datetime}.png"
    filepath = os.path.join(plot_filepath, prefix)
    # save figure to file
    fig.write_image(filepath)
    logger.info(f"Plot saved: {prefix}")

@ut.timer
def train_model(conf, params, param_names, batch_size, epochs, aug_gen, valX, valY, datetime, i=0):
    logger.info(f"Training model {i} with Params: {params}")
    # create dictionary of hyperparameters for this model
    param_dict = dict(zip(param_names, params))
    # define model with current set of hyperparameters
    model, es = define_model(**param_dict)
    # train model with current set of hyperparameters
    history = train(model, es, aug_gen, valX, valY, 
                            batch_size, epochs,
                            conf.paths.a1_model, "cifar10", i, datetime)
    
    # save trained model and training history
    # save_model(conf.paths.a1_model, "cifar10", i, datetime)
    save_history(conf.paths.a1_train_history, history, "cifar10", i, datetime)
    fig = ut.plot_performance(history, dataset="validation")
    save_plot(conf.paths.a1_train_plots, fig, "validation", datetime)

    # Extract the accuracy and f1 score from the history
    train_accuracy = history.history["train_accuracy"][-1]
    train_f1_score = history.history["train_f1_score"][-1]
    val_accuracy = history.history["val_accuracy"][-1]
    val_f1_score = history.history["val_f1_score"][-1]
    return train_accuracy, train_f1_score, val_accuracy, val_f1_score


def update_results_df(results_df, epochs, batch_size, params, train_accuracy, train_f1_score, val_accuracy, val_f1_score, i=0):
    results_df = results_df.append({
            "iteration": i,
            "epochs": epochs,
            "batch_size": batch_size,
            "num_conv_layers": params["num_conv_layers"],
            'num_filters': params["num_filters"],
            'filter_size': params["filter_size"],
            "num_dense_layers": params["num_dense_layers"],
            "dense_units": params["dense_units"],
            "learning_rate": params["learning_rate"],
            "momentum": params["momentum"],
            "activation": params["activation"],
            "padding": params["padding"],
            "use_batch_norm": params["use_batch_norm"],
            "use_dropout": params["use_dropout"],
            "dropout_rate": params["dropout_rate"],
            "use_early_stopping": params["use_early_stopping"],
            "patience": params["patience"],
            "train_accuracy": train_accuracy, 
            "train_f1_score": train_f1_score,
            "val_accuracy": val_accuracy,
            "val_f1_score": val_f1_score,
        }, ignore_index=True)
    return results_df


@ut.timer
def train_models_primary(conf, data):
    """
    Trains multiple models with different hyperparameters.
    """
    # augment data
    aug_gen = augment_data(data['trainX'], data['trainY'], batch_size=conf.a1_q1_primary_params.batch_size)
    # generate all possible combinations of hyperparameters
    param_values = list(itertools.product(*conf.a1_q1_hyperparams.values()))
    param_names = list(conf.a1_q1_hyperparams.keys())
    # Create an empty dataframe to store the results
    results_df = pd.DataFrame(columns=["iteration", "epochs", "batch_size", *param_names, "train_accuracy", "train_f1_score", "val_accuracy", "val_f1_score"])
    batch_size = conf.a1_q1_primary_params.batch_size, 
    epochs = conf.a1_q1_primary_params.epochs
    for i, params in enumerate(param_values):
        datetime = ut.get_current_dt()
        # Train the model
        train_accuracy, train_f1_score, val_accuracy, val_f1_score = train_model(conf, 
                        params, param_names, batch_size, epochs, aug_gen, data['valX'], data['valY'], datetime, i)
        # Append the results to the dataframe
        results_df = update_results_df(results_df, conf.a1_q1_primary_params.epochs, 
                                       conf.a1_q1_primary_params.batch_size, params, 
                                       train_accuracy, train_f1_score, val_accuracy, 
                                       val_f1_score, i=0)
    # Save the results dataframe to a csv file
    results_df.to_csv(f"{conf.paths.a1_hyptuning}{datetime}_hyperparameter_tuning_primary_results.csv", index=False)
    return results_df

@ut.timer
def filter_results(results_df, param_names, result_col="val_f1_score", top_n=3):
    results_df.sort_values(by=result_col, ascending=False, inplace=True)
    secondary = results_df.loc[:, param_names].head(top_n).to_dict(orient="list")
    return secondary

@ut.timer
def train_models_secondary(conf, data, secondary):
    """
    Trains multiple models with different hyperparameters.
    """
    # generate all possible combinations of hyperparameters
    param_values = list(itertools.product(*secondary.values()))
    param_names = list(conf.a1_q1_hyperparams.keys())
    # Create an empty dataframe to store the results
    results_df = pd.DataFrame(columns=["iteration", "epochs", "batch_size", *param_names, "train_accuracy", "train_f1_score", "val_accuracy", "val_f1_score"])
    for batch_size in conf.a1_q1_secondary_params.batch_size:
        epochs = conf.a1_q1_secondary_params.epochs
        # augment data
        aug_gen = augment_data(data['trainX'], data['trainY'], batch_size)
        for i, params in enumerate(param_values):
            datetime = ut.get_current_dt()
            # Train the model
            train_accuracy, train_f1_score, val_accuracy, val_f1_score = train_model(conf, 
                        params, param_names, batch_size, epochs, aug_gen, data['valX'], data['valY'], datetime, i)
            # Append the results to the dataframe
            results_df = update_results_df(results_df, params, train_accuracy, train_f1_score, val_accuracy, val_f1_score, i=0)
    # Save the results dataframe to a csv file
    results_df.to_csv(f"{conf.paths.a1_hyptuning}{datetime}_hyperparameter_tuning_secondary_results.csv", index=False)
    return results_df


@ut.timer
def augment_data(X_train, Y_train, batch_size=32):
    # create ImageDataGenerator object for data augmentation
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest')
    
    # fit generator to data
    datagen.fit(X_train)
    # create generator for augmented data
    return datagen.flow(X_train, Y_train, batch_size=batch_size)


@ut.timer
def main():
    # load config
    conf = ut.load_config()
    # load data
    data = ut.load_npz(f"{conf.paths.a1_input_data}cifar10.npz")
    # train models
    primary_results_df = train_models_primary(conf, data)
    # filter results
    secondary = filter_results(primary_results_df, list(conf.a1_q1_hyperparams.keys()))
    # train secondary models
    secondary_results_df = train_models_secondary(conf, data, secondary)
    # filter results
    tertiary = filter_results(secondary_results_df, list(conf.a1_q1_hyperparams.keys()), top_n=1)
    # train tertiary model
    train_accuracy, train_f1_score, val_accuracy, val_f1_score = train_model(conf, tertiary, 
                list(conf.a1_q1_hyperparams.keys()), 
                tertiary["batch_size"].values[0], 
                conf.a1_q1_tertiary_params.epochs, 
                data['trainX'], data['trainY'], 
                data['valX'], data['valY'], 
                ut.get_current_dt(), 999)
    logger.info(f"Train accuracy: {train_accuracy}")
    logger.info(f"Train f1 score: {train_f1_score}")
    logger.info(f"Validation accuracy: {val_accuracy}")
    logger.info(f"Validation f1 score: {val_f1_score}")
    logger.info("Done!")

if __name__ == "__main__":
    main()