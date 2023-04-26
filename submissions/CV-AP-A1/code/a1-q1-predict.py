import numpy as np
from utils import timer, load_config, get_current_dt, load_npz, f1_score
import os
import log
import plotly.express as px
from sklearn.metrics import confusion_matrix
from keras.models import load_model
from keras.optimizers import gradient_descent_v2

logger = log.get_logger(__name__)

@timer
def load_best_model(model_path, learning_rate, momentum):
    """
    Loads a trained model from disk.
    """
    model = load_model(model_path, custom_objects={'f1_score': f1_score})
    opt = gradient_descent_v2.SGD(learning_rate, momentum)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy', f1_score])
    return model

@timer
def evaluate_model(model, testX, testY):
    """
    Evaluates the trained model on the test set.
    """
    # evaluate model on test set
    loss, acc, f1 = model.evaluate(testX, testY, verbose=1)
    # print results
    logger.info(f"Test Loss: {loss:.3f}")
    logger.info(f"Test Accuracy: {acc:.3f}")
    logger.info(f"Test F1 Score: {f1:.3f}")
    return loss, acc, f1

@timer
def predict(model, testX, testY):
    """
    Predicts the class labels for the test set.
    """
    # predict class labels for test set
    yhat = model.predict(testX)
    # convert predictions to class labels
    yhat = np.argmax(yhat, axis=1)
    # convert one-hot encoded labels to class labels
    testY = np.argmax(testY, axis=1)
    return yhat, testY

@timer
def plot_confusion_matrix(cm, classes, normalize=False, 
                          title='Confusion matrix', cmap='Blues', 
                          loss=None, accuracy=None, f1=None):
    """
    Plots the confusion matrix.
    """
    # normalize confusion matrix
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        logger.info("Normalized confusion matrix")
    else:
        logger.info('Confusion matrix, without normalization')
    # print confusion matrix
    logger.info(cm)
    
    # create figure
    fig = px.imshow(cm, labels=dict(x="Predicted label", y="True label"), 
                    x=classes, y=classes, color_continuous_scale=cmap,
                    title=title)
    
    # add text for loss, accuracy, and f1 score
    if loss is not None:
        fig.add_annotation(text=f"Loss: {loss:.4f}", xref="paper", yref="paper",
                            x=0.5, y=1.1, showarrow=False)
    if accuracy is not None:
        fig.add_annotation(text=f"Accuracy: {accuracy:.4f}", xref="paper", yref="paper",
                            x=0.5, y=1.05, showarrow=False)
    if f1 is not None:
        fig.add_annotation(text=f"F1 Score: {f1:.4f}", xref="paper", yref="paper",
                            x=0.5, y=1.0, showarrow=False)
    
    return fig


@timer
def save_confusion_matrix(conf, cm, dataset, loss, accuracy, f1):
    """
    Saves the confusion matrix to file.
    """
    datetime = get_current_dt()
    # generate file name
    prefix = f"cifar10-classification-{dataset}-confusion-matrix-{datetime}.png"
    filepath = os.path.join(conf.paths.a1_reporting_plots, prefix)
    classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # save figure to file
    fig = plot_confusion_matrix(cm, classes, normalize=False, loss=loss, accuracy=accuracy, f1=f1)
    fig.write_image(filepath)
    logger.info(f"Confusion matrix saved: {filepath}")

@timer
def main():
    # load config
    conf = load_config()
    # load data
    data = load_npz(f"{conf.paths.a1_input_data}cifar10.npz")
    # load model
    model = load_best_model(conf.paths.a1_q1_model, 
                            conf.a1_q1_hyperparams.learning_rate[0], 
                            conf.a1_q1_hyperparams.momentum[0])
    # evaluate model on test set
    loss, acc, f1 = evaluate_model(model, data['testX'], data['testY'])
    predY, testY = predict(model, data['testX'], data['testY'])
    cm = confusion_matrix(testY, predY)
    save_confusion_matrix(conf, cm, dataset='test', loss=loss, accuracy=acc, f1=f1)


if __name__ == "__main__":
    main()