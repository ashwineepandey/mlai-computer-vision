import os
import numpy as np
import utils as ut
import log
from scipy.stats import entropy
from typing import Tuple
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.datasets import fashion_mnist
from tensorflow.keras.layers import Dense, Flatten, Reshape, LeakyReLU, BatchNormalization, Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from keras.models import load_model
import plotly.express as px
import pandas as pd

logger = log.get_logger(__name__)

@ut.timer
def load_data():
    # Load the Fashion-MNIST dataset
    (X_train, _), (_, _) = fashion_mnist.load_data()
    X_train = X_train / 127.5 - 1.0 # Normalize the images to [-1, 1]
    return np.expand_dims(X_train, axis=3)


# Generator
@ut.timer
def create_generator():
    generator = Sequential([
        Dense(128, input_dim=100),
        LeakyReLU(alpha=0.2),
        BatchNormalization(momentum=0.8),
        Dense(256),
        LeakyReLU(alpha=0.2),
        BatchNormalization(momentum=0.8),
        Dense(512),
        LeakyReLU(alpha=0.2),
        BatchNormalization(momentum=0.8),
        Dense(1024),
        LeakyReLU(alpha=0.2),
        Dense(28 * 28 * 1, activation='tanh'),
        Reshape((28, 28, 1))
    ])

    return generator


# Discriminator
@ut.timer
def create_discriminator():
    discriminator = Sequential([
        Input(shape=(28, 28, 1)),
        Flatten(),
        Dense(512),
        LeakyReLU(alpha=0.2),
        Dense(256),
        LeakyReLU(alpha=0.2),
        Dense(1, activation='sigmoid')
    ])

    return discriminator


# GAN
@ut.timer
def create_gan(discriminator, generator):
    discriminator.trainable = False
    gan_input = Input(shape=(100,))
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = Model(gan_input, gan_output)
    return gan


@ut.timer
def compile_models(discriminator, generator):
    discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])
    gan = create_gan(discriminator, generator)
    gan.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))
    return gan


@ut.timer
def sample_images(generator, epoch, img_out_path, datetime):
    r, c = 5, 5
    noise = np.random.normal(0, 1, (r * c, 100))
    gen_imgs = generator.predict(noise)
    # Rescale images from [-1, 1] to [0, 1]
    gen_imgs = 0.5 * gen_imgs + 0.5
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
            axs[i, j].axis('off')
            cnt += 1
    fig.savefig(f"{img_out_path}/{datetime}_epoch_{epoch}.png")
    plt.close()

def _train_discriminator(discriminator, generator, X_train, conf):
    # Train discriminator
    idx = np.random.randint(0, X_train.shape[0], conf.a3.gan_params.batch_size)
    real_imgs = X_train[idx]
    noise = np.random.normal(0, 1, (conf.a3.gan_params.batch_size, 100))
    gen_imgs = generator.predict(noise)
    
    real_y = np.ones((conf.a3.gan_params.batch_size, 1))
    fake_y = np.zeros((conf.a3.gan_params.batch_size, 1))
    
    d_loss_real = discriminator.train_on_batch(real_imgs, real_y)
    d_loss_fake = discriminator.train_on_batch(gen_imgs, fake_y)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
    return d_loss


def _train_generator(gan, conf):
    # Train generator
    noise = np.random.normal(0, 1, (conf.a3.gan_params.batch_size, 100))
    y = np.ones((conf.a3.gan_params.batch_size, 1))
    g_loss = gan.train_on_batch(noise, y)
    return g_loss


def _preprocess_images(images):
    # Resize images from (28, 28, 1) to (32, 32, 1)
    resized_images = tf.image.resize(images, (32, 32), method=tf.image.ResizeMethod.BILINEAR)
    # Convert grayscale images to RGB by duplicating the single channel three times
    rgb_images = tf.repeat(resized_images, repeats=3, axis=-1)
    return rgb_images

# Train GAN
@ut.timer
def train_gan(X_train, generator, discriminator, gan, classifier, conf, datetime):
    inception_scores = []
    for epoch in range(conf.a3.gan_params.epochs):
        # Train discriminator
        d_loss = _train_discriminator(discriminator, generator, X_train, conf)
        # Train generator
        g_loss = _train_generator(gan, conf)
        if epoch % conf.a3.gan_params.sample_interval == 0:
            logger.info(f"Epoch {epoch}, D-Loss: {d_loss[0]}, G-Loss: {g_loss}")
            sample_images(generator, epoch, conf.a3.paths.training_inspection_plots, datetime)
            gen_imgs = generate_images(generator, 
                                    conf.a3.gan_params.noise_dim, 
                                    conf.a3.gan_params.num_samples)
            gen_imgs_processed = _preprocess_images(gen_imgs)
            pred, _ = make_prediction(classifier, gen_imgs_processed)
            score = inception_score(pred, 
                                       conf.a3.gan_params.num_classes, 
                                       conf.a3.gan_params.epsilon)
            inception_scores.append((epoch, score))
            logger.info(f"Inception Score: {score}")
    pred, labels = make_prediction(classifier, gen_imgs_processed)
    fig = plot_class_distribution(labels, 
                                  conf.a3.fashion_mnist_class_labels)
    save_plot(fig, conf.a3.paths.training_inspection_plots, "classification-distribution", datetime)
    fig = plot_inception_score(inception_scores)
    save_plot(fig, conf.a3.paths.training_inspection_plots, "inception-score", datetime)   
    return generator, discriminator, gan

@ut.timer
def save_models(generator, discriminator, gan, model_out_path, datetime):
    generator.save(f"{model_out_path}{datetime}_generator.h5")
    discriminator.save(f"{model_out_path}{datetime}_discriminator.h5")
    gan.save(f"{model_out_path}{datetime}_gan.h5")
    logger.info("Models saved successfully")


@ut.timer
def load_pretrained_classifier(model_path):
    model = load_model(model_path)
    return model

@ut.timer
def make_prediction(classifier, images: np.ndarray):
    # Predict the labels
    pred = classifier.predict(images)
    # Get the class with the highest probability for each image
    class_labels = np.argmax(pred, axis=1)
    return pred, class_labels


@ut.timer
def inception_score(pred, num_classes: int, eps: float = 1e-16) -> Tuple[float, float]:
    # Compute the KL divergence for each image
    kl = pred * (np.log(pred + eps) - np.log(1.0 / num_classes))
    # Compute the average KL divergence
    avg_kl = np.mean(np.sum(kl, axis=1))
    # Compute the inception score
    score = np.exp(avg_kl)
    return score


@ut.timer
def generate_images(generator, noise_dim, num_samples):
    # Generate a batch of images
    noise = np.random.normal(0, 1, (num_samples, noise_dim))
    gen_imgs = generator.predict(noise)
    gen_imgs = 0.5 * gen_imgs + 0.5  # Rescale images from [-1, 1] to [0, 1]
    return gen_imgs


def plot_class_distribution(labels, class_labels):
    # Count the number of images per class
    counts = np.zeros(len(class_labels))
    for label in labels:
        counts[label] += 1
    # Create a dataframe with the labels and counts
    data = pd.DataFrame({'Label': [class_labels[label] for label in list(range(10))], 'Count': counts})
    # Plot the barplot using plotly.express
    fig = px.bar(data, x='Label', y='Count', title='Generated Images per Class', text='Count')
    fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
    return fig

@ut.timer
def save_plot(fig, plot_filepath, title, datetime):
    prefix = f"{title}-{datetime}.png"
    filepath = os.path.join(plot_filepath, prefix)
    # save figure to file
    fig.write_image(filepath)
    logger.info(f"Plot saved: {filepath}")

@ut.timer
def plot_inception_score(inception_scores):
    iterations = [score[0] for score in inception_scores]
    scores = [score[1] for score in inception_scores]
    data = pd.DataFrame({'Iterations': iterations, 'Inception Score': scores})
    fig = px.line(data, x='Iterations', y='Inception Score', markers=True, title='Inception Score vs Training Iterations')
    return fig

@ut.timer
def main():
    # load config
    conf = ut.load_config()
    classifier = load_pretrained_classifier(model_path=conf.a3.paths.classifier_model)
    X_train = load_data()
    generator = create_generator()
    discriminator = create_discriminator()
    gan = compile_models(discriminator, generator)
    dt = ut.get_current_dt()
    # Set parameters and train the GAN
    generator, discriminator, gan = train_gan(X_train, generator, discriminator, gan, classifier, conf, dt)
    # Save models
    save_models(generator, discriminator, gan, conf.a3.paths.model, dt)


if __name__ == "__main__":
    main()