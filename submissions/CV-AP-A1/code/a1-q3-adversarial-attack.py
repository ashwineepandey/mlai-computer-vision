from utils import timer, load_config
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
import pickle

@timer
def load_pretrained_model(filepath, local=True):
    if local:
        return load_model(filepath)
    else:
        return VGG16(weights='imagenet')


@timer
def _preprocess_img(img):
    # convert to array
    img = image.img_to_array(img)
    # reshape to 1 sample
    img = np.expand_dims(img, axis=0)
    # preprocess image
    # img = preprocess_input(img)
    return img

@timer
def load_sample_img(filepath):
    """
    Loads a sample image from the input data folder.
    """
    # load image
    img = image.load_img(f"{filepath}", target_size=(224, 224))
    return _preprocess_img(img)


@timer
def get_image_class(img_arr, model):
    # Get the predicted class and the corresponding output tensor
    preds = model.predict(img_arr)
    class_idx = np.argmax(preds[0])
    class_output = model.output[:, class_idx]
    return class_idx, class_output 

@timer
def define_loss(model, input_image):
    # Define the loss function
    target_class = tf.argmax(model(input_image), axis=1)
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    loss = lambda input_image: loss_fn(model(input_image), tf.one_hot(target_class, 1000))
    return loss

@timer
def compute_grad(input_image, loss):
    # Compute the gradient of the loss with respect to the input
    input_tensor = tf.Variable(input_image)
    with tf.GradientTape() as tape:
        tape.watch(input_tensor)
        loss_value = loss(input_tensor)
    gradient = tape.gradient(loss_value, input_tensor)
    
    # Normalize the gradient
    gradient /= tf.maximum(tf.reduce_mean(tf.abs(gradient)), tf.keras.backend.epsilon())
    return gradient

@timer
def generate_adversarial_image(gradient, input_image, epsilon):
    # Generate the adversarial example
    perturbation = tf.sign(gradient)
    adversarial_image = input_image + epsilon * perturbation
    adversarial_image = tf.clip_by_value(adversarial_image, 0, 255)
    adversarial_image = tf.cast(adversarial_image, tf.uint8)
    return adversarial_image

@timer
def generate_adversarial_example(model, input_image, epsilon):
    loss = define_loss(model, input_image)
    # Compute the gradient of the loss with respect to the input
    gradient = compute_grad(input_image, loss)
    adversarial_image = generate_adversarial_image(gradient, input_image, epsilon)
    return adversarial_image.numpy()

def plot_adversarial_example(filepath, img, adversarial_image, 
                             original_label, adversarial_label, 
                             epsilon, img_path):
    # Plot the original image, the noise, and the adversarial image
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 6))
    axes[0].imshow(img / 255.)
    axes[0].set_title(f"Original Image ({original_label})")
    axes[1].imshow((adversarial_image - img) / (2 * epsilon) + 0.5)
    axes[1].set_title("Perturbation")
    axes[2].imshow(adversarial_image / 255)
    axes[2].set_title(f"Adversarial Image ({adversarial_label})")
    # Save the plot
    plt.savefig(f'{filepath}adversarial_{epsilon}_{img_path}.png')


@timer
def main():
    conf = load_config()
    img_paths = ['example_pomeranian.jpeg', 'example_carwheel.jpg', 'example_afghanhound.jpg']
    model = load_pretrained_model(conf, local=False)
    # Load the labels for the ImageNet dataset
    imagenet_labels = pickle.load(open(f'{conf.paths.a1_imagenet_labels}', 'rb'))
    for img_path in img_paths:
        img = load_sample_img(f"{conf.paths.a1_raw_data}{img_path}")
        class_idx, _ = get_image_class(img, model)
        original_label = imagenet_labels[class_idx]
        adversarial_image = generate_adversarial_example(model, img, conf.a1_q3_params.epsilon)
        # Get the predicted label for the adversarial image
        preds_adv = model.predict(adversarial_image)
        adversarial_label = tf.keras.applications.vgg16.decode_predictions(preds_adv, top=1)[0][0][1]
        img = np.squeeze(img)
        adversarial_image = np.squeeze(adversarial_image)
        plot_adversarial_example(conf.paths.a1_adversarial_plots, img, adversarial_image, 
                             original_label, adversarial_label, 
                             conf.a1_q3_params.epsilon, img_path)

if __name__ == "__main__":
    main()