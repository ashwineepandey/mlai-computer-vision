from utils import timer, load_config
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
import tensorflow.keras.backend as K
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
    img = preprocess_input(img)
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
def get_loss(class_output, model):
    # Define the loss function as the mean of the output tensor
    loss = tf.reduce_mean(class_output)
    # Compute the gradient of the loss function with respect to the input image
    grads = tf.GradientTape(loss, model.input)
    return loss, grads


@timer
def generate_saliency_map(x, model, class_idx, patch_size=50, stride=10):
    # Iterate over patches of the input image
    saliency_map = np.zeros((224, 224))
    for i in range(0, 224-patch_size, stride):
        for j in range(0, 224-patch_size, stride):
            # Apply occlusion to the patch by setting it to zero
            occluded_x = x.copy()
            occluded_x[:, i:i+patch_size, j:j+patch_size, :] = 0
            # Compute the output probability of the correct class for the occluded patch
            output = model.predict(occluded_x)
            saliency_map[i:i+patch_size, j:j+patch_size] = output[0, class_idx]
    # Normalize the saliency map to [0, 1]
    saliency_map -= np.min(saliency_map)
    saliency_map /= np.max(saliency_map)
    return saliency_map

def plot_saliency_map(filepath, img, saliency_map, class_label, img_path):
    fig = plt.figure(figsize=(10, 5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    # Original image subplot
    ax1.imshow(np.squeeze(img))
    ax1.set_title(f'Original Image\nPredicted Class: {class_label}')
    ax1.axis('off')
    # Saliency map subplot
    im = ax2.imshow(saliency_map, cmap='jet')
    fig.colorbar(im, ax=ax2)
    ax2.set_title(f'Saliency Map\nPredicted Class: {class_label}')
    ax2.axis('off')
    plt.tight_layout()
    plt.savefig(f'{filepath}saliency_map_{img_path}', bbox_inches='tight')
    plt.close()

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
        class_label = imagenet_labels[class_idx]
        saliency_map = generate_saliency_map(img, model, class_idx)
        plot_saliency_map(conf.paths.a1_saliency_plots, img, saliency_map, class_label, img_path)
        

if __name__ == "__main__":
    main()