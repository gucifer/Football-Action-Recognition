import matplotlib.pyplot as plt
import numpy as np


def visualize_attr_maps(path, X, title, attributions, titles,
                        attr_preprocess=lambda attr: attr.permute(1, 2, 0).detach().numpy(),
                        cmap='viridis', alpha=0.7, N = 0):
    """
    A helper function to visualize captum attributions for a list of captum attribution algorithms.

    path (str): name of the final saved image with extension (note: if batch of images are in X, 
                      all images/plots saved together in one final output image with filename equal to path)
    X (numpy array): shape (N, H, W, C)
    y (numpy array): shape (N,)
    class_names (dict): length equal to number of classes
    attributions(A list of torch tensors): Each element in the attributions list corresponds to an
                      attribution algorithm, such as Saliency, Integrated Gradient, Perturbation, etc.
    titles(A list of strings): A list of strings, names of the attribution algorithms corresponding to each element in
                      the `attributions` list. len(attributions) == len(titles)
    attr_preprocess: A preprocess function to be applied on each image attribution before visualizing it with
                      matplotlib. Note that if there are a batch of images and multiple attributions 
                      are visualized at once, this would be applied on each individual image for each attribution
                      i.e. attr_preprocess(attributions[j][i])
    """
    if not N:
        N = attributions[0].shape[0]
    framesToSave = []
    skipFrames = 30/N
    for i in range(N):
        framesToSave.append(1 + int(i * skipFrames))
    plt.figure()
    for i, f in enumerate(framesToSave):
        plt.subplot(len(attributions) + 1, N + 1, i+1)
        plt.imshow(X[f])
        plt.axis('off')
        plt.title(title)

    plt.subplot(len(attributions) + 1, N + 1, N + 1)
    plt.text(0.0, 0.5, 'Original Images', fontsize=14)
    plt.axis('off')
    for j in range(len(attributions)):
        for it, i in enumerate(framesToSave):
            plt.subplot(len(attributions) + 1, N + 1, (N + 1) * (j + 1) + it + 1)
            attr = np.array(attr_preprocess(attributions[j][i]))
            attr = (attr - np.mean(attr)) / np.std(attr).clip(1e-20)
            attr = attr * 0.2 + 0.5
            attr = attr.clip(0.0, 1.0)
            plt.imshow(attr, cmap=cmap, alpha=alpha)
            plt.axis('off')
        plt.subplot(len(attributions) + 1, N + 1, (N + 1) * (j + 1) + N + 1)
        plt.text(0.0, 0.5, titles[j], fontsize=14)
        plt.axis('off')

    plt.gcf().set_size_inches(20, 13)
    plt.savefig(path, bbox_inches='tight')
    plt.close()


def compute_attributions(algo, inputs, **kwargs):
    """
    A common function for computing captum attributions
    """
    return algo.attribute(inputs, **kwargs)
