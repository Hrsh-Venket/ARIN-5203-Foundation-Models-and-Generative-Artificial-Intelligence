import argparse
import torch
from cleanfid import fid
from matplotlib import pyplot as plt


def save_plot(x, y, xlabel, ylabel, title, filename):
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(filename + ".png")


@torch.no_grad()
def get_fid(gen, dataset_name, dataset_resolution, z_dimension, batch_size, num_gen):
    gen_fn = lambda z: (gen.forward_given_samples(z) / 2 + 0.5) * 255
    score = fid.compute_fid(
        gen=gen_fn,
        dataset_name=dataset_name,
        dataset_res=dataset_resolution,
        num_gen=num_gen,
        z_dim=z_dimension,
        batch_size=batch_size,
        verbose=True,
        dataset_split="custom",
    )
    return score


@torch.no_grad()
def interpolate_latent_space(gen, path):
    ##################################################################
    # TODO: 1.2: Generate and save out latent space interpolations.
    # 1. Generate 100 samples of 128-dim vectors. Do so by linearly
    # interpolating for 10 steps across each of the first two
    # dimensions between -1 and 1. Keep the rest of the z vector for
    # the samples to be some fixed value (e.g. 0).
    # 2. Forward the samples through the generator.
    # 3. Save out an image holding all 100 samples.
    # Use torchvision.utils.save_image to save out the visualization.
    ##################################################################
    import torchvision.utils

    # Create 100 latent vectors (10x10 grid)
    # Get device from generator's dummy buffer
    device = gen._dummy.device if hasattr(gen, '_dummy') else 'cuda'
    z_samples = torch.zeros(100, 128, device=device)

    # Linearly interpolate the first two dimensions
    # First dimension varies across rows, second across columns
    steps = torch.linspace(-1, 1, 10)

    idx = 0
    for i in range(10):
        for j in range(10):
            z_samples[idx, 0] = steps[i]  # First dimension
            z_samples[idx, 1] = steps[j]  # Second dimension
            idx += 1

    # Forward through generator
    generated_images = gen.forward_given_samples(z_samples)

    # Rescale from [-1, 1] to [0, 1]
    generated_images = (generated_images + 1) / 2

    # Save image as 10x10 grid
    torchvision.utils.save_image(generated_images, path, nrow=10)
    ##################################################################
    #                          END OF YOUR CODE                      #
    ##################################################################

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--disable_amp", action="store_true")
    args = parser.parse_args()
    return args
