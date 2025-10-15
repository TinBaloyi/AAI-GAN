import matplotlib.pyplot as plt


def plot_gan_losses(g_losses, d_losses):
    plt.figure(figsize=(8,5))
    plt.plot(g_losses, label='Generator Loss')
    plt.plot(d_losses, label='Discriminator Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('GAN Loss Progression')
    plt.legend()
    plt.show()