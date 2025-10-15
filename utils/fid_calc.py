import numpy as np
from scipy.linalg import sqrtm  # ✅ correct import for matrix square root

def calculate_fid(real_samples, fake_samples):
    """
    Compute a simplified Fréchet Inception Distance (FID)-like score 
    between real and fake feature distributions.
    """
    mu1, sigma1 = np.mean(real_samples, axis=0), np.cov(real_samples, rowvar=False)
    mu2, sigma2 = np.mean(fake_samples, axis=0), np.cov(fake_samples, rowvar=False)

    # Compute mean difference
    diff = mu1 - mu2

    # Compute sqrt of product of covariance matrices
    covmean = sqrtm(sigma1.dot(sigma2))  # ✅ SciPy version
    if np.iscomplexobj(covmean):
        covmean = covmean.real  # remove numerical noise

    # FID formula
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return np.real(fid)
