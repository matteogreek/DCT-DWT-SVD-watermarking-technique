import numpy as np

def awgn(img, std, seed):
  mean = 0.0
  np.random.seed(seed)
  attacked = img + np.random.normal(mean, std, img.shape)
  attacked = np.clip(attacked, 0, 255)
  return attacked