from scipy.ndimage.filters import gaussian_filter

def blur(img, sigma):
  attacked = gaussian_filter(img, sigma)
  return attacked