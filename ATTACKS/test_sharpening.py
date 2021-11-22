from scipy.ndimage import gaussian_filter

def sharpening(img, sigma, alpha):
  filter_blurred_f = gaussian_filter(img, sigma)
  attacked = img + alpha * (img - filter_blurred_f)
  
  return attacked