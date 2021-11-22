from scipy.signal import medfilt

def median(img, kernel_size):
  attacked = medfilt(img, kernel_size)
  return attacked