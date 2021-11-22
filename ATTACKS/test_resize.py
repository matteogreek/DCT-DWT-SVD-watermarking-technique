from cv2 import resize

def resizing(img, scale):
  x, y = img.shape
  _x = int(x*scale)
  _y = int(y*scale)

  attacked = resize(img, (_x, _y))
  attacked = resize(attacked, (x, y))

  return attacked