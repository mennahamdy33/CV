
def getGaussian_kernel(self):
    gaussian_kernel = np.empty([5, 5])
    kernelSum = 0.0

    for y in xrange(-2, 3):
        for x in xrange(-2, 3):
            my = y + 2
            mx = x + 2
            gaussian_kernel[mx,my] = np.exp(-float(x **2 + y **2) / (2.*0.5**2))/(2.*np.pi*0.5**2)
            kernelSum += gaussian_kernel[mx,my]

    ratio = 1.0/kernelSum

    for y in xrange(5):
        for x in xrange(5):
            gaussian_kernel[x,y] = gaussian_kernel[x,y] * ratio

    return gaussian_kernel
    
def computeHarrisValues(self, srcImage):
  
    height, width = srcImage.shape[:2]

    harrisImage = np.zeros(srcImage.shape[:2])
    orientationImage = np.zeros(srcImage.shape[:2])

  
    Ix = scipy.ndimage.sobel(srcImage, 1)
    Iy = scipy.ndimage.sobel(srcImage, 0)

    gaussian_kernel = self.getGaussian_kernel()

    for x in xrange(width):
        for y in xrange(height):
            h00, h01, h10, h11 = 0, 0, 0, 0
            for m in xrange(-2, 3):
                for n in xrange(-2, 3):
                    mm = m + 2
                    nn = n + 2
                    yn = y + n
                    xm = x + m
                    if (yn < 0):
                        yn = -yn
                    if (xm < 0):
                        xm = -xm
                    if (yn > height - 1):
                        yn = (height - 1) - (yn - (height - 1))
                    if (xm > width - 1):
                        xm = (width - 1) - (xm - (width - 1))

                    h00 += gaussian_kernel[mm, nn] * Ix[yn, xm] **2
                    h01 += gaussian_kernel[mm, nn] * Ix[yn, xm] * Iy[yn, xm]
                    h10 = h01
                    h11 += gaussian_kernel[mm, nn] * Iy[yn, xm] **2

            r = (h00 * h11 - h01 * h10) - 0.1 * (h00 + h11) **2
            harrisImage[y][x] = r
            orientationImage[y][x] = np.arctan2(Iy[y, x], Ix[y, x]) * 180. / np.pi

    # TODO-BLOCK-END

    return harrisImage, orientationImage

def computeLocalMaxima(self, harrisImage):

    destImage = np.zeros_like(harrisImage, np.bool)

    height, width = harrisImage.shape[:2]

    for x in xrange(width):
        for y in xrange(height):
            destImage[y, x] = True
            for m in xrange(-3, 4):
                for n in xrange(-3, 4):
                    try:
                        if (harrisImage[y, x] < harrisImage[y + n, x + m]):
                            destImage[y, x] = False
                    except:
                        pass

    return destImage


def detectKeypoints(self, image):
      
        image = image.astype(np.float32)
        image /= 255.
        height, width = image.shape[:2]
        features = []

        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        harrisImage, orientationImage = self.computeHarrisValues(grayImage)

        harrisMaxImage = self.computeLocalMaxima(harrisImage)

        List =[]

        for y in range(height):
            for x in range(width):
                if not harrisMaxImage[y, x]:
                    continue
                
                # cv2.circle(firstimage, (col, row), radius, color, thickness)
                # List.append((row, col))

                f = cv2.KeyPoint()
                f.size = 10
                f.pt = x, y
                f.angle = orientationImage[y, x]
                f.response = harrisImage[y, x]
                List.append([x,y])

                features.append(f)
        return List

detectKeypoints(self, image)