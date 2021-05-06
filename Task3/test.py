def computeHarrisValues(self, srcImage):
        '''
        Input:
            srcImage -- Grayscale input image in a numpy array with
                        values in [0, 1]. The dimensions are (rows, cols).
        Output:
            harrisImage -- numpy array containing the Harris score at
                           each pixel.
            orientationImage -- numpy array containing the orientation of the
                                gradient at each pixel in degrees.
        '''
        height, width = srcImage.shape[:2]

        harrisImage = np.zeros(srcImage.shape[:2])
        orientationImage = np.zeros(srcImage.shape[:2])

        # TODO 1: Compute the harris corner strength for 'srcImage' at
        # each pixel and store in 'harrisImage'.  See the project page
        # for direction on how to do this. Also compute an orientation
        # for each pixel and store it in 'orientationImage.'
        # TODO-BLOCK-BEGIN

        Ix = scipy.ndimage.sobel(srcImage, 1)
        Iy = scipy.ndimage.sobel(srcImage, 0)

        gaussian_kernel = getGaussian_kernel()

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
        '''
        Input:
            harrisImage -- numpy array containing the Harris score at
                           each pixel.
        Output:
            destImage -- numpy array containing True/False at
                         each pixel, depending on whether
                         the pixel value is the local maxima in
                         its 7x7 neighborhood.
        '''
        destImage = np.zeros_like(harrisImage, np.bool)

        # TODO 2: Compute the local maxima image
        # TODO-BLOCK-BEGIN
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
        # TODO-BLOCK-END

        return destImage


def detectKeypoints(self, image):

    
        '''
        Input:
            image -- BGR image with values between [0, 255]
        Output:
            list of detected keypoints, fill the cv2.KeyPoint objects with the
            coordinates of the detected keypoints, the angle of the gradient
            (in degrees), the detector response (Harris score for Harris detector)
            and set the size to 10.
        '''
        image = image.astype(np.float32)
        image /= 255.
        height, width = image.shape[:2]
        features = []

        # Create grayscale image used for Harris detection
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # computeHarrisValues() computes the harris score at each pixel
        # position, storing the result in harrisImage.
        # You will need to implement this function.
        harrisImage, orientationImage = self.computeHarrisValues(grayImage)

        # Compute local maxima in the Harris image.  You will need to
        # implement this function. Create image to store local maximum harris
        # values as True, other pixels False
        harrisMaxImage = self.computeLocalMaxima(harrisImage)

        # Loop through feature points in harrisMaxImage and fill in information
        # needed for descriptor computation for each point.
        # You need to fill x, y, and angle.
        for y in range(height):
            for x in range(width):
                if not harrisMaxImage[y, x]:
                    continue

                f = cv2.KeyPoint()

                # TODO 3: Fill in feature f with location and orientation
                # data here. Set f.size to 10, f.pt to the (x,y) coordinate,
                # f.angle to the orientation in degrees and f.response to
                # the Harris score
                # TODO-BLOCK-BEGIN
                f.size = 10
                f.pt = x, y
                f.angle = orientationImage[y, x]
                f.response = harrisImage[y, x]
                # TODO-BLOCK-END

                features.append(f)
        return features

