from skimage.metrics import structural_similarity


def ssim(ref_img, rend_img, data_range=255, multichannel=False):
    mssim = structural_similarity(ref_img, rend_img, data_range=data_range, multichannel=multichannel)
    return mssim


if __name__ == "__main__":
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    from skimage import img_as_float
    from skimage.metrics import mean_squared_error
    img = cv2.imread("../input/Lenna_orig.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    multichannel = True
    if multichannel:
        rows, cols, ch = img.shape
    else:
        rows, cols = img.shape
    img = img_as_float(img)

    noise = np.ones_like(img) * 0.2 * (img.max() - img.min())
    noise[np.random.random(size=noise.shape) > 0.5] *= -1

    img_noise = img + noise
    img_const = img + abs(noise)

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 4),
                             sharex=True, sharey=True)
    ax = axes.ravel()

    mse_none = mean_squared_error(img, img)
    ssim_none = ssim(img, img, data_range=img.max() - img.min(), multichannel=multichannel)

    mse_noise = mean_squared_error(img, img_noise)
    ssim_noise = ssim(img, img_noise,
                      data_range=img_noise.max() - img_noise.min(), multichannel=multichannel)

    mse_const = mean_squared_error(img, img_const)
    ssim_const = ssim(img, img_const,
                      data_range=img_const.max() - img_const.min(), multichannel=multichannel)
    label = 'MSE: {:.2f}, SSIM: {:.2f}'

    ax[0].imshow(img, cmap=plt.cm.gray, vmin=0, vmax=1)
    ax[0].set_xlabel(label.format(mse_none, ssim_none))
    ax[0].set_title('Original image')

    ax[1].imshow(img_noise, cmap=plt.cm.gray, vmin=0, vmax=1)
    ax[1].set_xlabel(label.format(mse_noise, ssim_noise))
    ax[1].set_title('Image with noise')

    ax[2].imshow(img_const, cmap=plt.cm.gray, vmin=0, vmax=1)
    ax[2].set_xlabel(label.format(mse_const, ssim_const))
    ax[2].set_title('Image plus constant')

    plt.tight_layout()
    plt.show()