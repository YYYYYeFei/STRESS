import torch
import numpy as np
from scipy.stats import pearsonr
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image import PeakSignalNoiseRatio
from audtorch.metrics.functional import pearsonr

def cal_rmse(y,y_hat):
    # y= max_min(y)
    # y_hat = max_min(y_hat)
    return torch.sqrt(torch.mean((y-y_hat)**2))

# def normalize(a):
#     mean_a = torch.mean(a)
#     std_a = torch.std(a)
#     n1 = (a - mean_a) / std_a
#     print('max:{},min:{}'.format(torch.max(n1),torch.min(n1)))
#     return n1
def max_min(x):
    min_vals = torch.min(x)
    max_vals = torch.max(x)

    # 最小-最大缩放，将x的范围缩放到[0, 1]
    n1 = (x - min_vals) / (max_vals - min_vals)

    # print('max:{},min:{}'.format(torch.max(n1),torch.min(n1)))
    return n1

# def cal_psnr(img, img2, **kwargs):
#     """Calculate PSNR (Peak Signal-to-Noise Ratio).
#
#     Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
#
#     Args:
#         img (ndarray): Images with range [0, 255].
#         img2 (ndarray): Images with range [0, 255].
#
#
#     Returns:
#         float: psnr result.
#     """
#     img = max_min(img)
#     img2 = max_min(img2)
#     mse = torch.mean((img - img2)**2)
#     if mse == 0:
#         return float('inf')
#     # print('mse:',mse)
#     return 20. * torch.log10(1. / torch.sqrt(mse))
# def _ssim(img, img2):
#     """Calculate SSIM (structural similarity) for one channel images.
#
#     It is called by func:`calculate_ssim`.
#
#     Args:
#         img (ndarray): Images with range [0, 255] with order 'HWC'.
#         img2 (ndarray): Images with range [0, 255] with order 'HWC'.
#
#     Returns:
#         float: ssim result.
#     """
#
#     c1 = (0.01 * 255)**2
#     c2 = (0.03 * 255)**2
#
#     img = img.astype(np.float64)
#     img2 = img2.astype(np.float64)
#     kernel = cv2.getGaussianKernel(11, 1.5)
#     window = np.outer(kernel, kernel.transpose())
#
#     mu1 = cv2.filter2D(img, -1, window)[5:-5, 5:-5]
#     mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
#     mu1_sq = mu1**2
#     mu2_sq = mu2**2
#     mu1_mu2 = mu1 * mu2
#     sigma1_sq = cv2.filter2D(img**2, -1, window)[5:-5, 5:-5] - mu1_sq
#     sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
#     sigma12 = cv2.filter2D(img * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
#
#     ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
#
#     return ssim_map.mean()
# def cal_ssim(img, img2):
#     """Calculate SSIM (structural similarity).
#
#     Ref:
#     Image quality assessment: From error visibility to structural similarity
#
#     The results are the same as that of the official released MATLAB code in
#     https://ece.uwaterloo.ca/~z70wang/research/ssim/.
#
#     For three-channel images, SSIM is calculated for each channel and then
#     averaged.
#
#     Args:
#         img (ndarray): Images with range [0, 255].
#         img2 (ndarray): Images with range [0, 255].
#     Returns:
#         float: ssim result.
#     """
#     img = max_min(img).numpy()
#     img2 = max_min(img2).numpy()
#     ssims = []
#     for i in range(img.shape[2]):
#         ssims.append(_ssim(img[..., i], img2[..., i]))
#     return np.array(ssims).mean()

_ = torch.manual_seed(123)


def psnr_torch(img1, img2):
    mse = ((img1 - img2) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def cal_psnr(img1, img2):
    img1= max_min(img1)
    img2 = max_min(img2)
    metric = PeakSignalNoiseRatio()
    return metric(img1, img2)

def cal_ssim(img1, img2):
    img1= max_min(img1)
    img2 = max_min(img2)
    metric = StructuralSimilarityIndexMeasure(data_range=1.0)
    return metric(img1, img2)

def cal_pcc(img1, img2):
    flatten_img1=img1.flatten()
    flatten_img2 = img2.flatten()
    pcc = pearsonr(flatten_img1, flatten_img2)
    # pcc = pearsonr(img1, img2)
    # pcc=torch.corrcoef(flatten_img1,flatten_img2)
    return pcc

def cal_pcc01(img1, img2):
    img1=img1.flatten()
    img2=img2.flatten()
    corr = np.corrcoef(img1, img2)[0, 1]
    return corr

if __name__ == '__main__':
    img = torch.randn(1,2000,64,64)
    img2 = torch.randn(1,2000, 64, 64)
    results_rmse=rmse(img,img2)
    print('rmse:', results_rmse)
    results_psnr=psnr(img,img2)
    print('psnr:', results_psnr)
    results_ssim = ssim(img, img2)
    print('ssim:',results_ssim)


