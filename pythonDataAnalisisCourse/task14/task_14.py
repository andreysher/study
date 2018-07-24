from sklearn.cluster import KMeans
from skimage.io import imread, imsave
from skimage import img_as_float
import numpy as np
import math
import pandas as pd

# Загрузите картинку parrots.jpg. Преобразуйте изображение, приведя все значения в интервал
# от 0 до 1. Для этого можно воспользоваться функцией img_as_float из модуля skimage.
image = imread('parrots.jpg')
flt_image = img_as_float(image)

# 2. Создайте матрицу объекты-признаки: характеризуйте каждый пик-
# сель тремя координатами - значениями интенсивности в простран-
# стве RGB.
w, h, d = flt_image.shape
X = np.reshape(flt_image, (w * h, d))
pixels = pd.DataFrame(X, columns=['R', 'G', 'B'])
# 3. Запустите алгоритм K-Means с параметрами init=’k-means++’ и
# random_state=241. После выделения кластеров все пиксели, отне-
# сенные в один кластер, попробуйте заполнить двумя способами:
# медианным и средним цветом по кластеру.
def clustering(pixels, n_clusters=8):
    pixels = pixels.copy()
    model = KMeans(random_state=241, init='k-means++', n_clusters=n_clusters)
    cluster_numbers = model.fit_predict(X)
    pixels['cluster'] = cluster_numbers

    mean = pixels.groupby('cluster').mean().values
    mean_pix = [mean[c] for c in pixels['cluster'].values]
    mean_img = np.reshape(mean_pix, (w, h, d))
    imsave(str(n_clusters) + 'mean.jpg', mean_img)

    median = pixels.groupby('cluster').median().values
    med_pix = [median[c] for c in pixels['cluster'].values]
    med_img = np.reshape(med_pix, (w, h, d))
    imsave(str(n_clusters) + 'median.jpg', med_img)

    return mean_img, med_img

# 4. Измерьте качество получившейся сегментации с помощью метрики
# PSNR. Эту метрику нужно реализовать самостоятельно (см. опре-
# деление).
def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    return 10 * math.log10(1.0 / mse)




# 5. Найдите минимальное количество кластеров, при котором значение PSNR выше 20 (можно рассмотреть
# не более 20кластеров). Это число и будет ответом в данной задаче.
for clusters in range(1,21):
    mean_img, med_img = clustering(pixels, n_clusters=clusters)
    psnr_mean = psnr(flt_image, mean_img)
    psnr_median = psnr(flt_image, med_img)
    if psnr_mean > 20 or psnr_median > 20:
        print(clusters)
        break