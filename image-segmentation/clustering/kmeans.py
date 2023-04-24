import sys

from matplotlib import pyplot as plt
sys.path.insert(1, "..")
import img_utils as utils
from sklearn.cluster import KMeans

if __name__ == "__main__":
    print("This program is meant to segment an example image with the help of kmeans clustering")
    eg_image_path = "../waterworld_imgs/example.png"
    img = utils.load_img_and_convert_to_three_channels(eg_image_path)
    img_2d = img.reshape(img.shape[0] * img.shape[1], img.shape[2])
    kmeans = KMeans(n_clusters=8, random_state=0).fit(img_2d)
    to_show = kmeans.cluster_centers_[kmeans.labels_]
    cluster_pic = to_show.reshape(img.shape[0], img.shape[1], img.shape[2])
    plt.imshow(cluster_pic)
    plt.show()

