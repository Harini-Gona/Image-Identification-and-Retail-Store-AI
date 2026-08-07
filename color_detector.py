import cv2
import numpy as np
from sklearn.cluster import KMeans


def get_dominant_color(image_path, k=3):
    image = cv2.imread(image_path)

    if image is None:
        return "#888888"

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.reshape((-1, 3))

    # reduce size for speed
    image = image[np.random.choice(image.shape[0], 10000, replace=False)]

    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(image)

    colors = kmeans.cluster_centers_
    counts = np.bincount(kmeans.labels_)

    dominant = colors[np.argmax(counts)]

    r, g, b = dominant
    return "#{:02x}{:02x}{:02x}".format(int(r), int(g), int(b))