from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def extract_colors(image_path, num_colors):
    img = Image.open(image_path)
    img = img.resize((150, 150)) #resize to reduce processing
    img_np = np.array(img)
    img_np = img_np.reshape (-1,3)

    kmeans = KMeans(n_clusters=num_colors, n_init=10)
    kmeans.fit(img_np)

    colors = kmeans.cluster_centers_.astype(int)
    return colors


def show_colors(colors):
    plt.figure(figsize=(8,2))
    for i, color in enumerate(colors):
        plt.subplot(1, len(colors), i + 1)
        plt.imshow([[color / 255]])
        plt.axis('off')
    plt.show(block=False)
    plt.pause(10)  # seconds
    plt.close()

colors = extract_colors("/Users/esteerutten/Downloads/ -58.jpg", num_colors=10)
show_colors(colors)
print ("Hello")


