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

colors = extract_colors("/Users/esteerutten/Downloads/ -60.jpg", num_colors=10)
show_colors(colors)
print ("Hello")

def create_gradient(colors, width=300, height=300):
    print("Starting gradient creation...")
    n = len(colors)
    gradient = np.zeros((height, width, 3), dtype=np.uint8)

    # Compute the positions along the width for each segment
    segment_positions = np.linspace(0, width, n, dtype=int)

    for i in range(n - 1):
        start_pos = segment_positions[i]
        end_pos = segment_positions[i + 1]
        segment_length = end_pos - start_pos

        start_color = np.array(colors[i], dtype=float)
        end_color = np.array(colors[i + 1], dtype=float)

        # Create linear interpolation ratios for the segment
        ratios = np.linspace(0, 1, segment_length).reshape(-1, 1)  # shape (segment_length, 1)

        # Interpolate colors for the segment
        segment_colors = (1 - ratios) * start_color + ratios * end_color  # shape (segment_length, 3)

        # Broadcast colors vertically for the height
        gradient[:, start_pos:end_pos, :] = segment_colors.astype(np.uint8)

    img = Image.fromarray(gradient)
    img.save("color_transition.png")
    img.show()


# Example
create_gradient(colors)
