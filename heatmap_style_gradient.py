from PIL import Image, ImageFilter
import numpy as np
from sklearn.cluster import KMeans
import random

def extract_colors(image_path, num_colors=4):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((150, 150))
    img_np = np.array(img).reshape(-1, 3)

    kmeans = KMeans(n_clusters=num_colors, n_init=10)
    kmeans.fit(img_np)
    colors = kmeans.cluster_centers_.astype(int)
    return colors

def create_liquid_heatmap(colors, width=800, height=800, blur_radius=50, swirl_strength=0.6):
    canvas = np.zeros((height, width, 3), dtype=float)

    # Place random blobs of color on the canvas
    for color in colors:
        center_x = random.randint(100, width - 100)
        center_y = random.randint(100, height - 100)
        radius = random.randint(150, 300)

        y, x = np.ogrid[:height, :width]
        distance = (x - center_x)**2 + (y - center_y)**2
        blob = np.exp(-distance / (2 * (radius ** 2))).reshape(height, width, 1)
        canvas += blob * color

    # Normalize
    canvas = np.clip(canvas, 0, 255).astype(np.uint8)

    # Convert to PIL and apply heavy blur
    img = Image.fromarray(canvas)
    img = img.filter(ImageFilter.GaussianBlur(blur_radius))

    return img

# Example usage
image_path = "/Users/esteerutten/Downloads/Headache (sped up) - Asal.jpg"  # Replace with your own
colors = extract_colors(image_path, num_colors=4)
img = create_liquid_heatmap(colors)
img.save("liquid_heatmap.png")
img.show()

