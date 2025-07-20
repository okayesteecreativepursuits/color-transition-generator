from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from color_extractor import extract_colors
from tqdm import tqdm


colors = extract_colors("/Users/esteerutten/Downloads/ -58.jpg", num_colors=3)

def create_gradient(colors, width=300, height=300):
    print("Starting gradient creation...")
    n = len(colors)
    gradient = np.zeros((height, width, 3), dtype=np.uint8)

    # Compute the positions along the width for each segment
    segment_positions = np.linspace(0, width, n, dtype=int)

    for i in tqdm(range(n - 1), desc="Creating gradient segments"):
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
