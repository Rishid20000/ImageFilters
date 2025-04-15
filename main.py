import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import generic_filter
import os

# Create output folders
output_dir = 'output_plots'
os.makedirs(output_dir, exist_ok=True)

# Load and convert the image to RGB
img = cv2.imread('input.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# Filters
gaussian = cv2.GaussianBlur(img, (5, 5), 0)
median = cv2.medianBlur(img, 5)
bilateral = cv2.bilateralFilter(img, 9, 75, 75)
edges_canny = cv2.Canny(gray, 100, 200)
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)

def ordered_dithering(gray_img):
    norm_img = gray_img / 255.0
    bayer_matrix = (1/17) * np.array([
        [0, 8, 2, 10], [12, 4, 14, 6],
        [3, 11, 1, 9], [15, 7, 13, 5]
    ])
    h, w = gray_img.shape
    dithered = np.zeros_like(gray_img)
    for y in range(h):
        for x in range(w):
            threshold = bayer_matrix[y % 4][x % 4]
            dithered[y, x] = 255 if norm_img[y, x] > threshold else 0
    return dithered

def harmonic_filter(img, size=3):
    def harmonic_func(pixels):
        pixels = np.array(pixels)
        pixels = pixels[pixels != 0]
        return len(pixels) / np.sum(1.0 / pixels) if pixels.size > 0 else 0
    return generic_filter(img.astype(np.float32), harmonic_func, size=(size, size)).astype(np.uint8)

def min_filter(img, size=3):
    return cv2.erode(img, np.ones((size, size), np.uint8))

def max_filter(img, size=3):
    return cv2.dilate(img, np.ones((size, size), np.uint8))

def avg_min_max_filter(img, size=3):
    min_f = min_filter(img, size)
    max_f = max_filter(img, size)
    return ((min_f.astype(np.uint16) + max_f.astype(np.uint16)) // 2).astype(np.uint8)

# Apply filters
dithered_img = ordered_dithering(gray)
harmonic = harmonic_filter(gray)
minf = min_filter(gray)
maxf = max_filter(gray)
avg_min_max = avg_min_max_filter(gray)

# Save plots
results = {
    "Original": img,
    "Gaussian Blur": gaussian,
    "Median Blur": median,
    "Bilateral Filter": bilateral,
    "Canny Edges": edges_canny,
    "Sobel Edges": sobelx + sobely,
    "Grayscale": gray,
    "Ordered Dithering": dithered_img,
    "Harmonic Filter": harmonic,
    "Min Filter": minf,
    "Max Filter": maxf,
    "Avg(Max, Min) Filter": avg_min_max,
}

html_content = '<html><head><title>Image Processing Output</title></head><body>'
html_content += '<div style="height:95vh;overflow-y:scroll;">'

for i, (title, image) in enumerate(results.items()):
    plt.figure()
    cmap = 'gray' if len(image.shape) == 2 else None
    plt.imshow(image, cmap=cmap)
    plt.axis('off')
    plt.title(title)
    fname = f"{output_dir}/{i}_{title.replace(' ', '_')}.png"
    plt.savefig(fname, bbox_inches='tight')
    plt.close()
    html_content += f'<div style="margin:20px;"><h3>{title}</h3><img src="{fname}" style="max-width:100%;border:1px solid #ccc;"></div>'

html_content += '</div></body></html>'

# Save HTML
html_path = os.path.join(output_dir, "results.html")
with open(html_path, "w") as f:
    f.write(html_content)

print(f"\n✅ Output saved! Open this file in your browser:\n{html_path}")
