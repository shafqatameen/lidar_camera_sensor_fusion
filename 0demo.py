import matplotlib.pyplot as plt
import cv2
import glob

# Assuming images are stored in a list
image_files = sorted(glob.glob("data/img/*.png"))  # Get image paths
images = [cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB) for img in image_files[:100]]  # Load 100 images

# Create a 10x10 grid of subplots
fig, axes = plt.subplots(10, 10, figsize=(20, 20))

# Loop through each axis and display an image
for i, ax in enumerate(axes.flat):  # Flatten axes for easy indexing
    ax.imshow(images[i])
    ax.axis("off")  # Hide axis

plt.tight_layout()  # Adjust spacing
plt.show()
