import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# Using pathlib to handle the image path and gather all .png files
image_files = sorted(Path("data/img").glob('*.png'))  # List all .png images

# Loop through all the images
for image_path in image_files:
    # Load the image (using OpenCV)
    img = cv2.imread(str(image_path))  # Convert the Path object to string for OpenCV
    if img is None:
        print(f"Error: Unable to read the image file {image_path}")
        continue
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB for Matplotlib

    # Construct the corresponding label file path
    label_file = Path("data/label") / f"{image_path.stem}.txt"

    # Check if the label file exists
    if not label_file.is_file():
        print(f"Error: Label file for {image_path.name} not found.")
        continue

    # Read the label file
    with open(label_file, 'r') as file:
        labels = file.readlines()

    # Image dimensions
    img_height, img_width, _ = img.shape

    # Loop through each label to draw bounding boxes
    for label in labels:
        label_parts = label.split()

        # Extract label information
        class_name = label_parts[0]
        x_center, y_center = float(label_parts[4]), float(label_parts[5])  # Center of the box (in normalized coordinates)
        length, width, height = float(label_parts[7]), float(label_parts[8]), float(label_parts[9])  # Size of the object

        # Convert from normalized coordinates to pixel coordinates
        x_center = int(x_center * img_width)  # x_center in pixels
        y_center = int(y_center * img_height)  # y_center in pixels
        length = int(length * img_width)  # Length of the object in pixels
        width = int(width * img_height)  # Width of the object in pixels

        # Calculate the bounding box coordinates (top-left and bottom-right)
        x1 = int(x_center - length / 2)
        y1 = int(y_center - width / 2)
        x2 = int(x_center + length / 2)
        y2 = int(y_center + width / 2)

        # Ensure the coordinates stay within image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_width - 1, x2)
        y2 = min(img_height - 1, y2)

        # Draw the bounding box on the image
        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Draw a red rectangle

        # Optionally add the class label text on the box
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img_rgb, class_name, (x1, y1 - 10), font, 0.5, (255, 0, 0), 2)

    # Display the image with bounding boxes
    plt.imshow(img_rgb)
    plt.title(f"Image with Bounding Boxes: {image_path.name}")
    plt.axis("off")  # Hide axes
    plt.show()
