import cv2
import os
import matplotlib.pyplot as plt

# Paths to image and label directories
img_dir = "data/img/"
label_dir = "data/label/"

# Get all image filenames
image_files = [f for f in os.listdir(img_dir) if f.endswith((".png", ".jpg", ".jpeg"))]

# Loop through all images
for img_filename in image_files:
    # Construct full image path
    img_path = os.path.join(img_dir, img_filename)
    
    # Read image
    img = cv2.imread(img_path)

    # Check if the image is loaded correctly
    if img is None:
        print(f"Error: Unable to read the image file at {img_path}")
        continue

    # Convert image to RGB (for Matplotlib)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Construct label file path
    label_filename = os.path.splitext(img_filename)[0] + ".txt"
    label_path = os.path.join(label_dir, label_filename)

    # Check if the label file exists
    if not os.path.exists(label_path):
        print(f"Warning: Label file missing for {img_filename}")
        continue

    # Read label file
    with open(label_path, "r") as file:
        label_lines = file.readlines()

    # Get image dimensions
    img_height, img_width, _ = img.shape

    # Process each line in the label file
    for label in label_lines:
        label_parts = label.split()

        # Skip empty lines
        if len(label_parts) < 8:
            continue

        class_name = label_parts[0]

        # Skip 'DontCare' class
        if class_name == "DontCare":
            continue  

        try:
            # Extract bounding box coordinates
            x1, y1 = float(label_parts[4]), float(label_parts[5])
            x2, y2 = float(label_parts[6]), float(label_parts[7])

            # Convert to integer pixel coordinates
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Ensure coordinates are within image bounds
            x1 = max(0, min(x1, img_width - 1))
            y1 = max(0, min(y1, img_height - 1))
            x2 = max(0, min(x2, img_width - 1))
            y2 = max(0, min(y2, img_height - 1))

            # Draw bounding box
            cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red box

            # Add class label text
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img_rgb, class_name, (x1, y1 - 10), font, 0.5, (0, 0, 255), 2)

        except ValueError as e:
            print(f"Error processing label data for {img_filename}: {e}")
            continue

    # Display the image with bounding boxes
    plt.figure(figsize=(8, 6))
    plt.imshow(img_rgb)
    plt.title(f"Image: {img_filename}")
    plt.axis("off")  # Hide axes
    plt.show()
