import cv2
import os
import matplotlib.pyplot as plt
from pathlib import Path

# Paths to image and label directories
img_dir = "data/img/"
label_dir = "data/label/"

# Get all image filenames
#image_files = [f for f in os.listdir(img_dir) if f.endswith((".png", ".jpg", ".jpeg"))]
#image_files=list(sorted(Path("data/img/").glob("*.png")))
image_files = [str(f.name) for f in sorted(Path("data/img/").glob("*.png"))]

for img_filename in image_files:
    print(img_filename)

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


'''import cv2
import matplotlib.pyplot as plt

# Load the image
img_path = 'data/img/000031.png'  # Specify your image path
img = cv2.imread(img_path)

# Check if the image is loaded correctly
if img is None:
    print(f"Error: Unable to read the image file at {img_path}")
else:
    print(f"Image loaded successfully with shape: {img.shape}")

    # Convert the image to RGB for Matplotlib (as OpenCV uses BGR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print("Image converted to RGB for Matplotlib")

    # Label data as provided (normally this would come from a label file)
    label_data = """
    Car 0.00 0 -1.35 334.28 180.65 490.02 297.48 1.65 1.67 3.81 -3.18 1.79 12.20 -1.60
    Car 0.00 0 -1.93 785.85 179.67 1028.82 340.75 1.52 1.51 3.10 3.18 1.61 8.46 -1.59
    Car 0.00 1 -1.80 711.98 179.65 848.82 277.62 1.53 1.58 3.53 2.90 1.66 13.14 -1.59
    Truck 0.00 2 1.71 445.91 131.69 539.77 228.36 2.60 2.06 5.42 -3.48 1.52 22.27 1.56
    Car 0.00 0 -1.64 660.82 178.72 713.09 222.03 1.46 1.60 3.71 2.71 1.69 26.41 -1.54
    Van 0.00 0 -1.62 620.95 171.59 649.36 200.46 1.97 1.82 4.41 1.75 1.91 51.90 -1.58
    DontCare -1 -1 -10 547.66 172.93 591.26 194.02 -1 -1 -1 -1000 -1000 -1000 -10
    DontCare -1 -1 -10 603.92 176.25 622.75 191.58 -1 -1 -1 -1000 -1000 -1000 -10
    DontCare -1 -1 -10 1035.60 236.48 1240.00 373.00 -1 -1 -1 -1000 -1000 -1000 -10
    """
    
    # Image dimensions
    img_height, img_width, _ = img.shape
    print(f"Image dimensions: Height={img_height}, Width={img_width}")

    # Process each label line
    for label in label_data.strip().split("\n"):
        label_parts = label.split()
        
        # Debugging check: print label data
        print(f"Processing label: {label_parts}")

        # Extract label information
        class_name = label_parts[0]
        
        # Skip 'DontCare' class
        if class_name == "DontCare":
            print("Skipping 'DontCare' label")
            continue  

        # Extract bounding box coordinates directly from the label
        try:
            x1, y1 = float(label_parts[4]), float(label_parts[5])
            x2, y2 = float(label_parts[6]), float(label_parts[7])

            # Convert to integer pixel coordinates
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Ensure coordinates are within image bounds
            x1 = max(0, min(x1, img_width - 1))
            y1 = max(0, min(y1, img_height - 1))
            x2 = max(0, min(x2, img_width - 1))
            y2 = max(0, min(y2, img_height - 1))

            # Debugging check: print final bounding box coordinates
            print(f"Bounding box coordinates: (x1, y1)=({x1}, {y1}), (x2, y2)=({x2}, {y2})")

            # Draw the bounding box on the image
            cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red bounding box

            # Optionally add the class label text on the box
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img_rgb, class_name, (x1, y1 - 10), font, 0.5, (0, 0, 255), 2)

        except ValueError as e:
            print(f"Error processing label data: {e}")
            continue

    # Display the image with bounding boxes
    plt.imshow(img_rgb)
    plt.title("Image with Bounding Boxes")
    plt.axis("off")  # Hide axes
    plt.show()

'''