import cv2 as cv
from pathlib import Path

# Get all PCD files from the velodyne directory
img_files = sorted(Path("data/img").glob("*.png"))

# Check if files exist
if not img_files:
    print("No .png files found in 'data/img/'")
else:

    for file in img_files:
       img=cv.imread(str(file))
       print(f"Showing: {file.name}")
       cv.imshow("Image",img)
       cv.waitKey(1100)
       cv.destroyAllWindows()

