from transformers import pipeline
from PIL import Image
import requests

class DepthEstimation:
    def __init__(self, model_name="depth-anything/Depth-Anything-V2-Small-hf"):
        self.pipe = pipeline(task="depth-estimation", model=model_name)

    def estimate_depth(self, image):
        return self.pipe(image)["depth"]
    
if __name__ == "__main__":
    # Example usage
    image_path = "/Users/ducle/Library/CloudStorage/GoogleDrive-leduchtth@gmail.com/My Drive/My_learning/my_works/cobot/dji/capture/frame_30.jpg"  # Replace with your image path
    image = Image.open(image_path)

    depth_estimator = DepthEstimation()
    depth_map = depth_estimator.estimate_depth(image)

    # Save or display the depth map
    depth_map.save("depth_map.png")
    print("Depth estimation completed and saved as 'depth_map.png'.")
