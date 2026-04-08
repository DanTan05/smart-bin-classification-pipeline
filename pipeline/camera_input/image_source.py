import cv2
import os
import shutil
import time


class CameraCapture:
    """Captures a single frame from the Raspberry Pi camera using OpenCV."""

    def __init__(self, save_folder="captured_images", device_index=0, warmup_seconds=1):
        self.save_folder = save_folder
        self.device_index = device_index
        self.warmup_seconds = warmup_seconds
        os.makedirs(self.save_folder, exist_ok=True)

    def capture(self):
        """
        Opens the camera, captures one frame, saves it to disk, and returns
        (image_bgr, saved_path). Raises RuntimeError if the camera fails.
        """
        cap = cv2.VideoCapture(self.device_index)
        if not cap.isOpened():
            raise RuntimeError(
                f"Camera index {self.device_index} could not be opened. "
                "Check the USB/CSI connection."
            )

        try:
            # Warm up — let auto-exposure settle
            time.sleep(self.warmup_seconds)

            ret, frame = cap.read()
            if not ret or frame is None:
                raise RuntimeError("Camera opened but failed to read a frame.")

            # Resize to the model's expected input size
            frame_resized = cv2.resize(frame, (224, 224))

            # Save full-resolution frame for logging/audit
            filename = f"capture_{int(time.time())}.jpg"
            save_path = os.path.join(self.save_folder, filename)
            cv2.imwrite(save_path, frame)

            print(f"  Camera: captured {frame.shape[1]}x{frame.shape[0]} → saved to {save_path}")
            return frame, save_path

        finally:
            cap.release()


class ImageSource:
    def __init__(self, image_folder, processed_folder="processed_images"):
        self.image_folder = image_folder
        self.processed_folder = processed_folder

        # Create processed folder if it doesn't exist
        os.makedirs(self.processed_folder, exist_ok=True)

    def get_latest_image(self):
        # Get all image files in the folder
        files = [
            os.path.join(self.image_folder, f)
            for f in os.listdir(self.image_folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if not files:
            raise ValueError("No images found in the image folder")

        # Pick the latest by modification time
        latest_file = max(files, key=os.path.getmtime)

        # Load the image
        image = cv2.imread(latest_file)
        if image is None:
            raise ValueError(f"Failed to load image: {latest_file}")

        # Move to processed folder so it won't be used again
        dest_path = os.path.join(self.processed_folder, os.path.basename(latest_file))
        shutil.move(latest_file, dest_path)

        return image, latest_file
