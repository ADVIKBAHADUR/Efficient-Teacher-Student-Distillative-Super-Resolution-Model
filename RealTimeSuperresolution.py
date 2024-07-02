import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
from PIL import Image, ImageFilter
import time

# Pre-filtering
def prefilter_image(image):
    return cv2.GaussianBlur(image, (3, 3), 0)

# Enhance interpolation
def enhance_interpolation(image, new_size):
    return image.resize(new_size, Image.LANCZOS)

# Post-filtering
def postfilter_image(image):
    return cv2.GaussianBlur(image, (3, 3), 0)

# Denoising
def denoise_image(image):
    return cv2.fastNlMeansDenoisingColored(image, None, 3, 3, 7, 21)

# Median filtering
def median_filter_image(image):
    return cv2.medianBlur(image, 3)

# Sharpening
def sharpen_image(image):
    kernel = np.array([[0, -1, 0], 
                       [-1, 5,-1],
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

# Define your super-resolution model (example model structure)
class StudentModel(nn.Module):
    def __init__(self, scale_factor=4):
        super(StudentModel, self).__init__()
        self.upsample = nn.Sequential(
            nn.Conv2d(3, 3 * (scale_factor ** 2), kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(scale_factor)
        )
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.relu(x)
        return x

# Load the pre-trained model
model_path = "G://My Drive//SuperResolution//SuperResolution//ESRGAN//models//SupervisedLossOnFeatureMaps_v2.pth"
model = StudentModel()
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Transform to convert PIL image to tensor
transform = transforms.ToTensor()

# Downscale function
def downscale_image(image, scale_factor=0.25):
    width, height = image.size
    new_size = (int(width * scale_factor), int(height * scale_factor))
    return image.resize(new_size, Image.BICUBIC)

# Super-resolve function
def super_resolve_frame(model, frame):
    # Convert frame to PIL image
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Downscale image
    lr_image = downscale_image(pil_image)
    #lr_image_pre = lr_image.filter(ImageFilter.GaussianBlur(radius=0.1))
    
    # Convert to tensor and add batch dimension
    lr_tensor = transform(lr_image).unsqueeze(0)
    
    # Super-resolve
    start_time = time.time()
    with torch.no_grad():
        sr_tensor = model(lr_tensor)
    inference_time = (time.time() - start_time) * 1000  # Convert to milliseconds
    # Remove batch dimension and convert to numpy array
    sr_image = sr_tensor.squeeze().permute(1, 2, 0).numpy()
    
    # Clip values to [0, 1] range
    sr_image = np.clip(sr_image, 0, 1)
    
    # Convert to uint8 and BGR format for OpenCV
    sr_image = (sr_image * 255).astype(np.uint8)
    sr_image = cv2.cvtColor(sr_image, cv2.COLOR_RGB2BGR)

    # Post-process the image
    postfiltered_image = postfilter_image(sr_image)
    denoised_image = denoise_image(postfiltered_image)
    sharpened_image = sharpen_image(denoised_image)
    
    # Convert low-resolution image to BGR format for display
    lr_image = cv2.cvtColor(np.array(lr_image), cv2.COLOR_RGB2BGR)
    
    return lr_image, sr_image, sharpened_image, inference_time

# Initialize webcam
cap = cv2.VideoCapture(0)

# Set target FPS
target_fps = 24
frame_time = 1 / target_fps

# FPS calculation variables
fps_start_time = time.time()
fps = 0
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Crop the frame to 256x256
    frame_cropped = frame[:480, :480]

    # Super-resolve the cropped frame
    lr_frame, sr_frame, post_processed_image, inference_time = super_resolve_frame(model, frame_cropped)
    
    # Resize the low-resolution image back to 256x256
    lr_frame_resized = cv2.resize(lr_frame, (480, 480))
    
    # Resize the super-resolved image to 256x256
    sr_frame_resized = cv2.resize(sr_frame, (480, 480))

    post_process_sized = cv2.resize(post_processed_image, (480, 480))
    
    # Concatenate original, downscaled, and super-resolved frames horizontally
    concatenated_frame = np.hstack((frame_cropped, lr_frame_resized, sr_frame_resized, post_process_sized))

    # Calculate FPS
    frame_count += 1
    if time.time() - fps_start_time >= 1.0:
        fps = frame_count
        frame_count = 0
        fps_start_time = time.time()
    
    # Add FPS and inference time overlay to the super-resolved frame
    cv2.putText(concatenated_frame, f'FPS: {fps}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(concatenated_frame, f'Inference Time: {inference_time:.2f} ms', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    
    # Display the concatenated frame
    cv2.imshow('Original -> Downscaled -> Super-Resolution -> Post Processing', concatenated_frame)

    # Wait to maintain target FPS
    if cv2.waitKey(int(frame_time * 1000)) & 0xFF == ord('q'):
        break

# Release webcam and close windows
cap.release()
cv2.destroyAllWindows()
