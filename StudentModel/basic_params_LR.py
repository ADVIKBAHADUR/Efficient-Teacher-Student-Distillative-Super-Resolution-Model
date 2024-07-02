import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import time
import numpy as np
from thop import profile
from torchsummary import summary
import psutil
from skimage.metrics import structural_similarity as ssim
import cv2
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt

process = psutil.Process(os.getpid())


# Enhance interpolation
def enhance_interpolation(image, new_size):
    return image.resize(new_size, Image.LANCZOS)

# Post-filtering
def postfilter_image(image):
    return cv2.GaussianBlur(image, (3, 3), 0)

# Denoising
def denoise_image(image):
    return cv2.fastNlMeansDenoisingColored(image, None, 2, 3, 7, 21)

# Median filtering
def median_filter_image(image):
    return cv2.medianBlur(image, 3)

# Sharpening
def sharpen_image(image):
    kernel = np.array([[0, -1, 0], 
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

# Define your dataset for testing
class TestDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, transform=None):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.lr_images = sorted([f for f in os.listdir(lr_dir) if os.path.isfile(os.path.join(lr_dir, f)) and not f.startswith('.')])
        self.hr_images = sorted([f for f in os.listdir(hr_dir) if os.path.isfile(os.path.join(hr_dir, f)) and not f.startswith('.')])
        self.transform = transform

        print(f"Found {len(self.lr_images)} low-resolution images and {len(self.hr_images)} high-resolution images")

    def __len__(self):
        return len(self.lr_images)

    def __getitem__(self, idx):
        lr_image_path = os.path.join(self.lr_dir, self.lr_images[idx])
        hr_image_path = os.path.join(self.hr_dir, self.hr_images[idx])

        lr_image = Image.open(lr_image_path).convert('RGB')
        hr_image = Image.open(hr_image_path).convert('RGB')

        if self.transform:
            lr_image = self.transform(lr_image)
            hr_image = self.transform(hr_image)

        return lr_image, hr_image

# PSNR calculation
def calculate_psnr(img1, img2):
    mse = nn.MSELoss()(img2, img1)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

# SSIM calculation
def calculate_ssim(img1, img2):
    img1_np = img1.squeeze().permute(1, 2, 0).cpu().numpy()
    img2_np = img2.squeeze().permute(1, 2, 0).cpu().numpy()
    return ssim(img1_np, img2_np, data_range=img2_np.max() - img2_np.min(), channel_axis=2)

# Evaluate the model
def evaluate_model(dataloader, device, num_images_to_eval=25):
    psnr_list = []
    ssim_list = []
    inference_times = []
    max_memory = 0

    with torch.no_grad():
        print("Starting evaluation...")
        for i, (lr_images, hr_images) in enumerate(dataloader):
            if i >= num_images_to_eval:
                break

            lr_images = lr_images.to(device)
            hr_images = hr_images.to(device)

            try:
                # Resize outputs to match hr_images dimensions
                outputs_resized = nn.functional.interpolate(lr_images, size=hr_images.shape[-2:], mode='bicubic', align_corners=False)
                
                # Convert tensor to numpy for post-processing
                outputs_np = outputs_resized.squeeze().permute(1, 2, 0).cpu().numpy()
                outputs_np = (outputs_np * 255).astype(np.uint8)
                outputs_np = cv2.cvtColor(outputs_np, cv2.COLOR_RGB2BGR)
                
                # Apply filters
                postfiltered_image = postfilter_image(outputs_np)
                denoised_image = denoise_image(postfiltered_image)
                sharpened_image = sharpen_image(denoised_image)
                
                # Convert back to tensor for PSNR and SSIM calculation
                processed_image = cv2.cvtColor(sharpened_image, cv2.COLOR_BGR2RGB)
                processed_image = (processed_image / 255.0).astype(np.float32)
                processed_image = torch.tensor(processed_image).permute(2, 0, 1).unsqueeze(0).to(device)

                # cv2.imshow('processed_image', processed_image)
                
                # Calculate PSNR
                psnr = calculate_psnr(processed_image, hr_images)
                psnr_list.append(psnr.item())

                # Calculate SSIM
                ssim_index = calculate_ssim(hr_images, processed_image)
                ssim_list.append(ssim_index)

                                # Display the images using matplotlib
                lr_image_np = lr_images.squeeze().permute(1, 2, 0).cpu().numpy()
                lr_image_np = (lr_image_np * 255).astype(np.uint8)
                
                hr_image_np = hr_images.squeeze().permute(1, 2, 0).cpu().numpy()
                hr_image_np = (hr_image_np * 255).astype(np.uint8)
                
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                axes[0].imshow(lr_image_np)
                axes[0].set_title("Low-Resolution Image")
                axes[0].axis("off")
                
                axes[1].imshow(hr_image_np)
                axes[1].set_title("High-Resolution Image")
                axes[1].axis("off")
                
                axes[2].imshow(cv2.cvtColor(sharpened_image, cv2.COLOR_BGR2RGB))
                axes[2].set_title("Processed Image")
                axes[2].axis("off")
                
                plt.show()

                # Measure GPU memory usage
                if torch.cuda.is_available():
                    max_memory = max(max_memory, torch.cuda.max_memory_allocated(device) / 1024 ** 2)  # in MB
                else:
                    max_memory = process.memory_info().rss / 1024 ** 2

            except Exception as e:
                print(f"Error during evaluation: {e}")

    avg_psnr = np.mean(psnr_list) if psnr_list else float('nan')
    avg_inference_time = np.mean(inference_times) if inference_times else float('nan')
    avg_ssim = np.mean(ssim_list) if ssim_list else float('nan')
    return avg_inference_time, avg_psnr, max_memory, avg_ssim

# Paths
lr_dir = 'G://My Drive//SuperResolution//SuperResolution//Data//DIV2K_train_LR_x8//DIV2K_train_LR_x8'
hr_dir = 'G://My Drive//SuperResolution//SuperResolution//Data//DIV2K_train_HR'
# Dataset and DataLoader
transform = transforms.Compose([transforms.ToTensor()])
test_dataset = TestDataset(lr_dir=lr_dir, hr_dir=hr_dir, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

# Evaluate the models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

avg_inference_time_student, avg_psnr_student, max_memory_student, student_ssim = evaluate_model(test_dataloader, device)
# avg_inference_time_teacher, avg_psnr_teacher, max_memory_teacher, teacher_ssim = evaluate_model(teacher_model, test_dataloader, device)


print(f'Average Inference Time: {avg_inference_time_student:.2f} ms')
print(f'Average PSNR: {avg_psnr_student:.2f} dB')
print(f'Average SSIM: {student_ssim}')
print(f'GPU Memory Usage: {max_memory_student:.2f} MB')

