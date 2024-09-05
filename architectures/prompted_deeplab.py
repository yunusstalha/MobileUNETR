
# The code structure is inpisred by the following repo:
# https://github.com/OSUPCVLab/MobileUNETR

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50
from torchvision.models.segmentation import deeplabv3_resnet50

class MaskEncoder(nn.Module):
    def __init__(self, input_channels=1):
        super(MaskEncoder, self).__init__()
        
        # Load a pretrained ResNet18 model
        self.resnet = resnet18(pretrained=True)
        
        # Replace the first conv layer for the mask input
        self.resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Remove the final FC layer and avgpool
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])
        
        # Add a final conv layer to match DeepLabV3+ encoder output dimensions (2048 channels)
        self.final_conv = nn.Conv2d(512, 2048, kernel_size=1)

        # Upsample the output to match the other dimensions
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)


    def forward(self, x):
        x = self.resnet(x)
        x = self.final_conv(x)
        x = self.upsample(x)
        return x
    
class PromptedDeepLabV3(nn.Module):
    def __init__(self, num_classes=1, mode='masked_sum'):
        super(PromptedDeepLabV3, self).__init__()
        
        # Load pretrained DeepLabV3+ with ResNet50 backbone
        self.deeplab = deeplabv3_resnet50(pretrained=True, progress=True)
        self.mode = mode
        # Replace the classifier to match the number of classes
        self.deeplab.classifier = nn.Sequential(
            nn.Conv2d(2048, 2048, 3, padding=1, bias=False),
            nn.BatchNorm2d(2048),
            nn.ReLU(),
            nn.Conv2d(2048, num_classes, 1)
        )
        
        # Freeze the backbone
        for param in self.deeplab.backbone.parameters():
            param.requires_grad = False

        self.mask_encoder = MaskEncoder()


    def forward(self, x, mask):
        
        input_shape = x.shape[-2:] # (H, W)
        features = self.deeplab.backbone(x)
        # Get the lowest resolution feature map
        encoder_output = features["out"] # (B, 2048, H/4, W/4)
        mask_encoder_output = self.mask_encoder(mask) # (B, 2048, H/4, W/4)

        # Ensure the auxiliary encoder output has the same spatial dimensions 
        if encoder_output.shape[2:] != mask_encoder_output.shape[2:]:
            mask_encoder_output = F.interpolate(mask_encoder_output, size=encoder_output.shape[2:], mode='bilinear', align_corners=False)
        if self.mode == 'masked_sum':
            combined_output = encoder_output + mask_encoder_output  
        elif self.mode == 'masked_cat':
            combined_output = torch.cat([encoder_output, mask_encoder_output], dim=1) # concatenate along the channel dimension 
        elif self.mode == 'masked_mul':
            combined_output = encoder_output * mask_encoder_output
        elif self.mode == 'standart':
            combined_output = encoder_output
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
        # TODO: Check the ASPP stuff.
        x = self.deeplab.classifier(combined_output)
        # Upsample the output to match input resolution
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False) # (B, num_classes, H, W)
        return x

def build_prompted_deeplabv3(config=None, num_classes=2, image_size=512):
    if config is None:
        config = {
            "model_parameters": {
                "num_classes": num_classes,
            }
        }
    
    model = PromptedDeepLabV3(num_classes=config["model_parameters"]["num_classes"])
    return model

if __name__ == "__main__":

    ####################################
    ##                                ## 
    ##       Test the dimensions      ##
    ##          of the model!         ##
    ##                                ##
    ####################################
    # Set up parameters

    batch_size = 1
    image_size = 512
    num_classes = 2  # Binary segmentation
    
    import cv2
    import numpy as np


    image_path = "/home/erzurumlu.1/git/PIDNet/data/sky_finder/test/204/20130606_164951.jpg"
    mask_path = "/home/erzurumlu.1/git/PIDNet/data/sky_finder/skyfinder_masks/204.png"
    random_input = T

    rgb_image = cv2.imread(image_path)
    semantic_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if random_input:
        # Create random input tensors
        rgb_image = torch.randn(batch_size, 3, image_size, image_size)  # RGB image
        semantic_mask = torch.randint(0, 2, (batch_size, 1, image_size, image_size)).float()  # Binary mask
    else:
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        rgb_image = cv2.resize(rgb_image, (image_size, image_size))
        rgb_image_display = rgb_image.copy()  # Keep a copy for display
        rgb_image = rgb_image.transpose(2, 0, 1)
        rgb_image = rgb_image.reshape(1, 3, image_size, image_size)

        semantic_mask = cv2.resize(semantic_mask, (image_size, image_size))
        semantic_mask_display = semantic_mask.copy()  # Keep a copy for display
        semantic_mask = semantic_mask.reshape(1, 1, image_size, image_size)
        semantic_mask = semantic_mask.astype(np.float32)

        rgb_image = torch.from_numpy(rgb_image).float()
        semantic_mask = torch.from_numpy(semantic_mask).float() / 255.0
    
    # Initialize the model
    model = build_prompted_deeplabv3(num_classes=num_classes, image_size=image_size)
    
    # Set model to evaluation mode
    model.eval()
    
    # Forward pass
    with torch.no_grad():
        output = model(rgb_image, semantic_mask)
        # Get encoded embeddings
        features = model.deeplab.backbone(rgb_image)
        encoder_output = features["out"]
        mask_encoder_output = model.mask_encoder(semantic_mask)
    
    # Print shapes
    print(f"Input RGB image shape: {rgb_image.shape}")
    print(f"Input semantic mask shape: {semantic_mask.shape}")
    print(f"Output shape: {output.shape}")

    import matplotlib.pyplot as plt


    # Convert output to numpy for visualization
    output_np = output.squeeze().cpu().numpy()
    output_mask = np.argmax(output_np, axis=0)

    # Visualizations
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    
    # # Input Image
    # axs[0, 0].imshow(rgb_image_display)
    # axs[0, 0].set_title('Input Image')
    # axs[0, 0].axis('off')
    
    # # Input Mask
    # axs[0, 1].imshow(semantic_mask_display, cmap='gray')
    # axs[0, 1].set_title('Input Mask')
    # axs[0, 1].axis('off')
    
    # Output Mask
    axs[0, 2].imshow(output_mask, cmap='gray')
    axs[0, 2].set_title('Output Mask')
    axs[0, 2].axis('off')
    
    # Encoded Image Embeddings
    axs[1, 0].imshow(encoder_output.squeeze().mean(dim=0).cpu().numpy())
    axs[1, 0].set_title('Encoded Image Embeddings')
    axs[1, 0].axis('off')
    
    # Encoded Mask Embeddings
    axs[1, 1].imshow(mask_encoder_output.squeeze().mean(dim=0).cpu().numpy())
    axs[1, 1].set_title('Encoded Mask Embeddings')
    axs[1, 1].axis('off')
    
    # Combined Embeddings
    combined_output = encoder_output + mask_encoder_output  # Assuming 'masked_sum' mode
    axs[1, 2].imshow(combined_output.squeeze().mean(dim=0).cpu().numpy())
    axs[1, 2].set_title('Combined Embeddings')
    axs[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('visualizations.png')
    plt.close()

    print("Visualizations saved as 'visualizations.png'")

    # Print shapes
    print(f"Input RGB image shape: {rgb_image.shape}")
    print(f"Input semantic mask shape: {semantic_mask.shape}")
    print(f"Output shape: {output.shape}")
    
    print("All dimension checks passed!")