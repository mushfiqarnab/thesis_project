import os
import cv2
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from diffusers import StableDiffusionInpaintPipeline
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class OrganicMaskGenerator:
    """Generates organic, irregular masks mimicking real surgical trauma bounds."""
    def __init__(self, size=(512, 512)):
        self.size = size

    def generate(self):
        mask = np.zeros(self.size, dtype=np.uint8)
        # Random center for the scar
        cx = np.random.randint(self.size[0] // 4, 3 * self.size[0] // 4)
        cy = np.random.randint(self.size[1] // 4, 3 * self.size[1] // 4)
        
        # Base organic shape using random polygons/bezier approximations
        num_points = np.random.randint(5, 10)
        points = []
        radius = np.random.randint(30, 80)
        for i in range(num_points):
            angle = i * (2 * np.pi / num_points)
            r = radius + np.random.randint(-20, 30)
            x = int(cx + r * np.cos(angle))
            y = int(cy + r * np.sin(angle))
            points.append([x, y])
            
        points = np.array(points, np.int32)
        points = points.reshape((-1, 1, 2))
        cv2.fillPoly(mask, [points], 255)
        
        # Apply organic irregularities via morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.dilate(mask, kernel, iterations=np.random.randint(1, 4))
        mask = cv2.GaussianBlur(mask, (21, 21), 0)
        
        # Hard threshold to keep it binary
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        return Image.fromarray(mask)

class SourceImageDataset(Dataset):
    def __init__(self, image_paths, size=(512, 512)):
        self.image_paths = image_paths
        self.size = size
        self.mask_gen = OrganicMaskGenerator(size=size)
        self.transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert("RGB").resize(self.size)
        mask = self.mask_gen.generate()
        return {
            "path": path,
            "image": self.transform(img),
            "mask": transforms.ToTensor()(mask),
            "raw_image": img,
            "raw_mask": mask
        }

def compute_ssim_unmasked(orig_img, gen_img, mask):
    """
    Computes SSIM strictly on the unmasked regions to verify identity lock.
    orig_img, gen_img: PIL Images (RGB)
    mask: PIL Image (L), where 255 is the inpaint region.
    """
    orig_np = np.array(orig_img)
    gen_np = np.array(gen_img)
    mask_np = np.array(mask)
    
    # Invert mask: 255 means background (unmasked), 0 means scar
    bg_mask = cv2.bitwise_not(mask_np)
    
    # Extract background only
    orig_bg = cv2.bitwise_and(orig_np, orig_np, mask=bg_mask)
    gen_bg = cv2.bitwise_and(gen_np, gen_np, mask=bg_mask)
    
    # Compute SSIM
    score, _ = ssim(orig_bg, gen_bg, channel_axis=2, full=True)
    return score

def main():
    # Configuration
    BATCH_SIZE = 4  # Tuned for RTX 4060 (8GB VRAM)
    IMAGE_DIR = "data/raw/images" # Placeholder
    OUTPUT_DIR = "data/counterfactual_diffusion"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Initialize Pipeline
    print("Loading Stable Diffusion Inpainting Pipeline...")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16,
        safety_checker=None
    )
    pipe = pipe.to("cuda")
    pipe.enable_xformers_memory_efficient_attention()
    
    # Get image paths (Mocking for 10k images)
    # image_paths = [os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR) if f.endswith(('.jpg', '.png'))]
    # For scaffolding, we will simulate the list
    image_paths = [f"mock_image_{i}.jpg" for i in range(10000)]
    
    dataset = SourceImageDataset(image_paths)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, collate_fn=lambda x: x)
    
    prompt = "highly detailed, photorealistic healed hypertrophic facial scar, surgical tissue, hyperpigmentation, blending seamlessly with natural skin texture."
    negative_prompt = "cartoon, flat color, distorted anatomy, artificial, MS-paint, obvious boundaries, deformed."

    print(f"Starting generation for {len(image_paths)} images...")
    
    # 3. Batch Generation Orchestration
    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Generating Counterfactuals"):
            
            # Extract lists for diffusers pipeline
            raw_images = [item["raw_image"] for item in batch]
            raw_masks = [item["raw_mask"] for item in batch]
            paths = [item["path"] for item in batch]
            
            # Keep trying until all images in the batch pass SSIM
            # To maximize GPU utility, we batch process, but if any fail, we redraw those specifically.
            pending_indices = list(range(len(batch)))
            final_images = [None] * len(batch)
            
            while pending_indices:
                curr_images = [raw_images[i] for i in pending_indices]
                curr_masks = [raw_masks[i] for i in pending_indices]
                
                generated = pipe(
                    prompt=[prompt] * len(curr_images),
                    negative_prompt=[negative_prompt] * len(curr_images),
                    image=curr_images,
                    mask_image=curr_masks,
                    num_inference_steps=25,
                    guidance_scale=7.5
                ).images
                
                next_pending = []
                for idx, gen_img, orig_img, orig_mask in zip(pending_indices, generated, curr_images, curr_masks):
                    score = compute_ssim_unmasked(orig_img, gen_img, orig_mask)
                    
                    if score >= 0.99:
                        final_images[idx] = gen_img
                    else:
                        # Identity drift detected, reject and redraw
                        next_pending.append(idx)
                
                pending_indices = next_pending
            
            # Save final verified images
            for p, img in zip(paths, final_images):
                out_name = os.path.basename(p).replace(".jpg", "_cf.jpg")
                # img.save(os.path.join(OUTPUT_DIR, out_name))

if __name__ == "__main__":
    main()
