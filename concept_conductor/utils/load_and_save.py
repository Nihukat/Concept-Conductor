from PIL import Image
import hashlib
import re
import torch
from pathlib import Path

def image_to_hash(image: Image.Image) -> str:
    """
    Convert a PIL.Image to a hash key using SHA-256.

    Args:
        image (PIL.Image.Image): Input image.

    Returns:
        str: SHA-256 hash key of the image.
    """
    # Convert image to bytes
    image_bytes = image.tobytes()

    # Create a SHA-256 hash object
    hash_object = hashlib.sha256()

    # Update the hash object with the image bytes
    hash_object.update(image_bytes)

    # Get the hexadecimal representation of the hash
    hash_key = hash_object.hexdigest()

    return hash_key

def truncate_text(text, length=150):
    text_truncated = text[:length]
    illegal_chars_pattern = r'[<>:"/\\|?*]'
    text_truncated = re.sub(illegal_chars_pattern, '_', text_truncated)   
    return text_truncated

def save_image_info(image_info, latents_outdir):
    prompt = image_info['prompt']
    hash_key = image_info['hash_key']
    text_truncated = truncate_text(prompt)
    file_name = f"{hash_key}_{text_truncated}.pt"
    Path(latents_outdir).mkdir(parents=True, exist_ok=True)
    file_path = str(Path(latents_outdir) / Path(file_name))
    torch.save(image_info, file_path)
    
def load_image_info(latents_outdir, hash_key, prompt):
    text_truncated = truncate_text(prompt)
    file_name = f"{hash_key}_{text_truncated}.pt"
    file_path = str(Path(latents_outdir) / Path(file_name))
    if Path(file_path).is_file():
        return torch.load(file_path)
    else:
        return None