import os
import ipywidgets as widgets
from IPython.display import display, HTML
import threading
import time
import subprocess
import concurrent.futures
from pathlib import Path
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
import json
import platform
import signal
import atexit

# Authentication Configuration
DEFAULT_CIVITAI_TOKEN = ""  # Put your Civitai token here
DEFAULT_GITHUB_TOKEN = ""   # Put your GitHub token here
DEFAULT_HUGGINGFACE_TOKEN = ""  # Put your HuggingFace token here

# Global variables for parallel processing
MAX_PARALLEL_DOWNLOADS = 5  # Reduced for stability
MAX_PARALLEL_GIT_CLONES = 3  # Reduced for stability
download_processes = []
git_processes = []

# -----------------------------
# Authentication Functions
# -----------------------------
def clean_civitai_url(url):
    """Remove existing token parameter from Civitai URL"""
    parsed = urlparse(url)
    if "civitai.com" in parsed.netloc:
        query_dict = parse_qs(parsed.query)
        query_dict.pop('token', None)  # Remove existing token
        new_query = urlencode(query_dict, doseq=True)
        return urlunparse(parsed._replace(query=new_query))
    return url

def update_url_with_token(url, token, domain):
    """Update URL with authentication token based on domain"""
    parsed = urlparse(url)
    
    if domain == "civitai.com" and token:
        clean_url = clean_civitai_url(url)
        parsed = urlparse(clean_url)
        query_dict = parse_qs(parsed.query)
        query_dict['token'] = [token]
        new_query = urlencode(query_dict, doseq=True)
        return urlunparse(parsed._replace(query=new_query))
    
    return url

def prepare_download_command(url, file_path, tokens=None):
    """Prepare download command with proper authentication"""
    if tokens is None:
        tokens = {
            'civitai': DEFAULT_CIVITAI_TOKEN,
            'github': DEFAULT_GITHUB_TOKEN,
            'huggingface': DEFAULT_HUGGINGFACE_TOKEN
        }
    
    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    
    # Base command for different platforms
    is_windows = platform.system() == "Windows"
    if is_windows:
        base_cmd = f'powershell -Command "Invoke-WebRequest -Uri'
    else:
        base_cmd = f'wget -q --show-progress --progress=dot:giga --tries=3 --timeout=300'
    
    if "civitai.com" in domain and tokens.get('civitai'):
        authenticated_url = update_url_with_token(url, tokens['civitai'], "civitai.com")
        if is_windows:
            return f'{base_cmd} \"{authenticated_url}\" -OutFile \"{file_path}\""'
        else:
            return f'{base_cmd} -O "{file_path}" "{authenticated_url}"'
    
    elif "huggingface.co" in domain and tokens.get('huggingface'):
        if is_windows:
            return f'{base_cmd} \"{url}\" -Headers @{{"Authorization"="Bearer {tokens["huggingface"]}"}} -OutFile \"{file_path}\""'
        else:
            return f'{base_cmd} --header="Authorization: Bearer {tokens["huggingface"]}" -O "{file_path}" "{url}"'
    
    elif "github.com" in domain and tokens.get('github'):
        if is_windows:
            return f'{base_cmd} \"{url}\" -Headers @{{"Authorization"="token {tokens["github"]}"}} -OutFile \"{file_path}\""'
        else:
            return f'{base_cmd} --header="Authorization: token {tokens["github"]}" -O "{file_path}" "{url}"'
    
    else:
        if is_windows:
            return f'{base_cmd} \"{url}\" -OutFile \"{file_path}\""'
        else:
            return f'{base_cmd} -O "{file_path}" "{url}"'

# -----------------------------
# Parallel Processing Functions
# -----------------------------
def download_single_file(url, file_path, tokens=None):
    """Download a single file with authentication"""
    try:
        # Create directory if it doesn't exist
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Skip if file already exists
        if Path(file_path).exists():
            return {'success': True, 'file': file_path, 'message': 'Already exists'}
        
        cmd = prepare_download_command(url, file_path, tokens)
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=900)
        
        if result.returncode == 0:
            return {'success': True, 'file': file_path, 'message': 'Downloaded successfully'}
        else:
            return {'success': False, 'file': file_path, 'message': f'Download failed: {result.stderr[:100]}'}
    
    except subprocess.TimeoutExpired:
        return {'success': False, 'file': file_path, 'message': 'Download timed out'}
    except Exception as e:
        return {'success': False, 'file': file_path, 'message': f'Download error: {str(e)[:100]}'}

def clone_single_repo(repo_url, target_dir):
    """Clone a single git repository"""
    try:
        # Skip if directory already exists
        if Path(target_dir).exists():
            return {'success': True, 'repo': repo_url, 'message': 'Already exists'}
        
        # Create parent directory
        Path(target_dir).parent.mkdir(parents=True, exist_ok=True)
        
        cmd = f'git clone "{repo_url}" "{target_dir}"'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            return {'success': True, 'repo': repo_url, 'message': 'Cloned successfully'}
        else:
            return {'success': False, 'repo': repo_url, 'message': f'Clone failed: {result.stderr[:100]}'}
    
    except subprocess.TimeoutExpired:
        return {'success': False, 'repo': repo_url, 'message': 'Clone timed out'}
    except Exception as e:
        return {'success': False, 'repo': repo_url, 'message': f'Clone error: {str(e)[:100]}'}

# -----------------------------
# Environment Variables
# -----------------------------
public_ip = os.environ.get('RUNPOD_PUBLIC_IP', 'localhost')
external_port = os.environ.get('RUNPOD_TCP_PORT_8188', '8188')
public_url = f"http://{public_ip}:{external_port}/"

# -----------------------------
# CSS Styling
# -----------------------------
css = """
<style>
    body, .jp-Notebook, .container, .output_wrapper, .output_area {
        background-color: #F5F5DC !important; /* full beige background */
    }
    .centered-vbox {
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: flex-start !important;
        min-height: 100vh !important;
        gap: 20px;
        padding-top: 30px;
    }
    .homogenized-button {
        background-color: #E74C3C !important; /* red */
        color: #FFFFFF !important;
        font-weight: bold !important;
        border-radius: 12px !important;
        border: none !important;
        width: 280px !important;
        height: 60px !important;
        font-size: 18px !important;
        cursor: pointer;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2) !important;
    }
    .homogenized-button:hover, .homogenized-button[aria-pressed="true"] {
        background-color: #C0392B !important; /* darker red */
        box-shadow: 0 0 20px 4px #FFF5E1 !important; /* light beige glow */
        transform: translateY(-3px) !important;
    }
    .comfy-title-normal {
        text-align:center;
        font-size:80px;
        font-weight:bold;
        color:#1A237E;
        margin-bottom:10px;
    }
    .status-text {
        text-align:center;
        font-size:20px;
        color:#C0392B;
        font-weight:bold;
        margin-bottom:10px;
    }
    .progress-container {
        background-color: #F5F5DC;
        width: 300px;
        height: 20px;
        border-radius: 10px;
        border: 2px solid #C0392B;
        margin-bottom: 20px;
        display: none; /* initially hidden */
    }
    .progress-bar {
        background-color: #E74C3C;
        height: 100%;
        width: 0%;
        border-radius: 8px;
        transition: width 0.2s ease;
    }
    .comfy-link {
        font-size: 24px;
        font-weight: bold;
        text-decoration: underline;
        color: #E74C3C;
        margin-bottom: 30px;
    }
    .comfy-link:hover {
        color: #C0392B;
    }
    .category-title {
        text-align: center;
        font-size: 16px;
        font-weight: bold;
        color: #1A237E;
        margin: 15px 0 8px 0;
        padding-bottom: 5px;
        border-bottom: 2px solid #C0392B;
        width: 100%;
    }
    .widget-checkbox {
        margin: 3px 0 !important;
    }
    .widget-checkbox .widget-label {
        font-size: 14px !important;
        color: #333 !important;
        font-weight: 500 !important;
    }
    .arrow-button {
        background-color: #E74C3C !important; /* red */
        color: #FFFFFF !important;
        font-weight: bold !important;
        border-radius: 50% !important;
        border: none !important;
        width: 40px !important;
        height: 40px !important;
        font-size: 16px !important;
        cursor: pointer;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2) !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    .arrow-button:hover, .arrow-button[aria-pressed="true"] {
        background-color: #C0392B !important; /* darker red */
        box-shadow: 0 0 20px 4px #FFF5E1 !important; /* light beige glow */
        transform: translateY(-3px) !important;
    }
    .arrow-button[aria-pressed="true"] {
        transform: rotate(180deg) translateY(-3px) !important;
    }
    .arrow-button:not([aria-pressed="true"]) {
        transform: translateY(0) !important;
    }
    .token-input {
        background-color: #F8F8F8 !important;
        border: 2px solid #C0392B !important;
        border-radius: 8px !important;
        padding: 8px !important;
    }
    .auth-status {
        text-align: center;
        font-size: 14px;
        margin: 10px 0;
        padding: 8px;
        border-radius: 6px;
        background-color: #F0F0F0;
    }
    .auth-valid {
        color: #27AE60;
        background-color: #E8F8F5;
        border: 1px solid #27AE60;
    }
    .auth-invalid {
        color: #E74C3C;
        background-color: #FDEDEC;
        border: 1px solid #E74C3C;
    }
</style>
"""
display(HTML(css))

# -----------------------------
# Title and Authentication
# -----------------------------
display(HTML('<div class="comfy-title-normal">ComfyUI</div>'))

# Token input widgets
civitai_token_input = widgets.Password(
    placeholder='Civitai API Key',
    description='',
    value=DEFAULT_CIVITAI_TOKEN,
    layout=widgets.Layout(width='400px', margin='5px 0'),
    style={'description_width': '0px'}
)

github_token_input = widgets.Password(
    placeholder='Enter your GitHub token (optional)',
    description='GitHub Token:',
    value=DEFAULT_GITHUB_TOKEN,
    layout=widgets.Layout(width='400px', margin='5px 0', display='none'),
    style={'description_width': '120px'}
)

huggingface_token_input = widgets.Password(
    placeholder='Enter your HuggingFace token (optional)',
    description='Hugging Face Token:',
    value=DEFAULT_HUGGINGFACE_TOKEN,
    layout=widgets.Layout(width='400px', margin='5px 0', display='none'),
    style={'description_width': '120px'}
)

# Authentication status display
auth_status_html = widgets.HTML(value="")

def update_auth_status():
    """Update authentication status display (Civitai only)"""
    civitai_status = "✓ Configured" if (civitai_token_input.value or DEFAULT_CIVITAI_TOKEN) else "✗ Not configured"
    civitai_class = "auth-valid" if (civitai_token_input.value or DEFAULT_CIVITAI_TOKEN) else "auth-invalid"
    # Match input field width and height (400px width, 38px height)
    auth_status_html.value = f"""
    <div class=\"auth-status {civitai_class}\" style=\"margin:0; padding:8px 12px; font-size:16px; display:inline-block; width:400px; height:38px; line-height:38px; text-align:center; vertical-align:middle;\">{civitai_status}</div>
    """

# Observe token changes to update status
civitai_token_input.observe(lambda change: update_auth_status(), names='value')
update_auth_status()

# Create authentication container
civitai_api_link = widgets.HTML(
    value='<a href="https://civitai.com/user/account" target="_blank" style="color:#E74C3C; text-decoration:underline; font-weight:bold; font-size:14px;">Get Api Key Here</a>'
)
civitai_input_container = widgets.VBox([
    civitai_api_link,
    civitai_token_input,
    auth_status_html
], layout=widgets.Layout(
    background='#1A237E',
    border_radius='18px',
    padding='18px 24px',
    width='fit-content',
    min_width='420px',
    box_shadow='0 2px 8px rgba(26,35,126,0.08)',
    margin='10px 0'
))

# -----------------------------
# Custom Nodes Configuration
# -----------------------------
CUSTOM_NODES_CONFIG = {
    'manager': {
        'name': 'ComfyUI-Manager',
        'url': 'https://github.com/ltdrdata/ComfyUI-Manager.git',
        'required': True,
        'info': 'Essential node manager for ComfyUI'
    },
    'rgthree': {
        'name': 'rgthree-comfy',
        'url': 'https://github.com/rgthree/rgthree-comfy.git',
        'required': False,
        'info': 'Quality of life nodes and utilities'
    },
    'lora_info': {
        'name': 'lora-info',
        'url': 'https://github.com/jitcoder/lora-info.git',
        'required': False,
        'info': 'LoRA info node'
    },
    'impact': {
        'name': 'ComfyUI-Impact-Pack',
        'url': 'https://github.com/ltdrdata/ComfyUI-Impact-Pack.git',
        'required': False,
        'info': 'Advanced image processing nodes'
    },
    'easy_use': {
        'name': 'ComfyUI-Easy-Use',
        'url': 'https://github.com/yolain/ComfyUI-Easy-Use.git',
        'required': False,
        'info': 'Easy use nodes'
    },
    'custom_scripts': {
        'name': 'ComfyUI-Custom-Scripts',
        'url': 'https://github.com/pythongosssss/ComfyUI-Custom-Scripts.git',
        'required': False,
        'info': 'Custom scripts for ComfyUI'
    },
    'inspyrenet_rembg': {
        'name': 'ComfyUI-Inspyrenet-Rembg',
        'url': 'https://github.com/john-mnz/ComfyUI-Inspyrenet-Rembg.git',
        'required': False,
        'info': 'Inspyrenet Rembg nodes'
    },
    'bjornulf': {
        'name': 'Bjornulf_custom_nodes',
        'url': 'https://github.com/justUmen/Bjornulf_custom_nodes.git',
        'required': False,
        'info': 'Bjornulf custom nodes'
    },
    'image_saver': {
        'name': 'comfy-image-saver',
        'url': 'https://github.com/giriss/comfy-image-saver.git',
        'required': False,
        'info': 'Image saver nodes'
    },
    'impact_subpack': {
        'name': 'ComfyUI-Impact-Subpack',
        'url': 'https://github.com/ltdrdata/ComfyUI-Impact-Subpack.git',
        'required': False,
        'info': 'Impact subpack nodes'
    },
    'was_node_suite': {
        'name': 'was-node-suite-comfyui',
        'url': 'https://github.com/ltdrdata/was-node-suite-comfyui.git',
        'required': False,
        'info': 'WAS Node Suite'
    },
    'jps_nodes': {
        'name': 'ComfyUI_JPS-Nodes',
        'url': 'https://github.com/JPS-GER/ComfyUI_JPS-Nodes.git',
        'required': False,
        'info': 'JPS custom nodes'
    },
    'controlnet_aux': {
        'name': 'comfyui_controlnet_aux',
        'url': 'https://github.com/Fannovel16/comfyui_controlnet_aux.git',
        'required': False,
        'info': 'ControlNet auxiliary preprocessors'
    },
    'kjnodes': {
        'name': 'ComfyUI-KJNodes',
        'url': 'https://github.com/kijai/ComfyUI-KJNodes.git',
        'required': False,
        'info': 'KJ custom nodes'
    },
    'essentials': {
        'name': 'ComfyUI_essentials',
        'url': 'https://github.com/cubiq/ComfyUI_essentials.git',
        'required': False,
        'info': 'Essential utility nodes'
    },
    'enricos_nodes': {
        'name': 'ComfyUI-enricos-nodes',
        'url': 'https://github.com/erosDiffusion/ComfyUI-enricos-nodes.git',
        'required': False,
        'info': 'Enrico custom nodes'
    }
}

# Create custom nodes checkboxes
custom_nodes_checkboxes = {}
for key, config in CUSTOM_NODES_CONFIG.items():
    if not config['required']:
        checkbox = widgets.Checkbox(
            value=False,
            description=config['name'],
            indent=False,
            layout=widgets.Layout(width='300px', margin='3px 0')
        )
        custom_nodes_checkboxes[key] = checkbox
        config['checkbox'] = checkbox

# -----------------------------
# Model Checkboxes Definition
# -----------------------------
# Base Models
deliberate_checkbox = widgets.Checkbox(value=True, description='Deliberate v6 (SD1.5)', indent=False, layout=widgets.Layout(margin='5px 0', width='300px'))
flux_checkbox = widgets.Checkbox(value=True, description='Flux Dev (fp8)', indent=False, layout=widgets.Layout(margin='5px 0', width='300px'))
realvis_checkbox = widgets.Checkbox(value=False, description='RealVisXL v4.0', indent=False, layout=widgets.Layout(margin='5px 0', width='300px'))

# VAE Models
sdxl_vae_checkbox = widgets.Checkbox(value=True, description='SDXL VAE', indent=False, layout=widgets.Layout(margin='5px 0', width='300px'))
vae_ft_mse_checkbox = widgets.Checkbox(value=False, description='VAE-ft-mse-840000', indent=False, layout=widgets.Layout(margin='5px 0', width='300px'))
ae_safetensors_checkbox = widgets.Checkbox(value=False, description='AE Safetensors', indent=False, layout=widgets.Layout(margin='5px 0', width='300px'))

# CLIP Models
clip_l_checkbox = widgets.Checkbox(value=True, description='CLIP-L', indent=False, layout=widgets.Layout(margin='5px 0', width='300px'))
t5xxl_checkbox = widgets.Checkbox(value=True, description='T5XXL (fp16)', indent=False, layout=widgets.Layout(margin='5px 0', width='300px'))
clip_g_checkbox = widgets.Checkbox(value=False, description='CLIP-G', indent=False, layout=widgets.Layout(margin='5px 0', width='300px'))

# ControlNet Models
flux_union_checkbox = widgets.Checkbox(value=False, description='FLUX Union ControlNet', indent=False, layout=widgets.Layout(margin='5px 0', width='300px'))
tile_controlnet_checkbox = widgets.Checkbox(value=False, description='Tile ControlNet (SDXL)', indent=False, layout=widgets.Layout(margin='5px 0', width='300px'))

# Upscale Models
realesrgan_checkbox = widgets.Checkbox(value=True, description='RealESRGAN x4plus', indent=False, layout=widgets.Layout(margin='5px 0', width='300px'))
ultrasharp_checkbox = widgets.Checkbox(value=False, description='4x UltraSharp', indent=False, layout=widgets.Layout(margin='5px 0', width='300px'))

# Additional Checkpoints
sdxl_base_checkbox = widgets.Checkbox(value=False, description='SDXL Base', indent=False, layout=widgets.Layout(margin='5px 0', width='300px'))
juggernautxl_v6_checkbox = widgets.Checkbox(value=False, description='JuggernautXL v6', indent=False, layout=widgets.Layout(margin='5px 0', width='300px'))

# LoRA Models
flux_lora_velvet_checkbox = widgets.Checkbox(value=False, description='Flux LoRA Velvet v2', indent=False, layout=widgets.Layout(margin='5px 0', width='300px'))
sdxl_lightning_2_steps_checkbox = widgets.Checkbox(value=False, description='SDXL Lightning 2 Steps', indent=False, layout=widgets.Layout(margin='5px 0', width='300px'))

# -----------------------------
# Model Downloads Configuration
# -----------------------------
MODEL_DOWNLOADS = {
    # Base Models
    'deliberate': {
        'checkbox': deliberate_checkbox,
        'path': 'models/checkpoints/deliberate_v6.safetensors',
        'url': 'https://huggingface.co/XpucT/Deliberate/resolve/main/Deliberate_v6.safetensors'
    },
    'flux': {
        'checkbox': flux_checkbox,
        'path': 'models/checkpoints/flux1-dev-fp8.safetensors',
        'url': 'https://huggingface.co/Comfy-Org/flux1-dev/resolve/main/flux1-dev-fp8.safetensors'
    },
    'realvis': {
        'checkbox': realvis_checkbox,
        'path': 'models/checkpoints/RealVisXL_V4.0.safetensors',
        'url': 'https://huggingface.co/SG161222/RealVisXL_V4.0/resolve/main/RealVisXL_V4.0.safetensors'
    },
    'sdxl_base': {
        'checkbox': sdxl_base_checkbox,
        'path': 'models/checkpoints/SDXL.safetensors',
        'url': 'https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors'
    },
    'juggernautxl_v6': {
        'checkbox': juggernautxl_v6_checkbox,
        'path': 'models/checkpoints/juggernautXL_version6Rundiffusion.safetensors',
        'url': 'https://huggingface.co/lllyasviel/fav_models/resolve/main/fav/juggernautXL_version6Rundiffusion.safetensors'
    },
    
    # VAE Models
    'sdxl_vae': {
        'checkbox': sdxl_vae_checkbox,
        'path': 'models/vae/SDXL_Vae.safetensors',
        'url': 'https://huggingface.co/stabilityai/sdxl-vae/resolve/main/sdxl_vae.safetensors'
    },
    'vae_ft_mse': {
        'checkbox': vae_ft_mse_checkbox,
        'path': 'models/vae/vae-ft-mse-840000-ema-pruned.ckpt',
        'url': 'https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.ckpt'
    },
    'ae_safetensors': {
        'checkbox': ae_safetensors_checkbox,
        'path': 'models/vae/ae.safetensors',
        'url': 'https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/ae.safetensors'
    },
    
    # CLIP Models
    'clip_l': {
        'checkbox': clip_l_checkbox,
        'path': 'models/clip/clip_l.safetensors',
        'url': 'https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors'
    },
    't5xxl': {
        'checkbox': t5xxl_checkbox,
        'path': 'models/clip/t5xxl_fp16.safetensors',
        'url': 'https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors'
    },
    'clip_g': {
        'checkbox': clip_g_checkbox,
        'path': 'models/clip/clip_g.safetensors',
        'url': 'https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/text_encoder_2/model.safetensors'
    },
    
    # ControlNet Models
    'flux_union': {
        'checkbox': flux_union_checkbox,
        'path': 'models/controlnet/flux-union-pro.safetensors',
        'url': 'https://huggingface.co/Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro/resolve/main/diffusion_pytorch_model.safetensors'
    },
    'tile_controlnet': {
        'checkbox': tile_controlnet_checkbox,
        'path': 'models/controlnet/control_v1p_sdxl_tile.safetensors',
        'url': 'https://huggingface.co/xinsir/controlnet-tile-sdxl-1.0/resolve/main/diffusion_pytorch_model.safetensors'
    },
    
    # Upscale Models
    'realesrgan': {
        'checkbox': realesrgan_checkbox,
        'path': 'models/upscale_models/RealESRGAN_x4plus.pth',
        'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
    },
    'ultrasharp': {
        'checkbox': ultrasharp_checkbox,
        'path': 'models/upscale_models/4x-UltraSharp.pth',
        'url': 'https://huggingface.co/FacehugmanIII/4x_foolhardy_Remacri/resolve/main/4x_foolhardy_Remacri.pth'
    },
    
    # LoRA Models
    'flux_lora_velvet': {
        'checkbox': flux_lora_velvet_checkbox,
        'path': 'models/loras/Flux_lora_Velvetv2.safetensors',
        'url': 'https://civitai.com/api/download/models/967375'
    },
    'sdxl_lightning_2_steps': {
        'checkbox': sdxl_lightning_2_steps_checkbox,
        'path': 'models/loras/SDXL_lightning_2_steps.safetensors',
        'url': 'https://huggingface.co/ByteDance/SDXL-Lightning/resolve/main/sdxl_lightning_2step_lora.safetensors'
    }
}

# -----------------------------
# Status and Progress Display
# -----------------------------
status_label = widgets.HTML(value="")
progress_container = widgets.HTML(value="")
open_link_html = widgets.HTML(value="")

display(status_label, progress_container, open_link_html)

# -----------------------------
# Progress Update Function
# -----------------------------
def update_progress(percent):
    progress_container.value = f"""
    <div style="display:flex; justify-content:center; margin-bottom:20px;">
        <div class="progress-container" style="display:block;">
            <div class="progress-bar" style="width:{percent}%;"></div>
        </div>
    </div>
    """

# -----------------------------
# Create Model Selection UI
# -----------------------------
def create_category_html(title):
    return widgets.HTML(value=f"<div class='category-title'>{title}</div>")

def create_expandable_category(title, elements):
    toggle_btn = widgets.ToggleButton(
        description=f"▼ {title}",
        value=False,
        layout=widgets.Layout(width='300px', margin='5px 0', background_color='#E74C3C', color='white')
    )
    
    category_container = widgets.VBox(
        elements,
        layout=widgets.Layout(
            display='none',
            padding='10px',
            margin='0 0 10px 20px'
        )
    )
    
    def toggle_category(change):
        category_container.layout.display = 'flex' if change['new'] else 'none'
        toggle_btn.description = f"▲ {title}" if change['new'] else f"▼ {title}"
    
    toggle_btn.observe(toggle_category, names='value')
    
    return widgets.VBox([toggle_btn, category_container], layout=widgets.Layout(margin='5px 0'))

# Create select all checkbox
def create_select_all_checkbox(category_name, checkboxes):
    select_all = widgets.Checkbox(
        value=False, 
        description=f"Select All {category_name}", 
        indent=False, 
        layout=widgets.Layout(margin='5px 0', width='300px')
    )
    
    def on_select_all_change(change):
        for checkbox in checkboxes:
            checkbox.value = change['new']
    
    select_all.observe(on_select_all_change, names='value')
    return select_all

# Define checkbox groups
checkpoint_checkboxes = [deliberate_checkbox, flux_checkbox, realvis_checkbox, sdxl_base_checkbox, juggernautxl_v6_checkbox]
vae_checkboxes = [sdxl_vae_checkbox, vae_ft_mse_checkbox, ae_safetensors_checkbox]
clip_checkboxes = [clip_l_checkbox, t5xxl_checkbox, clip_g_checkbox]
controlnet_checkboxes = [flux_union_checkbox, tile_controlnet_checkbox]
upscale_checkboxes = [realesrgan_checkbox, ultrasharp_checkbox]
lora_checkboxes = [flux_lora_velvet_checkbox, sdxl_lightning_2_steps_checkbox]

# Create select all checkboxes
select_all_checkpoints = create_select_all_checkbox("Checkpoints", checkpoint_checkboxes)
select_all_vae = create_select_all_checkbox("VAE", vae_checkboxes)
select_all_clip = create_select_all_checkbox("CLIP", clip_checkboxes)
select_all_controlnet = create_select_all_checkbox("ControlNet", controlnet_checkboxes)
select_all_upscale = create_select_all_checkbox("Upscale", upscale_checkboxes)
select_all_loras = create_select_all_checkbox("LoRAs", lora_checkboxes)

# Create category elements
checkpoint_elements = [select_all_checkpoints] + checkpoint_checkboxes
vae_elements = [select_all_vae] + vae_checkboxes
clip_elements = [select_all_clip] + clip_checkboxes
controlnet_elements = [select_all_controlnet] + controlnet_checkboxes
upscale_elements = [select_all_upscale] + upscale_checkboxes
lora_elements = [select_all_loras] + lora_checkboxes

# Create model container
all_model_elements = [
    create_expandable_category("CHECKPOINTS", checkpoint_elements),
    create_expandable_category("VAE Models", vae_elements),
    create_expandable_category("CLIP Models", clip_elements),
    create_expandable_category("ControlNet Models", controlnet_elements),
    create_expandable_category("Upscale Models", upscale_elements),
    create_expandable_category("LoRA Models", lora_elements)
]

# Model container
model_container = widgets.VBox(
    all_model_elements,
    layout=widgets.Layout(
        display='none',
        border='2px solid #C0392B',
        padding='15px',
        margin='10px 0',
        width='400px',
        align_items='flex-start',
        background_color='#FEFEFE'
    )
)


# Custom Nodes Expandable Table UI
custom_nodes_table_rows = []
for key, config in CUSTOM_NODES_CONFIG.items():
    if config.get('checkbox'):
        # Table row: name, info, checkbox
        row = widgets.HBox([
            widgets.Label(config['name'], layout=widgets.Layout(width='160px')),
            widgets.Label(config['info'], layout=widgets.Layout(width='180px', font_size='13px', color='#666')),
            config['checkbox']
        ], layout=widgets.Layout(margin='2px 0'))
        custom_nodes_table_rows.append(row)

custom_nodes_table = widgets.VBox(custom_nodes_table_rows, layout=widgets.Layout(margin='10px 0'))

custom_nodes_toggle_btn = widgets.ToggleButton(
    value=False,
    description='Custom Nodes',
    button_style='',
    layout=widgets.Layout(width='400px', height='38px', background_color='#F5F5DC', border='2px solid #C0392B', margin='0 0 0 0'),
    style={'button_color': '#F5F5DC', 'font_weight': 'bold', 'font_size': '18px', 'color': '#1A237E'}
)

custom_nodes_table_container = widgets.VBox([
    widgets.HTML(value="<p style='color:#666; font-size:14px; margin:10px;'>ComfyUI Manager is always installed</p>"),
    custom_nodes_table
], layout=widgets.Layout(display='none', border='2px solid #1A237E', padding='15px', margin='0 0 10px 0', width='400px', background_color='#F8F8FF'))

def custom_nodes_toggle_changed(change):
    custom_nodes_table_container.layout.display = 'flex' if change['new'] else 'none'
custom_nodes_toggle_btn.observe(custom_nodes_toggle_changed, names='value')

custom_nodes_container = widgets.VBox([
    custom_nodes_toggle_btn,
    custom_nodes_table_container
], layout=widgets.Layout(
    display='none',
    margin='10px 0',
    width='400px'
))

# -----------------------------
# Expandable Beige Buttons for Model Categories
# -----------------------------
def create_beige_expandable_button(title, elements):
    toggle_btn = widgets.ToggleButton(
        value=False,
        description=title,
        button_style='',
        layout=widgets.Layout(width='400px', height='38px', background_color='#F5F5DC', border='2px solid #C0392B', margin='0 0 0 0'),
        style={'button_color': '#F5F5DC', 'font_weight': 'bold', 'font_size': '18px', 'color': '#1A237E'}
    )
    table_container = widgets.VBox(elements, layout=widgets.Layout(display='none', border='2px solid #1A237E', padding='15px', margin='0 0 10px 0', width='400px', background_color='#F8F8FF'))
    def toggle_changed(change):
        table_container.layout.display = 'flex' if change['new'] else 'none'
    toggle_btn.observe(toggle_changed, names='value')
    return widgets.VBox([toggle_btn, table_container], layout=widgets.Layout(margin='10px 0', width='400px'))

# Prepare elements for each category

# --- Subcategory Button Helpers ---
def create_subcategory_button(title, elements):
    btn = widgets.ToggleButton(
        value=False,
        description=title,
        button_style='',
        layout=widgets.Layout(width='350px', height='32px', background_color='#F8F8FF', border='2px solid #C0392B', margin='5px 0'),
        style={'button_color': '#F8F8FF', 'font_weight': 'bold', 'font_size': '15px', 'color': '#1A237E'}
    )
    container = widgets.VBox(elements, layout=widgets.Layout(display='none', margin='0 0 0 20px'))
    def toggle_changed(change):
        container.layout.display = 'flex' if change['new'] else 'none'
    btn.observe(toggle_changed, names='value')
    return widgets.VBox([btn, container], layout=widgets.Layout(margin='0 0 0 0'))

# --- Checkpoints Subcategories ---
sd15_checkpoints = [deliberate_checkbox, flux_checkbox, realvis_checkbox]
sdxl_checkpoints = [sdxl_base_checkbox, juggernautxl_v6_checkbox]
flux_checkpoints = []

# SDXL sub-subcategories
illustrious_sdxl_checkpoints = []
pony_sdxl_checkpoints = []

sdxl_subcategories = [
    create_subcategory_button('Illustrious', illustrious_sdxl_checkpoints),
    create_subcategory_button('Pony', pony_sdxl_checkpoints)
]

checkpoints_subcategories = [
    create_subcategory_button('SD1.5', sd15_checkpoints),
    create_subcategory_button('SDXL', sdxl_checkpoints + sdxl_subcategories),
    create_subcategory_button('Flux', flux_checkpoints)
]
beige_checkpoints = create_beige_expandable_button('Checkpoints', checkpoints_subcategories)

# --- LoRAs Subcategories ---
sd15_loras = [flux_lora_velvet_checkbox]
sdxl_loras = [sdxl_lightning_2_steps_checkbox]
flux_loras = []

# SDXL sub-subcategories
illustrious_sdxl_loras = []
pony_sdxl_loras = []

sdxl_lora_subcategories = [
    create_subcategory_button('Illustrious', illustrious_sdxl_loras),
    create_subcategory_button('Pony', pony_sdxl_loras)
]

loras_subcategories = [
    create_subcategory_button('SD1.5', sd15_loras),
    create_subcategory_button('SDXL', sdxl_loras + sdxl_lora_subcategories),
    create_subcategory_button('Flux', flux_loras)
]
beige_lora = create_beige_expandable_button('LoRAs', loras_subcategories)

# --- Other Categories ---
beige_clip = create_beige_expandable_button('CLIP', clip_elements)
beige_clip_vision = create_beige_expandable_button('CLIP Vision', [])
beige_vae = create_beige_expandable_button('VAE', vae_elements)
beige_controlnet = create_beige_expandable_button('ControlNet', controlnet_elements)
beige_upscale = create_beige_expandable_button('Upscale Models', upscale_elements)

# Additional Downloads
all_model_keys = set(MODEL_DOWNLOADS.keys())
main_category_keys = set([
    'deliberate', 'flux', 'realvis', 'sdxl_base', 'juggernautxl_v6',
    'sdxl_vae', 'vae_ft_mse', 'ae_safetensors',
    'clip_l', 't5xxl', 'clip_g',
    'flux_union', 'tile_controlnet',
    'realesrgan', 'ultrasharp',
    'flux_lora_velvet', 'sdxl_lightning_2_steps'
])
additional_model_keys = list(all_model_keys - main_category_keys)
additional_model_elements = [MODEL_DOWNLOADS[key]['checkbox'] for key in additional_model_keys]
if additional_model_elements:
    select_all_additional = create_select_all_checkbox('Additional Downloads', additional_model_elements)
    additional_model_elements = [select_all_additional] + additional_model_elements
beige_additional = create_beige_expandable_button('Additional Downloads', additional_model_elements)

# --- Final Category Order ---
beige_model_categories_list = [
    beige_checkpoints,
    beige_lora,
    beige_clip,
    beige_clip_vision,
    beige_vae,
    beige_controlnet,
    beige_upscale,
    beige_additional
]
beige_model_categories = widgets.VBox(beige_model_categories_list, layout=widgets.Layout(margin='0 0 10px 0', width='400px'))

# Advanced configuration container

# Recommended Downloads Toggles
recommended_standard_toggle = widgets.ToggleButton(
    value=False,
    description='Recommended Standard Downloads',
    button_style='',
    layout=widgets.Layout(width='400px', height='38px', background_color='#F5F5DC', border='2px solid #C0392B', margin='10px 0'),
    style={'button_color': '#F5F5DC', 'font_weight': 'bold', 'font_size': '16px', 'color': '#1A237E'}
)
recommended_video_toggle = widgets.ToggleButton(
    value=False,
    description='Recommended Video Downloads',
    button_style='',
    layout=widgets.Layout(width='400px', height='38px', background_color='#F5F5DC', border='2px solid #C0392B', margin='10px 0'),
    style={'button_color': '#F5F5DC', 'font_weight': 'bold', 'font_size': '16px', 'color': '#1A237E'}
)
recommended_audio_toggle = widgets.ToggleButton(
    value=False,
    description='Recommended Audio Downloads',
    button_style='',
    layout=widgets.Layout(width='400px', height='38px', background_color='#F5F5DC', border='2px solid #C0392B', margin='10px 0'),
    style={'button_color': '#F5F5DC', 'font_weight': 'bold', 'font_size': '16px', 'color': '#1A237E'}
)
recommended_text_toggle = widgets.ToggleButton(
    value=False,
    description='Recommended Text Generation Downloads',
    button_style='',
    layout=widgets.Layout(width='400px', height='38px', background_color='#F5F5DC', border='2px solid #C0392B', margin='10px 0'),
    style={'button_color': '#F5F5DC', 'font_weight': 'bold', 'font_size': '16px', 'color': '#1A237E'}
)

advanced_config_container = widgets.VBox([
    widgets.HTML(value="<div style='text-align:center; margin-bottom:20px; color:#1A237E; font-size:18px; font-weight:bold;'>Configuration</div>"),
    recommended_standard_toggle,
    recommended_video_toggle,
    recommended_audio_toggle,
    recommended_text_toggle,
    civitai_input_container
], layout=widgets.Layout(
    display='none',
    padding='20px',
    margin='20px 0',
    width='100%',
    align_items='center'
))

# -----------------------------
# Installation Functions
# -----------------------------
def check_disk_space():
    """Check available disk space"""
    try:
        result = subprocess.run(['df', '-h', '.'], capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:
                parts = lines[1].split()
                available = parts[3]
                return available
    except:
        pass
    return "Unknown"

def install_system_dependencies():
    """Install system dependencies with error handling"""
    try:
        # Check if we're in a supported environment
        if not os.path.exists('/usr/bin/apt'):
            return False, "This installer requires Ubuntu/Debian with apt package manager"
        
        # Update package list
        result = subprocess.run(['apt', 'update'], capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            return False, f"Failed to update package list: {result.stderr[:200]}"
        
        # Install dependencies
        packages = ['git', 'python3', 'python3-venv', 'python3-pip', 'wget', 'curl', 'build-essential']
        for package in packages:
            result = subprocess.run(['apt', 'install', '-y', package], capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                return False, f"Failed to install {package}: {result.stderr[:200]}"
        
        return True, "Dependencies installed successfully"
    except subprocess.TimeoutExpired:
        return False, "Installation timed out"
    except Exception as e:
        return False, f"Installation error: {str(e)}"

def get_installation_summary():
    """Get a summary of what will be installed"""
    selected_models = [key for key, config in MODEL_DOWNLOADS.items() if config['checkbox'].value]
    selected_nodes = []
    
    # Always include manager
    selected_nodes.append("ComfyUI-Manager")
    
    # Add selected custom nodes
    for key, config in CUSTOM_NODES_CONFIG.items():
        if config.get('checkbox') and config['checkbox'].value:
            selected_nodes.append(config['name'])
    
    tokens = {
        'civitai': bool(civitai_token_input.value or DEFAULT_CIVITAI_TOKEN),
        'github': bool(github_token_input.value or DEFAULT_GITHUB_TOKEN),
        'huggingface': bool(huggingface_token_input.value or DEFAULT_HUGGINGFACE_TOKEN)
    }
    
    return {
        'models': selected_models,
        'model_count': len(selected_models),
        'nodes': selected_nodes,
        'node_count': len(selected_nodes),
        'tokens': tokens,
        'disk_space': check_disk_space()
    }

def show_installation_summary():
    """Display what will be installed"""
    summary = get_installation_summary()
    
    print("\n" + "="*50)
    print("INSTALLATION SUMMARY")
    print("="*50)
    
    print(f"\n💾 Available disk space: {summary['disk_space']}")
    print(f"📁 Models to download: {summary['model_count']}")
    print(f"🔌 Custom nodes to install: {summary['node_count']}")
    
    print(f"\n🔑 Authentication:")
    print(f"  • Civitai: {'✓' if summary['tokens']['civitai'] else '✗'}")
    
    print(f"\n📄 Selected Models (first 10):")
    for i, model in enumerate(summary['models'][:10], 1):
        print(f"  {i:2d}. {model}")
    if len(summary['models']) > 10:
        print(f"  ... and {len(summary['models']) - 10} more")
    
    print(f"\n🔌 Custom Nodes:")
    for i, node in enumerate(summary['nodes'], 1):
        print(f"  {i:2d}. {node}")
    
    print("\n" + "="*50)

# -----------------------------
# Startup Button
# -----------------------------
startup_btn = widgets.Button(description="Start Up", _dom_classes=["homogenized-button"])

def startup_comfyui(b):
    status_label.value = "<div class='status-text'>Starting installation...</div>"
    open_link_html.value = ""
    
    def run_installation():
        try:
            # Show installation summary
            show_installation_summary()
            
            # Check disk space
            update_progress(5)
            disk_space = check_disk_space()
            print(f"Available disk space: {disk_space}")
            
            # Step 1: Install system dependencies (5-15%)
            status_label.value = "<div class='status-text'>Installing system dependencies...</div>"
            update_progress(10)
            
            success, message = install_system_dependencies()
            if not success:
                status_label.value = f"<div class='status-text'>Failed: {message}</div>"
                progress_container.value = ""
                return
            
            # Step 2: Clone ComfyUI (15-25%)
            status_label.value = "<div class='status-text'>Cloning ComfyUI...</div>"
            update_progress(20)
            
            if not os.path.exists("ComfyUI"):
                result = subprocess.run(
                    ["git", "clone", "https://github.com/comfyanonymous/ComfyUI.git"],
                    capture_output=True, text=True, timeout=300
                )
                if result.returncode != 0:
                    status_label.value = "<div class='status-text'>Failed to clone ComfyUI repository</div>"
                    progress_container.value = ""
                    return
            
            # Step 3: Setup virtual environment (25-35%)
            status_label.value = "<div class='status-text'>Setting up Python environment...</div>"
            update_progress(30)
            
            comfyui_path = Path("ComfyUI")
            if not (comfyui_path / "venv").exists():
                result = subprocess.run(
                    ["python3", "-m", "venv", "venv"],
                    cwd=comfyui_path, capture_output=True, text=True, timeout=300
                )
                if result.returncode != 0:
                    status_label.value = "<div class='status-text'>Failed to create virtual environment</div>"
                    progress_container.value = ""
                    return
            
            # Step 4: Install PyTorch (35-55%)
            status_label.value = "<div class='status-text'>Installing PyTorch (this may take a while)...</div>"
            update_progress(40)
            
            pip_cmd = str(comfyui_path / "venv" / "bin" / "pip")
            
            # Upgrade pip
            subprocess.run([pip_cmd, "install", "--upgrade", "pip"], 
                         cwd=comfyui_path, capture_output=True, timeout=300)
            
            # Install PyTorch
            result = subprocess.run([
                pip_cmd, "install", "torch", "torchvision", "torchaudio", 
                "--index-url", "https://download.pytorch.org/whl/cu121"
            ], cwd=comfyui_path, capture_output=True, text=True, timeout=900)
            
            if result.returncode != 0:
                print(f"PyTorch installation warning: {result.stderr[:200]}")
            
            update_progress(55)
            
            # Step 5: Install requirements (55-65%)
            status_label.value = "<div class='status-text'>Installing ComfyUI requirements...</div>"
            update_progress(60)
            
            result = subprocess.run([pip_cmd, "install", "-r", "requirements.txt"],
                                  cwd=comfyui_path, capture_output=True, text=True, timeout=600)
            if result.returncode != 0:
                print(f"Requirements installation warning: {result.stderr[:200]}")
            
            # Step 6: Create model directories (65-70%)
            status_label.value = "<div class='status-text'>Creating model directories...</div>"
            update_progress(67)
            
            directories = [
                "custom_nodes",
                "models/checkpoints",
                "models/vae",
                "models/clip",
                "models/clip_vision",
                "models/unet",
                "models/unet/IC-Light",
                "models/upscale_models",
                "models/diffusion_models",
                "models/embeddings",
                "models/loras",
                "models/controlnet",
                "models/control-lora",
                "models/ipadapter",
                "models/ipadapter-flux",
                "models/xlabs/ipadapters",
                "models/inpaint",
                "models/style_models",
                "models/BiRefNet",
                "models/vitmatte",
                "models/model_patches",
                "models/CogVideo/loras",
                "models/animatediff",
                "models/dreambooth",
                "models/sam",
                "models/hypernetworks",
                "models/photomaker"
            ]
            for directory in directories:
                (comfyui_path / directory).mkdir(parents=True, exist_ok=True)
            
            # Step 7: Install custom nodes (70-75%)
            status_label.value = "<div class='status-text'>Installing custom nodes...</div>"
            update_progress(72)
            
            custom_nodes_dir = comfyui_path / "custom_nodes"
            custom_nodes_dir.mkdir(exist_ok=True)
            
            # Install ComfyUI Manager (required)
            manager_config = CUSTOM_NODES_CONFIG['manager']
            manager_dir = custom_nodes_dir / manager_config['name']
            if not manager_dir.exists():
                result = subprocess.run([
                    "git", "clone", manager_config['url'], str(manager_dir)
                ], capture_output=True, text=True, timeout=300)
                if result.returncode == 0:
                    print("✓ ComfyUI-Manager installed")
                else:
                    print(f"✗ Failed to install ComfyUI-Manager: {result.stderr[:100]}")
            
            # Install selected custom nodes
            for key, config in CUSTOM_NODES_CONFIG.items():
                if config.get('checkbox') and config['checkbox'].value:
                    node_dir = custom_nodes_dir / config['name']
                    if not node_dir.exists():
                        result = subprocess.run([
                            "git", "clone", config['url'], str(node_dir)
                        ], capture_output=True, text=True, timeout=300)
                        if result.returncode == 0:
                            print(f"✓ {config['name']} installed")
                        else:
                            print(f"✗ Failed to install {config['name']}: {result.stderr[:100]}")
            
            # Step 8: Download models (75-90%)
            status_label.value = "<div class='status-text'>Downloading models...</div>"
            update_progress(80)
            
            # Get tokens
            tokens = {
                'civitai': civitai_token_input.value or DEFAULT_CIVITAI_TOKEN,
                'github': github_token_input.value or DEFAULT_GITHUB_TOKEN,
                'huggingface': huggingface_token_input.value or DEFAULT_HUGGINGFACE_TOKEN
            }
            
            selected_models = [key for key, config in MODEL_DOWNLOADS.items() if config['checkbox'].value]
            
            if selected_models:
                download_tasks = []
                for model_key in selected_models:
                    config = MODEL_DOWNLOADS[model_key]
                    full_path = comfyui_path / config['path']
                    download_tasks.append((config['url'], str(full_path)))
                
                # Parallel downloads with reduced concurrency for stability
                with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                    future_to_download = {
                        executor.submit(download_single_file, url, path, tokens): (url, path) 
                        for url, path in download_tasks
                    }
                    
                    completed = 0
                    successful = 0
                    for future in concurrent.futures.as_completed(future_to_download):
                        url, path = future_to_download[future]
                        try:
                            result = future.result()
                            completed += 1
                            if result['success']:
                                successful += 1
                                print(f"✓ {os.path.basename(path)}: {result['message']}")
                            else:
                                print(f"✗ {os.path.basename(path)}: {result['message']}")
                            
                            # Update progress
                            progress = 80 + (10 * completed / len(download_tasks))
                            update_progress(int(progress))
                            
                        except Exception as e:
                            print(f"✗ {os.path.basename(path)}: Exception {str(e)}")
                
                print(f"Models downloaded: {successful}/{len(download_tasks)}")
            
            # Step 9: Final setup (90-95%)
            status_label.value = "<div class='status-text'>Final setup...</div>"
            update_progress(95)
            
            # Install additional useful packages
            subprocess.run([pip_cmd, "install", "opencv-python", "pillow"], 
                         cwd=comfyui_path, capture_output=True, timeout=300)
            
            # Step 10: Start ComfyUI (95-100%)
            status_label.value = "<div class='status-text'>Starting ComfyUI...</div>"
            update_progress(100)
            
            # Start ComfyUI in background
            python_cmd = str(comfyui_path / "venv" / "bin" / "python")
            subprocess.Popen([
                python_cmd, "main.py", "--listen", "--port", "8188"
            ], cwd=comfyui_path, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Wait for ComfyUI to start
            time.sleep(5)
            
            # Complete
            progress_container.value = ""
            status_label.value = "<div class='status-text'>ComfyUI is running!</div>"
            
            open_link_html.value = f"""
            <div style="text-align:center;">
                <a href="{public_url}" target="_blank" class="comfy-link">OPEN COMFY UI</a>
                <p style="color:#666; font-size:14px;">Access ComfyUI at {public_url}</p>
            </div>
            """
            
        except Exception as e:
            status_label.value = f"<div class='status-text'>Installation failed: {str(e)}</div>"
            progress_container.value = ""
            print(f"Installation error: {str(e)}")

    # Run installation in separate thread
    threading.Thread(target=run_installation, daemon=True).start()

startup_btn.on_click(startup_comfyui)

# -----------------------------
# Advanced Options Toggle
# -----------------------------

# Use Unicode downward triangle for the button
advanced_toggle = widgets.ToggleButton(description="▼", _dom_classes=["arrow-button"])

def advanced_changed(change):
    display_mode = 'flex' if change['new'] else 'none'
    advanced_config_container.layout.display = display_mode
    model_container.layout.display = display_mode
    custom_nodes_container.layout.display = display_mode
    
    # Update button text
    advanced_toggle.description = "▲ Advanced Options" if change['new'] else "▼ Advanced Options"

advanced_toggle.observe(advanced_changed, names='value')

# -----------------------------
# Download Only Button
# -----------------------------

# Helper function to check if ComfyUI is running
def is_comfyui_running():
    # Check for a running process on port 8188 (simple check)
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.settimeout(0.5)
        s.connect((public_ip, int(external_port)))
        s.close()
        return True
    except Exception:
        return False

# Helper function to check if all required folders exist
def all_required_folders_exist():
    comfyui_path = Path("ComfyUI")
    required_dirs = [
        "custom_nodes",
        "models/checkpoints",
        "models/vae",
        "models/clip",
        "models/clip_vision",
        "models/unet",
        "models/unet/IC-Light",
        "models/upscale_models",
        "models/diffusion_models",
        "models/embeddings",
        "models/loras",
        "models/controlnet",
        "models/control-lora",
        "models/ipadapter",
        "models/ipadapter-flux",
        "models/xlabs/ipadapters",
        "models/inpaint",
        "models/style_models",
        "models/BiRefNet",
        "models/vitmatte",
        "models/model_patches",
        "models/CogVideo/loras",
        "models/animatediff",
        "models/dreambooth",
        "models/sam",
        "models/hypernetworks",
        "models/photomaker"
    ]
    for d in required_dirs:
        if not (comfyui_path / d).exists():
            return False
    return True

download_btn = widgets.Button(description="Start selected Downloads", _dom_classes=["homogenized-button"])
download_btn.layout.display = 'none'  # Initially hidden

# Function to update download_btn visibility
def update_download_btn_visibility():
    if is_comfyui_running() or all_required_folders_exist():
        download_btn.layout.display = 'flex'
    else:
        download_btn.layout.display = 'none'

# Periodically check and update button visibility
import threading as _threading
def _periodic_download_btn_check():
    while True:
        update_download_btn_visibility()
        time.sleep(2)
_threading.Thread(target=_periodic_download_btn_check, daemon=True).start()

def download_models_only(b):
    selected_models = [key for key, config in MODEL_DOWNLOADS.items() if config['checkbox'].value]
    
    if not selected_models:
        status_label.value = "<div class='status-text'>No models selected!</div>"
        return
    
    status_label.value = "<div class='status-text'>Downloading models...</div>"
    open_link_html.value = ""
    
    def run_download():
        try:
            tokens = {
                'civitai': civitai_token_input.value or DEFAULT_CIVITAI_TOKEN,
                'github': github_token_input.value or DEFAULT_GITHUB_TOKEN,
                'huggingface': huggingface_token_input.value or DEFAULT_HUGGINGFACE_TOKEN
            }
            # Create ComfyUI directory structure
            comfyui_path = Path("ComfyUI")
            directories = [
                "custom_nodes",
                "models/checkpoints",
                "models/vae",
                "models/clip",
                "models/clip_vision",
                "models/unet",
                "models/unet/IC-Light",
                "models/upscale_models",
                "models/diffusion_models",
                "models/embeddings",
                "models/loras",
                "models/controlnet",
                "models/control-lora",
                "models/ipadapter",
                "models/ipadapter-flux",
                "models/xlabs/ipadapters",
                "models/inpaint",
                "models/style_models",
                "models/BiRefNet",
                "models/vitmatte",
                "models/model_patches",
                "models/CogVideo/loras",
                "models/animatediff",
                "models/dreambooth",
                "models/sam",
                "models/hypernetworks",
                "models/photomaker"
            ]
            for directory in directories:
                (comfyui_path / directory).mkdir(parents=True, exist_ok=True)
            
            # Download models
            download_tasks = [(MODEL_DOWNLOADS[key]['url'], str(comfyui_path / MODEL_DOWNLOADS[key]['path'])) 
                            for key in selected_models]
            
            update_progress(10)
            
            successful = 0
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                future_to_download = {
                    executor.submit(download_single_file, url, path, tokens): (url, path) 
                    for url, path in download_tasks
                }
                
                completed = 0
                for future in concurrent.futures.as_completed(future_to_download):
                    url, path = future_to_download[future]
                    try:
                        result = future.result()
                        completed += 1
                        if result['success']:
                            successful += 1
                            print(f"✓ {os.path.basename(path)}: {result['message']}")
                        else:
                            print(f"✗ {os.path.basename(path)}: {result['message']}")
                        
                        progress = 10 + (80 * completed / len(download_tasks))
                        update_progress(int(progress))
                        
                    except Exception as e:
                        print(f"✗ {os.path.basename(path)}: {str(e)}")
            
            update_progress(100)
            progress_container.value = ""
            status_label.value = f"<div class='status-text'>Download complete! {successful}/{len(download_tasks)} models downloaded.</div>"
            
        except Exception as e:
            status_label.value = f"<div class='status-text'>Download failed: {str(e)}</div>"
            progress_container.value = ""
    
    threading.Thread(target=run_download, daemon=True).start()

download_btn.on_click(download_models_only)

# -----------------------------
# Main Interface
# -----------------------------
main_buttons = widgets.HBox([
    startup_btn
], layout=widgets.Layout(justify_content='center', margin='20px 0'))

container = widgets.VBox([
    main_buttons,
    advanced_toggle,
    widgets.VBox([
        download_btn,
        advanced_config_container,
        custom_nodes_container,
        beige_model_categories
    ], layout=widgets.Layout(
        display='none',
        align_items='center',
        justify_content='center',
        width='100%'
    ), _dom_classes=["extending-mechanik"])
], _dom_classes=["centered-vbox"])

def advanced_changed(change):
    display_mode = 'flex' if change['new'] else 'none'
    container.children[2].layout.display = display_mode
    # Change to upward triangle when expanded, downward when collapsed
    advanced_toggle.description = "▲" if change['new'] else "▼"

advanced_toggle.observe(advanced_changed, names='value')

display(container)