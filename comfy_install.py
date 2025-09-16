# python ninja_start.py
# python ninja_restart.py



#!/usr/bin/env python3
"""
ComfyUI Installation Script - Fixed Token Authentication Version
"""

# Set your tokens here
DEFAULT_CIVITAI_TOKEN = "fd4ae815a82358dab77c19eb48c9f2cf"  # <- Put your Civitai token here between the quotes
DEFAULT_GITHUB_TOKEN = ""   # <- Put your GitHub token here if you have one
DEFAULT_HUGGINGFACE_TOKEN = ""       # <- Put your Hugging Face token here if you have one

import subprocess
import os
import time
import sys
import shutil
import platform
import signal
import atexit
import argparse
from pathlib import Path
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
from contextlib import contextmanager
# Global variables for Rich functionality
RICH_AVAILABLE = False
console = None

try:
    from rich.console import Console
    from rich.progress import Progress, BarColumn, TimeRemainingColumn, TextColumn, TaskID
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    pass

def parse_arguments():
    """Parse command line arguments for API tokens"""
    parser = argparse.ArgumentParser(description='ComfyUI Installation Script')
    parser.add_argument('--civitai-token', 
                      help='Civitai API token for authenticated downloads',
                      default=None)
    parser.add_argument('--github-token',
                      help='GitHub API token for authenticated downloads',
                      default=None)
    parser.add_argument('--huggingface-token',
                      help='Hugging Face API token for authenticated downloads',
                      default=None)
    return parser.parse_args()

def clean_civitai_url(url):
    """Remove existing token parameter from Civitai URL"""
    parsed = urlparse(url)
    if "civitai.com" in parsed.netloc:
        query_dict = parse_qs(parsed.query)
        # Remove any existing token parameter
        query_dict.pop('token', None)
        new_query = urlencode(query_dict, doseq=True)
        return urlunparse(parsed._replace(query=new_query))
    return url

def update_url_with_token(url, token, domain):
    """Update URL with authentication token based on domain"""
    parsed = urlparse(url)
    
    if domain == "civitai.com" and token:
        # Clean URL first, then add our token
        clean_url = clean_civitai_url(url)
        parsed = urlparse(clean_url)
        query_dict = parse_qs(parsed.query)
        query_dict['token'] = [token]
        new_query = urlencode(query_dict, doseq=True)
        return urlunparse(parsed._replace(query=new_query))
    
    return url

class Colors:
    """ANSI color codes for terminal output"""
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    END = '\033[0m'

class ComfyUIInstaller:
    """Main installer class"""
    
    def __init__(self, civitai_token=None, github_token=None, huggingface_token=None):
        self.workspace = Path(__file__).parent / "ComfyUI"
        self.venv_path = self.workspace / "venv"
        self.is_windows = platform.system() == "Windows"
        self.download_processes = []
        
        # Use provided tokens or fall back to defaults
        self.civitai_token = civitai_token or DEFAULT_CIVITAI_TOKEN
        self.github_token = github_token or DEFAULT_GITHUB_TOKEN
        self.huggingface_token = huggingface_token or DEFAULT_HUGGINGFACE_TOKEN
        
        print(f"{Colors.CYAN}Token Configuration:{Colors.END}")
        print(f"Civitai Token: {'✓ Configured' if self.civitai_token else '✗ Not configured'}")
        print(f"GitHub Token: {'✓ Configured' if self.github_token else '✗ Not configured'}")
        print(f"HuggingFace Token: {'✓ Configured' if self.huggingface_token else '✗ Not configured'}")
        
        # Setup cleanup handlers
        atexit.register(self._cleanup)
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _cleanup(self):
        """Cleanup on exit"""
        print('\033[?25h')  # Show cursor
        for _, proc in self.download_processes:
            if proc.poll() is None:
                proc.terminate()
    
    def _signal_handler(self, signum, frame):
        """Handle interrupt signals"""
        print("\nReceived interrupt signal, cleaning up...")
        self._cleanup()
        sys.exit(0)
    
    @contextmanager
    def _working_directory(self, path):
        """Context manager for changing working directory"""
        old_cwd = os.getcwd()
        try:
            os.chdir(path)
            yield
        finally:
            os.chdir(old_cwd)
    
    def clear_screen(self):
        """Clear terminal screen"""
        os.system('cls' if self.is_windows else 'clear')
    
    def run_command(self, cmd, check=True, capture_output=False):
        """Run a shell command with proper error handling"""
        print(f"\n{Colors.BLUE}>>> Running: {cmd[:100]}{'...' if len(cmd) > 100 else ''}{Colors.END}")
        try:
            if capture_output:
                result = subprocess.run(cmd, shell=True, check=check, 
                                      capture_output=True, text=True)
                return result.stdout.strip()
            else:
                subprocess.run(cmd, shell=True, check=check)
                return True
        except subprocess.CalledProcessError as e:
            print(f"{Colors.RED}Error running command: {e}{Colors.END}")
            if check:
                raise
            return False
        except Exception as e:
            print(f"{Colors.RED}Unexpected error: {e}{Colors.END}")
            if check:
                raise
            return False
    
    def ensure_directory(self, path):
        """Create directory if it doesn't exist"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        print(f"{Colors.GREEN}Directory ready: {path}{Colors.END}")
    
    def print_header(self, title):
        """Print formatted header"""
        print(f"\n{Colors.BOLD}{Colors.CYAN}=== {title} ==={Colors.END}\n")
    
    def setup_system_dependencies(self):
        """Install system dependencies"""
        self.print_header("System Dependencies")
        
        if not self.is_windows:
            self.run_command("apt update && apt upgrade -y")
            self.run_command("apt install -y git python3 python3-venv python3-pip wget ffmpeg tmux net-tools lsof curl")
        else:
            print("Windows detected - please ensure Python 3.8+ and Git are installed")
        
        time.sleep(2)
        self.clear_screen()
    
    def clone_comfyui(self):
        """Clone ComfyUI repository"""
        self.print_header("ComfyUI Core")
        
        if not self.workspace.exists():
            self.run_command(f"git clone https://github.com/comfyanonymous/ComfyUI.git {self.workspace}")
        else:
            print(f"{Colors.YELLOW}ComfyUI directory already exists, skipping clone.{Colors.END}")
        
        time.sleep(2)
        self.clear_screen()
    
    def setup_virtual_environment(self):
        """Setup Python virtual environment and dependencies"""
        # MUST declare globals first
        global RICH_AVAILABLE, console
        
        self.print_header("Python Environment")
        
        # Create virtual environment
        if not self.venv_path.exists():
            with self._working_directory(self.workspace):
                self.run_command("python3 -m venv venv")
        
        # Activate and upgrade pip
        pip_cmd = str(self.venv_path / ("Scripts/pip" if self.is_windows else "bin/pip"))
        self.run_command(f'"{pip_cmd}" install --upgrade pip')
        
        # Install Rich for progress display if not available
        if not RICH_AVAILABLE:
            self.run_command(f'"{pip_cmd}" install rich')
            try:
                from rich.console import Console
                from rich.progress import Progress, BarColumn, TimeRemainingColumn, TextColumn, TaskID
                RICH_AVAILABLE = True
                console = Console()
            except ImportError:
                pass
        
        # Install PyTorch with CUDA support if available
        cuda_available = self._check_cuda_availability()
        self.install_torch(pip_cmd, cuda_available)
        self.run_command(f'"{pip_cmd}" install onnxruntime-gpu opencv-python')
        
        # Install ComfyUI requirements
        requirements_file = self.workspace / "requirements.txt"
        if requirements_file.exists():
            self.run_command(f'"{pip_cmd}" install -r "{requirements_file}"')
        
        time.sleep(2)
        self.clear_screen()
    
    def _check_cuda_availability(self):
        """Check if CUDA is available"""
        try:
            result = self.run_command('nvidia-smi --query-gpu=driver_version --format=csv,noheader', 
                                    check=False, capture_output=True)
            return bool(result)
        except:
            return False
    
    def install_torch(self, pip_cmd, cuda_available):
        """Install PyTorch packages with proper CUDA support"""
        # MUST be the first thing in the function
        global RICH_AVAILABLE, console

        # Now you can use them safely
        if RICH_AVAILABLE:
            console.print("Installing PyTorch...")

        # Build the extra index URL if CUDA is available
        pytorch_extra = ""
        if cuda_available:
            pytorch_extra = " --index-url https://download.pytorch.org/whl/cu121"

        # Run the pip command
        self.run_command(f'"{pip_cmd}" install torch torchvision torchaudio{pytorch_extra}')
    
    def create_directory_structure(self):
        """Create required directories"""
        self.print_header("Directory Structure")
        
        directories = [
            "input", "output", "temp", "custom_nodes",
            "models/checkpoints", "models/vae", "models/clip_vision", "models/clip",
            "models/upscale_models", "models/diffusion_models", "models/loras", "models/ipadapter",
            "models/inpaint", "models/controlnet", "models/style_models", "models/BiRefNet",
            "models/embeddings",
            "models/unet/IC-Light", "models/CogVideo/loras", "models/ipadapter-flux",
            "models/xlabs/ipadapters", "models/vitmatte", "models/model_patches"
        ]
        
        for directory in directories:
            self.ensure_directory(self.workspace / directory)
        
        time.sleep(2)
        self.clear_screen()
    
    def prepare_download_command(self, url, file_path):
        """Prepare wget command with proper authentication"""
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        
        # Base wget command
        base_cmd = f'wget -q --show-progress --progress=dot:giga --tries=3 --timeout=60'
        
        if "civitai.com" in domain and self.civitai_token:
            # Update URL with token for Civitai
            authenticated_url = update_url_with_token(url, self.civitai_token, "civitai.com")
            return f'{base_cmd} -O "{file_path}" "{authenticated_url}"'
        
        elif "huggingface.co" in domain and self.huggingface_token:
            # Use Authorization header for Hugging Face
            return f'{base_cmd} --header="Authorization: Bearer {self.huggingface_token}" -O "{file_path}" "{url}"'
        
        elif "github.com" in domain and self.github_token:
            # Use Authorization header for GitHub
            return f'{base_cmd} --header="Authorization: token {self.github_token}" -O "{file_path}" "{url}"'
        
        else:
            # No authentication needed or token not available
            return f'{base_cmd} -O "{file_path}" "{url}"'
    
    def download_models(self):
        """Start model downloads"""
        self.print_header("Model Downloads")
        
        if RICH_AVAILABLE:
            console.rule("[bold cyan]COMFYUI MODEL DOWNLOADS[/bold cyan]")
        
        downloads = [
            # Flux Models
            ("models/checkpoints/flux1-dev-fp8.safetensors",
             "https://huggingface.co/Comfy-Org/flux1-dev/resolve/main/flux1-dev-fp8.safetensors?download=true"),

            # SDXL Checkpoints
            ("models/checkpoints/SDXL.safetensors",
             "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors?download=true"),
            ("models/checkpoints/SDXL_WildCard.safetensors",
             "https://civitai.com/api/download/models/345685"),
            ("models/checkpoints/SDXL_CyberRealisticXL.safetensors",
             "https://civitai.com/api/download/models/1609607"),
            ("models/checkpoints/zavychromaxl_v80.safetensors",
             "https://huggingface.co/misri/zavychromaxl_v80/resolve/main/zavychromaxl_v80.safetensors"),
            ("models/checkpoints/juggernautXL_version6Rundiffusion.safetensors",
             "https://huggingface.co/lllyasviel/fav_models/resolve/main/fav/juggernautXL_version6Rundiffusion.safetensors"),
            ("models/checkpoints/DreamShaperXL.safetensors",
             "https://civitai.com/api/download/models/351306"),
         
            # SD 1.5 Models
            ("models/checkpoints/SD1.5_DreamShaper.safetensors",
             "https://civitai.com/api/download/models/128713"),
            ("models/checkpoints/SD1.5_RevAnimated.safetensors",
             "https://civitai.com/api/download/models/425083"),
            ("models/checkpoints/SD1.5_Epic_Realism.safetensors", 
             "https://civitai.com/api/download/models/143906"),
            ("models/checkpoints/SD1.5_Deliberate.safetensors",
             "https://huggingface.co/XpucT/Deliberate/resolve/main/Deliberate_v6.safetensors?download=true"),

            # Pony Models
            ("models/checkpoints/Pony.safetensors",
             "https://civitai.com/api/download/models/290640"),
            ("models/checkpoints/Pony_CyberRealistic.safetensors",
             "https://civitai.com/api/download/models/2178176"),
            ("models/checkpoints/Pony_Lucent.safetensors",
             "https://civitai.com/api/download/models/1971591"),
            ("models/checkpoints/Pony_DucHaiten_Real.safetensors",
             "https://civitai.com/api/download/models/695106"),
            ("models/checkpoints/Pony_Real_Dream.safetensors",
             "https://civitai.com/api/download/models/2129811"),
            ("models/checkpoints/Pony_Real_Merge.safetensors",
             "https://civitai.com/api/download/models/994131"),
            ("models/checkpoints/Pony_Realism.safetensors",
             "https://civitai.com/api/download/models/914390"),

            # Illustrious Models
            ("models/checkpoints/Illustrious.safetensors",
             "https://civitai.com/api/download/models/889818"),
            ("models/checkpoints/Illustrious_AnIco.safetensors",
             "https://civitai.com/api/download/models/1641205"),
            ("models/checkpoints/Illustrious_Illustrij.safetensors",
             "https://civitai.com/api/download/models/2186168"),
            ("models/checkpoints/Illustrious_ToonMerge.safetensors",
             "https://civitai.com/api/download/models/1622588"),
            ("models/checkpoints/Illustrious_SEMImergeijV6.safetensors",
             "https://civitai.com/api/download/models/1920758"),
            
            # Style Models
            ("models/style_models/Flux_Redux.safetensors",
             "https://civitai.com/api/download/models/1086258"),
            
            # VAE Models
            ("models/vae/SDXL_Vae.safetensors", 
             "https://huggingface.co/stabilityai/sdxl-vae/resolve/main/sdxl_vae.safetensors?download=true"),
            ("models/vae/ae.safetensors",
             "https://huggingface.co/Comfy-Org/Lumina_Image_2.0_Repackaged/resolve/main/split_files/vae/ae.safetensors?download=true"),
            
            # CLIP Vision Models
            ("models/clip_vision/CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors", 
             "https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors"),
            ("models/clip_vision/CLIP-ViT-bigG-14-laion2B-39B-b160k.safetensors", 
             "https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/image_encoder/model.safetensors"),
            ("models/clip_vision/model_l.safetensors",
             "https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/model.safetensors?download=true"),
            ("models/clip_vision/clip-vision_vit-h.safetensors",
             "https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors"),
            ("models/clip_vision/clip_vision_h.safetensors",
             "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors?download=true"),
            ("models/clip_vision/sigclip_vision_patch14_384.safetensors",
             "https://huggingface.co/Comfy-Org/sigclip_vision_384/resolve/main/sigclip_vision_patch14_384.safetensors?download=true"),
            
            # CLIP Models
            ("models/clip/clip_l.safetensors", 
             "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors?download=true"),
            ("models/clip/t5xxl_fp16.safetensors", 
             "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors?download=true"),
            ("models/clip/clip_g.safetensors",
             "https://huggingface.co/calcuis/sd3.5-large-gguf/resolve/7f72f2a432131bba82ecd1aafb931ac99f0f05f7/clip_g.safetensors?download=true"),
            
            # Diffusion Models & Related Files
            ("models/diffusion_models/Flux_Fill.safetensors",
             "https://civitai.com/api/download/models/1086292"),
            ("models/loras/uso-flux1-dit-lora-v1.safetensors",
             "https://huggingface.co/Comfy-Org/USO_1.0_Repackaged/resolve/main/split_files/loras/uso-flux1-dit-lora-v1.safetensors"),
            ("models/model_patches/uso-flux1-projector-v1.safetensors",
             "https://huggingface.co/Comfy-Org/USO_1.0_Repackaged/resolve/main/split_files/model_patches/uso-flux1-projector-v1.safetensors"),
            
            # Upscale Models
            ("models/upscale_models/4x_foolhardy_Remacri.pth", 
             "https://huggingface.co/FacehugmanIII/4x_foolhardy_Remacri/resolve/main/4x_foolhardy_Remacri.pth?download=true"),
            ("models/upscale_models/RealESRGAN_x4plus.pth", 
             "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"),
            ("models/upscale_models/4x-UltraSharp.safetensors", 
             "https://civitai.com/api/download/models/125843"),
            ("models/upscale_models/4x_NMKD_Siax_200k.pth",
             "https://huggingface.co/uwg/upscaler/resolve/main/ESRGAN/4x_NMKD-Siax_200k.pth?download=true"),
            ("models/upscale_models/RealESRGAN_x4plus_anime_and_illustrations_6B.pth",
             "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"),
            ("models/upscale_models/4x_NMKD-Superscale-SP_178000_G.pth",
             "https://huggingface.co/gemasai/4x_NMKD-Superscale-SP_178000_G/resolve/main/4x_NMKD-Superscale-SP_178000_G.pth"),
            ("models/upscale_models/OmniSR_X2_DIV2K.safetensors",
             "https://huggingface.co/Acly/Omni-SR/resolve/main/OmniSR_X2_DIV2K.safetensors"),
            ("models/upscale_models/OmniSR_X3_DIV2K.safetensors",
             "https://huggingface.co/Acly/Omni-SR/resolve/main/OmniSR_X3_DIV2K.safetensors"),
            ("models/upscale_models/OmniSR_X4_DIV2K.safetensors",
             "https://huggingface.co/Acly/Omni-SR/resolve/main/OmniSR_X4_DIV2K.safetensors"),
            
            # ControlNet Models
            ("models/controlnet/FLUX.1-dev-Controlnet-Union.safetensors",
             "https://huggingface.co/InstantX/FLUX.1-dev-Controlnet-Union/resolve/main/diffusion_pytorch_model.safetensors"),
            ("models/controlnet/control_lora_rank128_v11f1e_sd15_tile_fp16.safetensors",
             "https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_lora_rank128_v11f1e_sd15_tile_fp16.safetensors"),
            ("models/controlnet/t2i-adapter-lineart-sdxl-1.0_fp16.safetensors",
             "https://huggingface.co/TencentARC/t2i-adapter-lineart-sdxl-1.0/resolve/main/diffusion_pytorch_model.fp16.safetensors?download=true"),
            ("models/controlnet/control_v11p_sd15_inpaint_fp16.safetensors",
             "https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_inpaint_fp16.safetensors"),
            ("models/controlnet/controlnet-union-sdxl-1.0_promax.safetensors",
             "https://huggingface.co/xinsir/controlnet-union-sdxl-1.0/resolve/main/diffusion_pytorch_model_promax.safetensors?download=true"),
            ("models/controlnet/control_lora_rank128_v11f1p_sd15_depth_fp16.safetensors",
             "https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_lora_rank128_v11f1p_sd15_depth_fp16.safetensors?download=true"),
            
            # LoRA Models (cleaned URLs - no hardcoded tokens)
            ("models/loras/PONY_Fernando_Style.safetensors",
             "https://civitai.com/api/download/models/452367"),
            ("models/loras/PONY_Majo.safetensors",
             "https://civitai.com/api/download/models/835055"),
            ("models/loras/PONY_Western_Comic_Art_Style.safetensors",
             "https://civitai.com/api/download/models/871611"),
            ("models/loras/PONY_Incase_unaesthetic_style.safetensors",
             "https://civitai.com/api/download/models/1128016"),
            ("models/loras/Pony_Lora_Water_Color_Anime.safetensors",
             "https://civitai.com/api/download/models/725772"),
            ("models/loras/Pony_Lora_Water_Color.safetensors",
             "https://civitai.com/api/download/models/720004"),
            ("models/loras/SDXL_Pop_Art_Style.safetensors",
             "https://civitai.com/api/download/models/192584"),
            ("models/loras/Pony_Lora_Sketch_Illustration.safetensors",
             "https://civitai.com/api/download/models/882225"),
            ("models/loras/Illustrious_USNR_Style.safetensors",
             "https://civitai.com/api/download/models/959419"),
            ("models/loras/Illustrious_Gennesis.safetensors",
             "https://civitai.com/api/download/models/1219983"),
            ("models/loras/Illustrious_Loras_Hassaku_Shiro_Styles.safetensors",
             "https://civitai.com/api/download/models/1580764"),
            ("models/loras/Illustrious_Loras_Power_Puff_Mix.safetensors",
             "https://civitai.com/api/download/models/1456601"),
            ("models/loras/Illustrious_Loras_Detailer_Tool.safetensors",
             "https://civitai.com/api/download/models/1191626"),
            ("models/loras/Illustrious_loRA_Semi_real_Fantasy_illustrious.safetensors",
             "https://civitai.com/api/download/models/1597800"),
            ("models/loras/Illustrious_loRA_Midjourney_watercolors.safetensors",
             "https://civitai.com/api/download/models/1510865"),
            ("models/loras/Illustrious_loRA_Commix_style.safetensors",
             "https://civitai.com/api/download/models/1227175"),
            ("models/loras/Illustrious_loRA_detailrej.safetensors",
             "https://civitai.com/api/download/models/1396529"),
            ("models/loras/Illustrious_loRA_Vixons_Dappled_Sunlight.safetensors",
             "https://civitai.com/api/download/models/1144547"),
                      ("models/loras/Illustrious_Vixon_Style.safetensors",
             "https://civitai.com/api/download/models/1382407"),
            ("models/loras/Illustrious_MagicalCircleTentacles.safetensors",
             "https://civitai.com/api/download/models/1323341"),
            ("models/loras/Pony_Peoples_Work.safetensors",
             "https://civitai.com/api/download/models/1036362"),
            ("models/loras/Stable_Diffusion_Loras_Detailed_Eyes.safetensors",
             "https://civitai.com/api/download/models/145907"),
            ("models/loras/Stable_Diffusion_Loras_Midjourney_Mimic.safetensors",
             "https://civitai.com/api/download/models/283697"),
            ("models/loras/Stable_Diffusion_Loras_Extremely_Detailed.safetensors",
             "https://civitai.com/api/download/models/258687"),
            ("models/loras/Stable_Diffusion_Loras_Juggernot_Cinematic.safetensors",
             "https://civitai.com/api/download/models/131991"),
            ("models/loras/Stable_Diffusion_Loras_Detail_Tweaker.safetensors",
             "https://civitai.com/api/download/models/135867"),
            ("models/loras/Stable_Diffusion_Loras_Wowifier.safetensors",
             "https://civitai.com/api/download/models/217866"),
            ("models/loras/SDXL_loras_2Steps.safetensors",
             "https://huggingface.co/ByteDance/SDXL-Lightning/resolve/main/sdxl_lightning_2step_lora.safetensors?download=true"),
            ("models/loras/Hyper-SDXL-8steps-CFG-lora.safetensors",
             "https://huggingface.co/ByteDance/Hyper-SD/resolve/main/Hyper-SDXL-8steps-CFG-lora.safetensors"),
            ("models/loras/SDXL_lightning_8_steps.safetensors",
             "https://civitai.com/api/download/models/391999"),
            ("models/loras/SDXL_lightning_2_steps.safetensors",
             "https://civitai.com/api/download/models/391994"),
            ("models/loras/Illustrious_lora_Add_Super_Details.safetensors",
             "https://civitai.com/api/download/models/1622964"),
            
            # Flux LoRA Models
            ("models/loras/Flux_lora_Semirealisticportraitpainting.safetensors",
             "https://civitai.com/api/download/models/978472"),
            ("models/loras/Flux_lora_Velvetv2.safetensors",
             "https://civitai.com/api/download/models/967375"),
            ("models/loras/Flux_lora_RetroAnimeStyle.safetensors",
             "https://civitai.com/api/download/models/806265"),
            ("models/loras/Flux_lora_VelvetMythicFantasyRealistic_Fantasy.safetensors",
             "https://civitai.com/api/download/models/1227179"),
            ("models/loras/Flux_lora_VelvetMythicFantasyGothicLines.safetensors",
             "https://civitai.com/api/download/models/1202162"),
            ("models/loras/Flux_lora_Mezzotint.safetensors",
             "https://civitai.com/api/download/models/757030"),

            # Embeddings
            ("models/embeddings/Pony_Embedding_Negative_Cyber_Realistic.pt",
             "https://civitai.com/api/download/models/1690589"),
            ("models/embeddings/Pony_Embedding_Negative_Stable_Yogi_Pony.pt",
             "https://civitai.com/api/download/models/772342"),
            ("models/embeddings/Pony_Embedding_Positive_Stable_Yogi_Pony.pt",
             "https://civitai.com/api/download/models/2044578"),
        ]
        
        successful_downloads = 0
        failed_downloads = 0
        
        print(f"{Colors.CYAN}Starting downloads for {len(downloads)} models...{Colors.END}")
        
        for filename, url in downloads:
            file_path = self.workspace / filename
            self.ensure_directory(file_path.parent)
            
            # Skip if file already exists
            if file_path.exists():
                print(f"{Colors.YELLOW}Skipping {filename} (already exists){Colors.END}")
                continue
            
            # Prepare download command with authentication
            cmd = self.prepare_download_command(url, file_path)
            
            print(f"\n{Colors.BLUE}Downloading: {filename}{Colors.END}")
            print(f"From: {url[:80]}{'...' if len(url) > 80 else ''}")
            
            try:
                proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, 
                                      stderr=subprocess.STDOUT, text=True)
                self.download_processes.append((filename, proc))
                successful_downloads += 1
            except Exception as e:
                print(f"{Colors.RED}Failed to start download for {filename}: {e}{Colors.END}")
                failed_downloads += 1
        
        print(f"\n{Colors.GREEN}Started {successful_downloads} downloads{Colors.END}")
        if failed_downloads > 0:
            print(f"{Colors.RED}Failed to start {failed_downloads} downloads{Colors.END}")
    
    def install_custom_nodes(self):
        """Install custom nodes"""
        self.print_header("Custom Nodes")
        
        custom_nodes = [
            ("https://github.com/rgthree/rgthree-comfy.git", "rgthree-comfy"),
            ("https://github.com/jitcoder/lora-info.git", "lora-info"),
            ("https://github.com/ltdrdata/ComfyUI-Impact-Pack.git", "ComfyUI-Impact-Pack"),
            ("https://github.com/yolain/ComfyUI-Easy-Use.git", "ComfyUI-Easy-Use"),
            ("https://github.com/ltdrdata/ComfyUI-Manager.git", "ComfyUI-Manager"),
            ("https://github.com/pythongosssss/ComfyUI-Custom-Scripts.git", "ComfyUI-Custom-Scripts"),
            ("https://github.com/john-mnz/ComfyUI-Inspyrenet-Rembg.git", "ComfyUI-Inspyrenet-Rembg"),
            ("https://github.com/justUmen/Bjornulf_custom_nodes.git", "Bjornulf_custom_nodes"),
            ("https://github.com/giriss/comfy-image-saver.git", "comfy-image-saver"),
            ("https://github.com/ltdrdata/ComfyUI-Impact-Subpack.git", "ComfyUI-Impact-Subpack"),
            ("https://github.com/ltdrdata/was-node-suite-comfyui.git", "was-node-suite-comfyui"),
        ]
        
        custom_nodes_dir = self.workspace / "custom_nodes"
        pip_cmd = str(self.venv_path / ("Scripts/pip" if self.is_windows else "bin/pip"))
        
        for repo_url, folder_name in custom_nodes:
            folder_path = custom_nodes_dir / folder_name
            
            print(f"\n{Colors.CYAN}Processing: {folder_name}{Colors.END}")
            
            if folder_path.exists():
                print(f"Updating existing repository...")
                with self._working_directory(folder_path):
                    self.run_command("git pull")
            else:
                print(f"Cloning new repository...")
                with self._working_directory(custom_nodes_dir):
                    self.run_command(f"git clone {repo_url}")
            
            # Install node-specific requirements
            requirements_file = folder_path / "requirements.txt"
            if requirements_file.exists():
                print(f"Installing requirements for {folder_name}...")
                self.run_command(f'"{pip_cmd}" install -r "{requirements_file}"')
        
        time.sleep(2)
        self.clear_screen()
    
    def start_comfyui_server(self):
        """Start ComfyUI server"""
        self.print_header("Starting Server")
        
        python_exe = str(self.venv_path / ("Scripts/python" if self.is_windows else "bin/python"))
        main_script = self.workspace / "main.py"
        
        print(f"{Colors.CYAN}Starting ComfyUI server...{Colors.END}")
        print(f"Server will be accessible at: {self.find_comfyui_url()}")
        
        with self._working_directory(self.workspace):
            subprocess.Popen([python_exe, str(main_script), "--listen"])
        
        time.sleep(2)
        self.clear_screen()
    
    def create_restart_script(self):
        """Create restart script"""
        restart_script_content = f'''#!/usr/bin/env python3
"""
ComfyUI Restart Script
"""
import os
import sys
import subprocess
import platform
from pathlib import Path

# Token configuration (copied from main script)
DEFAULT_CIVITAI_TOKEN = "{self.civitai_token}"
DEFAULT_GITHUB_TOKEN = "{self.github_token}"
DEFAULT_HUGGINGFACE_TOKEN = "{self.huggingface_token}"

def run_command(command, check=True):
    """Run a shell command with error handling"""
    try:
        subprocess.run(command, shell=True, check=check)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {{command}}")
        print(f"Error: {{e}}")
        return False

def main():
    # Get workspace directory
    workspace = Path(__file__).parent / "ComfyUI"
    
    if not workspace.exists():
        print("Error: ComfyUI directory not found!")
        sys.exit(1)
    
    os.chdir(workspace)
    print(f"Changed directory to: {{workspace}}")
    
    # Setup paths
    is_windows = platform.system() == "Windows"
    venv_path = workspace / "venv"
    
    if not venv_path.exists():
        print("Error: Virtual environment not found!")
        sys.exit(1)
    
    # Install/update ffmpeg on Linux
    if not is_windows:
        print("Updating system packages...")
        run_command("sudo apt update && sudo apt install -y ffmpeg")
    
    # Setup Python executables
    python_exe = venv_path / ("Scripts/python" if is_windows else "bin/python")
    pip_exe = venv_path / ("Scripts/pip" if is_windows else "bin/pip")
    
    # Update packages
    print("Updating PyTorch packages...")
    run_command(f'"{{pip_exe}}" install --upgrade torch torchvision torchaudio')
    
    # Start server
    print("Starting ComfyUI server...")
    run_command(f'"{{python_exe}}" main.py --listen')

if __name__ == "__main__":
    main()
'''
        
        restart_script_path = Path(__file__).parent / "ninja_restart.py"
        with open(restart_script_path, "w", newline="\n") as f:
            f.write(restart_script_content)
        
        # Make executable on Unix systems
        if not self.is_windows:
            os.chmod(restart_script_path, 0o755)
        
        print(f"{Colors.GREEN}Restart script created: {restart_script_path}{Colors.END}")
    
    def find_comfyui_url(self):
        """Find ComfyUI access URL"""
        # Check for RunPod environment and get public IP and port
        public_ip = os.environ.get('RUNPOD_PUBLIC_IP')
        external_port = os.environ.get('RUNPOD_TCP_PORT_8188', '8188')
        
        if public_ip:
            return f"http://{public_ip}:{external_port}/"
        
        # Fallback to localhost if not in RunPod environment
        try:
            ip = self.run_command('curl -s https://api.ipify.org', capture_output=True)
            if not ip:
                ip = self.run_command('curl -s ifconfig.me', capture_output=True)
            if not ip:
                ip = "127.0.0.1"
        except:
            ip = "127.0.0.1"
        
        return f"http://{ip}:8188"
    
    def wait_for_downloads(self):
        """Wait for downloads to complete and show progress"""
        if not self.download_processes:
            print(f"{Colors.YELLOW}No downloads to wait for.{Colors.END}")
            return
        
        total = len(self.download_processes)
        
        if RICH_AVAILABLE:
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(bar_width=None),
                "[progress.percentage]{task.percentage:>3.0f}%",
                TimeRemainingColumn(),
                console=console
            ) as progress:
                # Create download progress task
                download_task = progress.add_task(
                    f"[cyan]Downloading {total} models...[/cyan]", 
                    total=total
                )
                
                completed = 0
                while self.download_processes:
                    for i, (filename, proc) in enumerate(self.download_processes.copy()):
                        if proc.poll() is not None:  # Process finished
                            return_code = proc.returncode
                            if return_code == 0:
                                completed += 1
                                progress.update(download_task, advance=1)
                                progress.console.print(f"[green]✓ Completed: {filename}[/green]")
                            else:
                                progress.console.print(f"[red]✗ Failed: {filename} (exit code: {return_code})[/red]")
                            
                            self.download_processes.remove((filename, proc))
                    
                    if self.download_processes:
                        time.sleep(2)  # Check more frequently with visual feedback
                
                progress.console.print(f"\n[bold green]All downloads completed! ({completed}/{total} successful)[/bold green]")
        
        else:
            # Fallback to basic terminal output
            print(f"\n{Colors.CYAN}Waiting for {total} downloads to complete...{Colors.END}")
            completed = 0
            
            while self.download_processes:
                for i, (filename, proc) in enumerate(self.download_processes.copy()):
                    if proc.poll() is not None:  # Process finished
                        return_code = proc.returncode
                        if return_code == 0:
                            completed += 1
                            print(f"{Colors.GREEN}✓ Completed ({completed}/{total}): {filename}{Colors.END}")
                        else:
                            print(f"{Colors.RED}✗ Failed: {filename} (exit code: {return_code}){Colors.END}")
                        
                        self.download_processes.remove((filename, proc))
                
                if self.download_processes:
                    time.sleep(5)  # Check every 5 seconds
            
            print(f"\n{Colors.GREEN}All downloads completed! ({completed}/{total} successful){Colors.END}")
    
    def test_token_authentication(self):
        """Test if token authentication is working"""
        self.print_header("Token Authentication Test")
        
        # Test Civitai token with a small file
        if self.civitai_token:
            test_url = "https://civitai.com/api/download/models/217866"  # Small file
            test_path = self.workspace / "temp" / "token_test.tmp"
            self.ensure_directory(test_path.parent)
            
            cmd = self.prepare_download_command(test_url, test_path)
            print(f"{Colors.BLUE}Testing Civitai token authentication...{Colors.END}")
            
            try:
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
                if result.returncode == 0 and test_path.exists():
                    print(f"{Colors.GREEN}✓ Civitai token authentication successful{Colors.END}")
                    test_path.unlink()  # Clean up test file
                else:
                    print(f"{Colors.RED}✗ Civitai token authentication failed{Colors.END}")
                    print(f"Error: {result.stderr}")
            except Exception as e:
                print(f"{Colors.RED}✗ Civitai token test failed: {e}{Colors.END}")
        else:
            print(f"{Colors.YELLOW}⚠ No Civitai token configured{Colors.END}")
        
        # Similar tests could be added for GitHub and HuggingFace tokens
        time.sleep(2)
        self.clear_screen()
    
    def show_final_summary(self):
        """Show final installation summary"""
        self.print_header("Installation Complete!")
        
        url = self.find_comfyui_url()
        is_runpod = bool(os.environ.get('RUNPOD_PUBLIC_IP'))
        
        print(f"{Colors.GREEN}ComfyUI has been successfully installed!{Colors.END}\n")
        print(f"{Colors.BOLD}Access Information:{Colors.END}")
        if is_runpod:
            print(f"  • RunPod Public URL: {Colors.CYAN}{url}{Colors.END}")
            print(f"  • Public IP: {Colors.CYAN}{os.environ.get('RUNPOD_PUBLIC_IP')}{Colors.END}")
            print(f"  • External Port: {Colors.CYAN}{os.environ.get('RUNPOD_TCP_PORT_8188', '8188')}{Colors.END}")
        else:
            print(f"  • Web Interface: {Colors.CYAN}{url}{Colors.END}")
            print(f"  • Local Access: {Colors.CYAN}http://localhost:8188{Colors.END}")
        
        print(f"\n{Colors.BOLD}Important Files:{Colors.END}")
        print(f"  • Installation Directory: {Colors.YELLOW}{self.workspace}{Colors.END}")
        print(f"  • Restart Script: {Colors.YELLOW}ninja_restart.py{Colors.END}")
        
        print(f"\n{Colors.BOLD}Token Status:{Colors.END}")
        print(f"  • Civitai: {'✓ Configured' if self.civitai_token else '✗ Not configured'}")
        print(f"  • GitHub: {'✓ Configured' if self.github_token else '✗ Not configured'}")
        print(f"  • HuggingFace: {'✓ Configured' if self.huggingface_token else '✗ Not configured'}")
        
        print(f"\n{Colors.BOLD}Next Steps:{Colors.END}")
        print(f"  1. Wait for all model downloads to complete")
        print(f"  2. Access ComfyUI at {Colors.CYAN}{url}{Colors.END}")
        print(f"  3. Use {Colors.YELLOW}ninja_restart.py{Colors.END} to restart the server")
        
        print(f"\n{Colors.GREEN}Happy creating! 🎨{Colors.END}")
    
    def run_installation(self):
        """Run the complete installation process"""
        try:
            self.setup_system_dependencies()
            self.clone_comfyui()
            self.setup_virtual_environment()
            self.create_directory_structure()
            self.test_token_authentication()  # Test tokens before downloading
            self.download_models()
            self.install_custom_nodes()
            self.start_comfyui_server()
            self.create_restart_script()
            self.wait_for_downloads()  # Wait for downloads to complete
            self.show_final_summary()
            
        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}Installation interrupted by user{Colors.END}")
            sys.exit(1)
        except Exception as e:
            print(f"\n{Colors.RED}Installation failed: {e}{Colors.END}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

def main():
    """Main entry point"""
    args = parse_arguments()
    
    # Create installer with optional tokens
    installer = ComfyUIInstaller(
        civitai_token=args.civitai_token,
        github_token=args.github_token,
        huggingface_token=args.huggingface_token
    )
    installer.run_installation()

if __name__ == "__main__":
    main()