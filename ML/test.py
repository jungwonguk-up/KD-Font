import os, wandb, sys
import torch, torchvision
from PIL import Image

prj_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(prj_dir)

from modules.diffusion import Diffusion
from models.utils import UNet
from modules.utils import CharAttar , load_yaml

if __name__ == '__main__':
    # Load config
    config_path = os.path.join(prj_dir, 'config', 'test.yaml')
    config = load_yaml(config_path)
    
    # Set path
    sample_img_path = os.path.join(prj_dir, config['sample_img_path'])
    style_path = os.path.join(prj_dir,config['style_path'])
    model_path = os.path.join(prj_dir, config['model_path'])
    
    # Set wandb
    # wandb.init(project="diffusion_font_test_sampling", config={
    #             "learning_rate": 0.0003,
    #             "architecture": "UNET",
    #             "dataset": "HOJUN_KOREAN_FONT64",
    #             "notes":"content, yes_stoke, non_style/ 64 x 64, 420 dataset"
    #             },
    #         name = "self-attetnion condtion content stroke style_sampling 나눔손글씨강인한위로_갊") #####
    wandb.init(mode="disabled")
    
    # Set Device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config['gpu_num'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set Model
    model = UNet().to(device)
    ckpt = torch.load(model_path)
    model.load_state_dict(ckpt)

    # Set diffusion
    diffusion = Diffusion(first_beta=1e-4,
                              end_beta=0.02,
                              noise_step=1000,
                              beta_schedule_type='linear',
                              img_size=config['input_size'],
                              device=device)
    
    # Set transforms
    transforms = torchvision.transforms.Compose([
        # torchvision.transforms.Resize((input_size,input_size)),
        torchvision.transforms.Grayscale(num_output_channels=1),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5), (0.5))
    ])
    
    # Make sample image
    sample_img = Image.open(sample_img_path)
    sample_img = transforms(sample_img).to(device)
    sample_img = torch.unsqueeze(sample_img,1)
    sample_img = sample_img.repeat(len(config['sampling_chars']), 1, 1, 1)
    
    # Load Condition Making Class
    charAttar = CharAttar(num_classes=config['num_classes'],device=device,style_path=style_path)
    
    # Inference
    sampled_images = diffusion.portion_sampling(model, config['sampling_chars'], charAttar=charAttar, sample_img=sample_img,batch_size=config['batch_size'])