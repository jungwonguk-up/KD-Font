from functools import partial
from kfp.components import create_component_from_func
from kfp.dsl import ContainerOp, pipeline, Condition
from kfp import onprem, compiler
from kfp.components import load_component_from_url

@partial(create_component_from_func,
        base_image='python:3.10',
        packages_to_install=[])    
def make_mar_handle():
    import os
    import json
    os.mkdir('model-store')
    config_json = json.dumps({\
            "diffusion_serve_test": {\
                "1.0": {\
                    "defaultVersion": true,\
                    "marName": "diffusion_serve_test.mar",\
                    "minWorkers": 2,\
                    "maxWorkers": 12,\
                    "batchSize": 25,\
                    "maxBatchDelay": 1000,\
                    "responseTimeout": 1200\
                    }\
                }\
            }
        )
    config = {
        "inference_address":"http://192.168.0.80:9334",
        "management_address":"http://192.168.0.80:9335",
        "metrics_address":"http://192.168.0.80:9336",
        "enable_envvars_config":True,
        "model_store":"/home/hojun/Documents/code/kubeflow/model-store",
        "install_py_dep_per_model":True,
        "models": config_json
    }
    if not os.path.exists("pvc/model-store/config"):
        os.mkdir("pvc/model-store/config")
    with open("pvc/model-store/config/config.properties", "w") as f:
        for i, j in config.items():
            f.write(f"{i}={j}\n")
        f.close()
    x = '''
    from ts.torch_handler.base_handler import BaseHandler
    import os
    import tqdm
    import math
    import random
    import json
    import pandas as pd
    import numpy as np
    from PIL import Image

    import torch, torchvision
    from torch import optim
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset,Dataset
    from torchvision.transforms.functional import to_pil_image

    from model import UNet, Diffusion, CharAttar
    from functools import partial
    from utils import load_yaml

    from PIL import Image

    os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




    class DiffusionFontGenerateHandler(BaseHandler):#why use BaseHandler and abc
        def __init__(self):
            super(DiffusionFontGenerateHandler,self).__init__()
            self.config = load_yaml("config.yaml")
            self.initialized = False
            self.device = f"cuda:{self.config['gpu_num']}"
        
        def initialize(self,context):
            input_size = 64
            self.manifest = context.manifest
            properties = context.system_properties
            model_dir = properties.get("model_dir")
            serialized_file = self.manifest['model']['serializedFile']
            model_pt_path = os.path.join(model_dir, serialized_file)
            if not os.path.isfile(model_pt_path):
                raise RuntimeError("Missing the model.pt file")
            

            init_model = UNet().to(self.device)
            ckpt = torch.load(model_pt_path)
            init_model.load_state_dict(ckpt)
            self.model = init_model

            self.diffusion = Diffusion(first_beta=1e-4,
                                end_beta=0.02,
                                noise_step=1000,
                                beta_schedule_type='linear',
                                img_size=input_size,
                                device=self.device)
            
            self.initialized = True
        def preprocess(self,sample_img_path,contents_ch):
            transforms = torchvision.transforms.Compose([
                # torchvision.transforms.Resize((input_size,input_size)),
                torchvision.transforms.Grayscale(num_output_channels=1),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5), (0.5))
            ])
            sampleImage_len = len(contents_ch)

            # print(data)
            sample_img = Image.open(sample_img_path)
            sample_img = transforms(sample_img).to(self.device)
            sample_img = torch.unsqueeze(sample_img,1)
            sample_img = sample_img.repeat(sampleImage_len, 1, 1, 1)
            # print(len(sample_img))
            return sample_img
            
        def inference(self,sample_img,contents_ch,id):
            save_path = "./data"
            
            charAttar = CharAttar(num_classes=self.config['num_classes'],device=self.device,style_path=self.config['style_path'])
            x = self.diffusion.portion_sampling(model=self.model,sampling_chars=contents_ch,charAttar=charAttar,sample_img=sample_img,batch_size=4)
            os.makedirs(save_path,exist_ok=True)
            for img,ch in zip(x,contents_ch):
                pillow_img = to_pil_image(img)
                pillow_img.save(os.path.join(save_path,id)+f"_{ch}.png")
            
            return x




    _service = DiffusionFontGenerateHandler()

    def handle(data,context):
        try:
            if not _service.initialized:
                _service.initialize(context)
            if data is None:
                return None
            print(data)
            print(data[0]['body'])
            data = data[0]['body']['inputs']
            sample_img_path = data["cropped_img_path"]
            id = data["id"]
            contents_ch = data["text"]
            
            sample_img = _service.preprocess(sample_img_path=sample_img_path,contents_ch=contents_ch)
            data = _service.inference(sample_img,contents_ch,id)
            return [data.tolist()]
            
        except Exception as e:
            raise e
    
    

    '''
    with open("pvc/model-store/handler.py", "w") as f:
        f.write(x)
    f.close()
    print("Saving handler.py complete !!")

def create_marfile():
    return ContainerOp(
        name="Creating Marfile",
        command=["/bin/sh"],
        image="python:3.9",
        arguments=[
            "-c",
            "cd pvc/torch_model; pip install torchserve torch-model-archiver torch-workflow-archiver; torch-model-archiver --model-name torch-model --version 1.0 --serialized-file pytorch_model.bin --handler handler.py --extra-files config.json,vocab.txt --force; mkdir model-store; mv -f torch-model.mar model-store"
        ],  # pip install => create mar file => make model_store folder => mv marfile to model_store
    )
    
def create_inference_model():
    kserve_op = load_component_from_url('https://raw.githubusercontent.com/kubeflow/pipelines/'
                                               'master/components/kserve/component.yaml')
    model_name = "diffusion_serve"
    namespace = "kubeflow-user-example-com"
    model_uri = "pvc://diffusion/model-store"
    framework="pytorch"
    return kserve_op(action="apply",
              model_name=model_name,
              model_uri=model_uri,
              namespace=namespace,
              framework=framework)

@pipeline(
   name='korea diffusion train pipeline',
   description='An example pipeline that performs arithmetic calculations.'
)
def my_pipeline():
   
    handle = make_mar_handle()
    handle.apply(onprem.mount_pvc(pvc_name="diffusion", volume_name="test-lee", volume_mount_path="pvc"))
    handle.execution_options.caching_strategy.max_cache_staleness = "P0D"
    handle.set_display_name("Make A hanlder file & config.properties file")

    mar = create_marfile()
    mar.apply(onprem.mount_pvc(pvc_name="diffusion", volume_name="test-lee", volume_mount_path="pvc"))
    mar.set_display_name("Make Mar file for torchserve")
    mar.after(handle)
    
    
    inference = create_inference_model()
    inference.apply(onprem.mount_pvc(pvc_name="diffusion", volume_name="test-lee", volume_mount_path="pvc"))
    inference.after(mar)
    

    
if __name__ == "__main__": 
    compiler.Compiler().compile(my_pipeline, "KD_Font_kubeflow_serving.yaml")

