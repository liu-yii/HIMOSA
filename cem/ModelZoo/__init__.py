import os
import torch
from collections import OrderedDict
MODEL_DIR = './cem/ModelZoo/models/'


NN_LIST = [
    'HiTNet',
    'ESTNet',
    'CATANet',
    'ATDNet',
    'HIMOSA',
    'PFTNet',
]


MODEL_LIST = {
    'HiTNet': {
        'Base': 'HiTNet.pth',
    },
    'ESTNet': {
        'Base': 'ESTNet.pth',
    },
    'CATANet': {
        'Base': 'CATANet.pth',
    },
    'ATDNet': {
        'Base': 'ATDNet.pth',
    },
    'HIMOSA': {
        'Base': 'HIMOSA_x4.pth',
    },
    'PFTNet': {
        'Base': 'PFTNet.pth',
    },

}

def print_network(model, model_name):
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print('Network [%s] was created. Total number of parameters: %.1f kelo. '
          'To see the architecture, do print(network).'
          % (model_name, num_params / 1000))



def get_model(model_name, factor=4, num_channels=3):
    """
    All the models are defaulted to be X4 models, the Channels is defaulted to be RGB 3 channels.
    :param model_name:
    :param factor:
    :param num_channels:
    :return:
    """
    print(f'Getting SR Network {model_name}')
    if model_name.split('-')[0] in NN_LIST:
        if model_name == 'HIMOSA':
            from basicsr.archs.himosa_arch import HIMOSAv1
            net = HIMOSAv1(upscale= 4,
                            in_chans= 3,
                            img_size= 64,
                            embed_dim= 60,
                            depths= [6, 6, 6, 6, ],
                            num_heads= [6, 6, 6, 6, ],
                            mosa_heads= [8, 8, 8, 8],
                            sparsity= [1, 1, 2, 4, 8, 16],
                            base_win_size= [8, 8],
                            reducted_dim= 8,
                            convffn_kernel_size= 7,
                            img_range= 1.,
                            mlp_ratio= 1,
                            upsampler= 'pixelshuffledirect',
                            resi_connection= '1conv',
                            use_checkpoint= False,
                            hier_win_ratios= [0.5,1,2,4,6,8])
        elif model_name == 'PFTNet':
            from basicsr.archs.pft_arch import PFTNet
            net = PFTNet(upscale= 4,
                        in_chans= 3,
                        img_size= 64,
                        embed_dim= 52,
                        depths= [ 2, 4, 6, 6, 6 ],
                        num_heads= 4,
                        num_topk= [ 1024, 1024,
                                    256, 256, 256, 256,
                                    128, 128, 128, 128, 128, 128,
                                    64, 64, 64, 64, 64, 64,
                                    32, 32, 32, 32, 32, 32 ],
                        window_size= 32,
                        convffn_kernel_size= 7,
                        img_range= 1.,
                        mlp_ratio= 1,
                        upsampler= 'pixelshuffledirect',
                        resi_connection= '1conv',
                        use_checkpoint= False)
        elif model_name == 'HiTNet':
            from basicsr.archs.hit_arch import HiT_SRF
            net = HiT_SRF(upscale= 4,
                        in_chans= 3,
                        img_size= 64,
                        base_win_size= [8,8],
                        img_range= 1.,
                        depths= [6,6,6,6],
                        embed_dim= 60,
                        num_heads= [6,6,6,6],
                        expansion_factor= 2,
                        resi_connection= '1conv',
                        hier_win_ratios= [0.5,1,2,4,6,8],
                        upsampler= 'pixelshuffledirect')
        elif model_name == 'ESTNet':
            from basicsr.archs.est_arch import ESTNet
            net = ESTNet(upscale= 4,
                        in_chans= 3,
                        img_size= 64,
                        embed_dim= 240,
                        depths= [6, 6, 6 ],
                        num_heads= [6, 6, 6],
                        window_size= 8,
                        img_range= 1.,
                        mlp_ratio= 1,
                        upsampler= 'pixelshuffledirect',
                        resi_connection= '1conv',
                        use_checkpoint= False
                        )
        elif model_name == 'CATANet':
            from basicsr.archs.catanet_arch import CATANet
            net = CATANet(upscale=4)
        elif model_name == 'ATDNet':
            from basicsr.archs.atd_arch import ATD
            net = ATD(upscale= 4,
                    in_chans= 3,
                    img_size= 64,
                    embed_dim= 48,
                    depths= [6, 6, 6, 6, ],
                    num_heads= [4, 4, 4, 4, ],
                    window_size= 16,
                    category_size= 128,
                    num_tokens= 64,
                    reducted_dim= 8,
                    convffn_kernel_size= 7,
                    img_range= 1.,
                    mlp_ratio= 1,
                    upsampler= 'pixelshuffledirect',
                    resi_connection= '1conv',
                    use_checkpoint= False)
        else:
            raise NotImplementedError(f'Not yet implemented {model_name}, please check the model name in ModelZoo/ModelList.py')

        print_network(net, model_name)
        return net
    else:
        raise NotImplementedError()


def load_model(model_loading_name):
    """
    :param model_loading_name: model_name-training_name
    :return:
    """
    splitting = model_loading_name.split('@')
    if len(splitting) == 1:
        model_name = splitting[0]
        training_name = 'Base'
    elif len(splitting) == 2:
        model_name = splitting[0]
        training_name = splitting[1]
    else:
        raise NotImplementedError()
    assert model_name in NN_LIST or model_name in MODEL_LIST.keys(), 'check your model name before @'
    net = get_model(model_name)
    state_dict_path = os.path.join(MODEL_DIR, MODEL_LIST[model_name][training_name])
    print(f'Loading model {state_dict_path} for {model_name} network.')
    if model_name == 'ATDNet' or model_name == 'PFTNet' or model_name == 'HIMOSA' or model_name == 'HiTNet' or model_name == 'ESTNet' or model_name == 'CATANet':
        param_key_g = 'params'
        state_dict = torch.load(state_dict_path, map_location='cpu')
        net.load_state_dict(state_dict[param_key_g] if param_key_g in state_dict.keys() else state_dict, strict=True)
    else:
        state_dict = torch.load(state_dict_path, map_location='cpu')
        net.load_state_dict(state_dict)

    
    return net