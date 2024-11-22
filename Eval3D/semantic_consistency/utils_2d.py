import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules.utils as nn_utils
from typing import Union, List, Tuple
from PIL import Image
import numpy as np
from torchvision import transforms
from pathlib import Path
import types
import math
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA as sklearnPCA

class dino_args:
    load_size = 224
    stride = 14 # stride of first convolution layer. small stride -> higher resolution.
    model_type = "dinov2_vits14"
    facet = "token"
    layer = 11 # for vits layer = 11
    bin = None
    patch_size = 14
    device = 'cuda:0'

class ViTExtractor:
    def __init__(self, model_type: str = 'dino_vits8', stride: int = 4, model: nn.Module = None, device: str = 'cuda'):
        """
        :param model_type: A string specifying the type of model to extract from.
                          [dino_vits8 | dino_vits16 | dino_vitb8 | dino_vitb16 | vit_small_patch8_224 |
                          vit_small_patch16_224 | vit_base_patch8_224 | vit_base_patch16_224]
        :param stride: stride of first convolution layer. small stride -> higher resolution.
        :param model: Optional parameter. The nn.Module to extract from instead of creating a new one in ViTExtractor.
                      should be compatible with model_type.
        """
        self.model_type = model_type
        self.device = device
        if model is not None:
            self.model = model
        else:
            self.model = ViTExtractor.create_model(model_type)

        self.model = ViTExtractor.patch_vit_resolution(self.model, stride=stride)
        self.model.eval()
        self.model.to(self.device)
        self.p = self.model.patch_embed.patch_size
        if type(self.p)==tuple:
            self.p = self.p[0]
        self.stride = self.model.patch_embed.proj.stride

        self.mean = (0.485, 0.456, 0.406) if "dino" in self.model_type else (0.5, 0.5, 0.5)
        self.std = (0.229, 0.224, 0.225) if "dino" in self.model_type else (0.5, 0.5, 0.5)

        self._feats = []
        self.hook_handlers = []
        self.load_size = None
        self.num_patches = None

    @staticmethod
    def create_model(model_type: str) -> nn.Module:
        """
        :param model_type: a string specifying which model to load. [dino_vits8 | dino_vits16 | dino_vitb8 |
                           dino_vitb16 | vit_small_patch8_224 | vit_small_patch16_224 | vit_base_patch8_224 |
                           vit_base_patch16_224]
        :return: the model
        """
        torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
        if 'v2' in model_type:
            model = torch.hub.load('facebookresearch/dinov2', model_type)
            print("dinov2 model created")
        elif 'dino' in model_type:
            model = torch.hub.load('facebookresearch/dino:main', model_type)
        else:  # model from timm -- load weights from timm to dino model (enables working on arbitrary size images).
            temp_model = timm.create_model(model_type, pretrained=True)
            model_type_dict = {
            'vit_small_patch16_224': 'dino_vits16',
            'vit_small_patch8_224': 'dino_vits8',
            'vit_base_patch16_224': 'dino_vitb16',
            'vit_base_patch8_224': 'dino_vitb8'
            }
            model = torch.hub.load('facebookresearch/dino:main', model_type_dict[model_type])
            temp_state_dict = temp_model.state_dict()
            del temp_state_dict['head.weight']
            del temp_state_dict['head.bias']
            model.load_state_dict(temp_state_dict)
        return model
    
    @staticmethod
    def _fix_pos_enc(patch_size: int, stride_hw: Tuple[int, int]):
        """
        Creates a method for position encoding interpolation.
        :param patch_size: patch size of the model.
        :param stride_hw: A tuple containing the new height and width stride respectively.
        :return: the interpolation method
        """
        def interpolate_pos_encoding(self, x: torch.Tensor, w: int, h: int) -> torch.Tensor:
            npatch = x.shape[1] - 1
            N = self.pos_embed.shape[1] - 1
            if npatch == N and w == h:
                return self.pos_embed
            class_pos_embed = self.pos_embed[:, 0]
            patch_pos_embed = self.pos_embed[:, 1:]
            dim = x.shape[-1]
            # compute number of tokens taking stride into account
            w0 = 1 + (w - patch_size) // stride_hw[1]
            h0 = 1 + (h - patch_size) // stride_hw[0]
            assert (w0 * h0 == npatch), f"""got wrong grid size for {h}x{w} with patch_size {patch_size} and 
                                            stride {stride_hw} got {h0}x{w0}={h0 * w0} expecting {npatch}"""
            # we add a small number to avoid floating point error in the interpolation
            # see discussion at https://github.com/facebookresearch/dino/issues/8
            w0, h0 = w0 + 0.1, h0 + 0.1
            patch_pos_embed = nn.functional.interpolate(
                patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
                scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
                mode='bicubic',
                align_corners=False, recompute_scale_factor=False
            )
            assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
            patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
            return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

        return interpolate_pos_encoding
    
    @staticmethod
    def patch_vit_resolution(model: nn.Module, stride: int) -> nn.Module:
        """
        change resolution of model output by changing the stride of the patch extraction.
        :param model: the model to change resolution for.
        :param stride: the new stride parameter.
        :return: the adjusted model
        """
        patch_size = model.patch_embed.patch_size
        if type(patch_size) == tuple:
            patch_size = patch_size[0]
        if stride == patch_size:  # nothing to do
            return model

        stride = nn_utils._pair(stride)
        assert all([(patch_size // s_) * s_ == patch_size for s_ in
                    stride]), f'stride {stride} should divide patch_size {patch_size}'

        # fix the stride
        model.patch_embed.proj.stride = stride
        # fix the positional encoding code
        model.interpolate_pos_encoding = types.MethodType(ViTExtractor._fix_pos_enc(patch_size, stride), model)
        return model
    

    def preprocess(self, image_path: Union[str, Path],
                   load_size: Union[int, Tuple[int, int]] = None, patch_size: int = 14) -> Tuple[torch.Tensor, Image.Image]:
        """
        Preprocesses an image before extraction.
        :param image_path: path to image to be extracted.
        :param load_size: optional. Size to resize image before the rest of preprocessing.
        :return: a tuple containing:
                    (1) the preprocessed image as a tensor to insert the model of shape BxCxHxW.
                    (2) the pil image in relevant dimensions
        """
        def divisible_by_num(num, dim):
            return num * (dim // num)
        pil_image = Image.open(image_path).convert('RGB')
        if load_size is not None:
            pil_image = transforms.Resize(load_size, interpolation=transforms.InterpolationMode.LANCZOS)(pil_image)

            width, height = pil_image.size
            new_width = divisible_by_num(patch_size, width)
            new_height = divisible_by_num(patch_size, height)
            pil_image = pil_image.resize((new_width, new_height), resample=Image.LANCZOS)
            
        prep = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        prep_img = prep(pil_image)[None, ...]
        return prep_img, pil_image
    
    def _get_hook(self, facet: str):
        """
        generate a hook method for a specific block and facet.
        """
        if facet in ['attn', 'token']:
            def _hook(model, input, output):
                self._feats.append(output)
            return _hook

        if facet == 'query':
            facet_idx = 0
        elif facet == 'key':
            facet_idx = 1
        elif facet == 'value':
            facet_idx = 2
        else:
            raise TypeError(f"{facet} is not a supported facet.")

        def _inner_hook(module, input, output):
            input = input[0]
            B, N, C = input.shape
            qkv = module.qkv(input).reshape(B, N, 3, module.num_heads, C // module.num_heads).permute(2, 0, 3, 1, 4)
            self._feats.append(qkv[facet_idx]) #Bxhxtxd
        return _inner_hook
    
    def _register_hooks(self, layers: List[int], facet: str) -> None:
        """
        register hook to extract features.
        :param layers: layers from which to extract features.
        :param facet: facet to extract. One of the following options: ['key' | 'query' | 'value' | 'token' | 'attn']
        """
        for block_idx, block in enumerate(self.model.blocks):
            if block_idx in layers:
                if facet == 'token':
                    self.hook_handlers.append(block.register_forward_hook(self._get_hook(facet)))
                elif facet == 'attn':
                    self.hook_handlers.append(block.attn.attn_drop.register_forward_hook(self._get_hook(facet)))
                elif facet in ['key', 'query', 'value']:
                    self.hook_handlers.append(block.attn.register_forward_hook(self._get_hook(facet)))
                else:
                    raise TypeError(f"{facet} is not a supported facet.")
    
    def _unregister_hooks(self) -> None:
        """
        unregisters the hooks. should be called after feature extraction.
        """
        for handle in self.hook_handlers:
            handle.remove()
        self.hook_handlers = []
    
    def _extract_features(self, batch: torch.Tensor, layers: List[int] = 11, facet: str = 'key') -> List[torch.Tensor]:
        """
        extract features from the model
        :param batch: batch to extract features for. Has shape BxCxHxW.
        :param layers: layer to extract. A number between 0 to 11.
        :param facet: facet to extract. One of the following options: ['key' | 'query' | 'value' | 'token' | 'attn']
        :return : tensor of features.
                  if facet is 'key' | 'query' | 'value' has shape Bxhxtxd
                  if facet is 'attn' has shape Bxhxtxt
                  if facet is 'token' has shape Bxtxd
        """
        B, C, H, W = batch.shape
        self._feats = []
        self._register_hooks(layers, facet)
        _ = self.model(batch)
        self._unregister_hooks()
        self.load_size = (H, W)
        self.num_patches = (1 + (H - self.p) // self.stride[0], 1 + (W - self.p) // self.stride[1])
        return self._feats
    
    def preprocess_pil(self, pil_image):
        """
        Preprocesses an image before extraction.
        :param image_path: path to image to be extracted.
        :param load_size: optional. Size to resize image before the rest of preprocessing.
        :return: a tuple containing:
                    (1) the preprocessed image as a tensor to insert the model of shape BxCxHxW.
                    (2) the pil image in relevant dimensions
        """
        prep = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        prep_img = prep(pil_image)[None, ...]
        return prep_img


    def extract_descriptors(self, batch: torch.Tensor, layer: int = 11, facet: str = 'key',
                            bin: bool = False, include_cls: bool = False) -> torch.Tensor:
        """
        extract descriptors from the model
        :param batch: batch to extract descriptors for. Has shape BxCxHxW.
        :param layers: layer to extract. A number between 0 to 11.
        :param facet: facet to extract. One of the following options: ['key' | 'query' | 'value' | 'token']
        :param bin: apply log binning to the descriptor. default is False.
        :return: tensor of descriptors. Bx1xtxd' where d' is the dimension of the descriptors.
        """
        assert facet in ['key', 'query', 'value', 'token'], f"""{facet} is not a supported facet for descriptors. 
                                                             choose from ['key' | 'query' | 'value' | 'token'] """
        self._extract_features(batch, [layer], facet)
        
        x = self._feats[0]
        if facet == 'token':
            x.unsqueeze_(dim=1) #Bx1xtxd
        if not include_cls:
            x = x[:, :, 1:, :]  # remove cls token
        else:
            assert not bin, "bin = True and include_cls = True are not supported together, set one of them False."
        if not bin:
            desc = x.permute(0, 2, 3, 1).flatten(start_dim=-2, end_dim=-1).unsqueeze(dim=1)  # Bx1xtx(dxh)
        else:
            desc = self._log_bin(x)
        return desc



def extract_dino_features(dino_extractor, extractor_args, img_path, output_path=None):
    with torch.no_grad():
        image_batch, image_pil = dino_extractor.preprocess(img_path, extractor_args.load_size, extractor_args.patch_size)
        # print(f"Image {img_path} is preprocessed to tensor of size {image_batch.shape}.")
        descriptors = dino_extractor.extract_descriptors(image_batch.to(extractor_args.device), extractor_args.layer, extractor_args.facet, extractor_args.bin)
    # print(f"Descriptors are of size: {descriptors.shape}")
    if output_path is not None:
        torch.save(descriptors, extractor_args.output_path)
        print(f"Descriptors saved to: {extractor_args.output_path}")
    return descriptors
    


def apply_pca(features, n_components=4):
    # the first component is to seperate the object from the background
    pca = sklearnPCA(n_components=n_components)
    features = pca.fit_transform(features.cpu().numpy()) # shape (7200,3)
    return features

def dino_vis_pca_mask(feature_list, mask_list, normalize_pca_output=False, pca_n_components=4):
    masked_feature_list = []
    num_patches = int(math.sqrt(feature_list[0].shape[-2]))
    # pca_n_components = 4

    for feature, mask in zip(feature_list, mask_list):
        feature_shape = feature.shape
        feature = feature.permute(1,0).reshape(-1,num_patches,num_patches)
        mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(num_patches, num_patches), mode='nearest').squeeze()
        feature = feature * mask[None].repeat(feature.shape[0],1,1)
        feature = feature.reshape(feature_shape[-1],feature_shape[-2]).permute(1,0)
        masked_feature_list.append(feature)
    
    features = torch.cat(masked_feature_list, dim=0)
    pca_features = apply_pca(features, pca_n_components)

    pca_features_list = []
    for idx, feature in enumerate(feature_list):
        feature = pca_features[idx*feature.shape[0]:(idx+1)*feature.shape[0]]
        # print(feature.shape, feature.min(), feature.max(), "pca feature: shape, min, max")
        if normalize_pca_output: 
            feature = feature / np.linalg.norm(feature, axis=-1)[:,None]
            feature = (feature + 1.) / 2.
            # print(feature.min(), feature.max(), "feature min max")
        pca_features_list.append(feature)
    return pca_features_list

def visualize_pca_features(pca_features_list, vis_img_idx):
    pca_n_components = 4
    num_patches = int(math.sqrt(pca_features_list[0].shape[-2]))
    fig, axes = plt.subplots(4, len(vis_img_idx), figsize=(10, 14))
    for show_channel in range(pca_n_components):
        if show_channel==0:
            continue
        
        axes_idx = 0
        for idx, feature in enumerate(pca_features_list):
            if idx not in vis_img_idx: continue
            # feature = pca_features[idx*feature.shape[0]:(idx+1)*feature.shape[0]].copy()
            feature[:, show_channel] = (feature[:, show_channel] - feature[:, show_channel].min()) / (feature[:, show_channel].max() - feature[:, show_channel].min())
            pca_features_list[idx][:, show_channel] = feature[:, show_channel]

            feature_first_channel = feature[:, show_channel].reshape(num_patches, num_patches)

            im = axes[show_channel-1, axes_idx].imshow(feature_first_channel)
            axes[show_channel-1, axes_idx].axis('off')
            axes[show_channel-1, axes_idx].set_title('Image {} - Channel {}'.format(idx, show_channel), fontsize=14)
            axes_idx += 1
        fig.colorbar(im, ax=axes[show_channel-1, :], orientation='vertical')

    axes_idx = 0
    for idx, feature in enumerate(pca_features_list):
        if idx not in vis_img_idx: continue
        feature = feature[:, 1:4].reshape(num_patches,num_patches, 3)
        
        im = axes[3, axes_idx].imshow(feature)
        axes[3, axes_idx].axis('off')
        axes[3, axes_idx].set_title('Image {} - All Channels'.format(idx), fontsize=14)
        axes_idx += 1

    fig.colorbar(im, ax=axes[3, :], orientation='vertical')
    plt.tight_layout()
    plt.show()
    