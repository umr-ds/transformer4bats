import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import timm
from timm.models.layers import to_2tuple, trunc_normal_


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class DeitBat(nn.Module):

    def __init__(self, label_dim=2, fstride=10, tstride=10, input_fdim=128, input_tdim=1024,
                 model_size='base384', verbose=True):

        super(DeitBat, self).__init__()
        assert timm.__version__ == '0.4.5', 'Please use timm == 0.4.5, the code might not be compatible with newer ' \
                                            'versions. '

        # override timm input shape restriction
        timm.models.vision_transformer.PatchEmbed = PatchEmbed

        # if AudioSet pretraining is not used (but ImageNet pretraining may still apply)
        if model_size == 'tiny224':
            self.v = timm.create_model('vit_deit_tiny_distilled_patch16_224', pretrained=False)
        elif model_size == 'small224':
            self.v = timm.create_model('vit_deit_small_distilled_patch16_224', pretrained=False)
        elif model_size == 'base224':
            self.v = timm.create_model('vit_deit_base_distilled_patch16_224', pretrained=False)
        elif model_size == 'base384':
            self.v = timm.create_model('vit_deit_base_distilled_patch16_384', pretrained=False)
        else:
            raise Exception('Model size must be one of tiny224, small224, base224, base384.')
        self.original_num_patches = self.v.patch_embed.num_patches
        self.oringal_hw = int(self.original_num_patches ** 0.5)
        self.original_embedding_dim = self.v.pos_embed.shape[2]
        self.mlp_head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim),
                                      nn.Linear(self.original_embedding_dim, label_dim))

        # automatically get the intermediate shape
        f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
        num_patches = f_dim * t_dim
        self.v.patch_embed.num_patches = num_patches
        if verbose:
            print('frequncey stride={:d}, time stride={:d}'.format(fstride, tstride))
            print('number of patches={:d}'.format(num_patches))

        # the linear projection layer
        new_proj = torch.nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
        self.v.patch_embed.proj = new_proj

        new_pos_embed = nn.Parameter(
            torch.zeros(1, self.v.patch_embed.num_patches + 2, self.original_embedding_dim))
        self.v.pos_embed = new_pos_embed
        trunc_normal_(self.v.pos_embed, std=.02)

    def get_shape(self, fstride, tstride, input_fdim=128, input_tdim=1024):
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim

    @autocast()
    def forward(self, x):
        """
        :param x: the input spectrogram, expected shape: (batch_size, time_frame_num, frequency_bins)
        :return: prediction
        """
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)

        B = x.shape[0]
        x = self.v.patch_embed(x)
        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        for blk in self.v.blocks:
            x = blk(x)
        x = self.v.norm(x)
        x = (x[:, 0] + x[:, 1]) / 2

        x = self.mlp_head(x)
        return x
