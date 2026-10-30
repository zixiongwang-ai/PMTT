"""Official frozen DINOv3, BF16 forwards and FP32 exported patch states."""
import sys
import torch
from .settings import ARCH, backbone_paths
from .io import ensure_finite

class Backbone:
    """Final normalized patch tokens for S, B, L or H+; patch size 16."""
    def __init__(self, scale, source=None, checkpoint=None):
        if source is None or checkpoint is None:
            source, checkpoint = backbone_paths(scale)
        sys.path.insert(0, str(source))
        from dinov3.hub import backbones
        self.scale = scale
        self.width = ARCH[scale][1]
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.model = getattr(backbones, 'dinov3_' + ARCH[scale][0])(pretrained=False)
        state = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
        self.model.load_state_dict(state, strict=True)
        self.model.eval().requires_grad_(False).to('cuda')

    @torch.inference_mode()
    def __call__(self, images):
        network = self.model
        with torch.autocast('cuda', dtype=torch.bfloat16):
            tokens, (height, width) = network.prepare_tokens_with_masks(images)
            rope = network.rope_embed(H=height, W=width)
            for block in network.blocks:
                tokens = block(tokens, rope)
            final = network.norm(tokens)[:, network.n_storage_tokens + 1:].float()
        if final.shape[1:] != (height * width, self.width):
            raise ValueError('Patch grid mismatch')
        return dict(c=ensure_finite(final), hw=(height, width))
