import numpy as np
import torch

from octpred.models.resnet import resnet50

def test_resnet():
    resnet = resnet50(num_classes=2)
    x = torch.from_numpy(np.random.rand(1, 3, 128, 128).astype(np.float32))
    out = resnet(x)
    assert out.detach().shape == (1, 2), "output shape {} do not match expected {}".format(out.detach().shape, (1, 2))