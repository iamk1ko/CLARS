import unittest
import torch
from rsmamba import RSMamba
from replknet import create_RepLKNet31B


class TestRSMambaModel(unittest.TestCase):
    def test_rsmamba_model(self):
        model = RSMamba(arch='small', out_type='avg_featmap', img_size=256, num_classes=1000, patch_cfg=dict(stride=8),
                        init_cfg=[dict(type='Kaiming', layer='Conv2d', mode='fan_in', nonlinearity='linear')]).cuda()
        model.init_weights()
        model.train()

        inputs = torch.randn(1, 3, 256, 256).cuda()
        outputs = model(inputs)
        print(outputs.shape)
        self.assertEqual(outputs.shape, torch.Size([1, 1000]))


class TestRepLKNet(unittest.TestCase):
    def test_rep_lknet(self):
        model = create_RepLKNet31B(small_kernel_merged=False, use_checkpoint=False).cuda()
        model.eval()
        x = torch.randn(2, 3, 224, 224).cuda()
        origin_y = model(x)
        print(origin_y.shape)
        self.assertEqual(origin_y.shape, torch.Size([2, 1000]))


if __name__ == '__main__':
    unittest.main()
