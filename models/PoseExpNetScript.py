import os

import argparse
import torch
import torch.nn as nn
from torch import sigmoid
from torch.nn.init import xavier_uniform_, zeros_

parser = argparse.ArgumentParser(description='~',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("pretrained_posenet", type=str, help="pretrained PoseNet path")


def conv(in_planes, out_planes, kernel_size=3):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, stride=2),
        nn.ReLU(inplace=True)
    )


def upconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1),
        nn.ReLU(inplace=True)
    )


class PoseExpNetScript(torch.jit.ScriptModule):

    def __init__(self, nb_ref_imgs=2):
        super(PoseExpNetScript, self).__init__()
        self.nb_ref_imgs = nb_ref_imgs
        # self.output_exp = output_exp

        #conv_planes = [16, 32, 64, 128, 256, 256, 256]
        self.conv1 = conv(3 * (1 + self.nb_ref_imgs), 16, kernel_size=7)
        self.conv2 = conv(16, 32, kernel_size=5)
        self.conv3 = conv(32, 64)
        self.conv4 = conv(64, 128)
        self.conv5 = conv(128, 256)
        self.conv6 = conv(256, 256)
        self.conv7 = conv(256, 256)

        self.pose_pred = nn.Conv2d(256, 6 * self.nb_ref_imgs, kernel_size=1, padding=0)

        # if self.output_exp:
        #     #upconv_planes = [256, 128, 64, 32, 16]
        #     self.upconv5 = upconv(256, 256)
        #     self.upconv4 = upconv(256, 128)
        #     self.upconv3 = upconv(128, 64)
        #     self.upconv2 = upconv(64, 32)
        #     self.upconv1 = upconv(32, 16)
        #
        #     self.predict_mask4 = (nn.Conv2d(128, self.nb_ref_imgs, kernel_size=3, padding=1))
        #     self.predict_mask3 = (nn.Conv2d(64, self.nb_ref_imgs, kernel_size=3, padding=1))
        #     self.predict_mask2 = (nn.Conv2d(32, self.nb_ref_imgs, kernel_size=3, padding=1))
        #     self.predict_mask1 = (nn.Conv2d(16, self.nb_ref_imgs, kernel_size=3, padding=1))

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    zeros_(m.bias)

    # forward函数采用特殊形式的控制流（if....else...），不能采用tracing的方法；只能用注释的方法得到script model
    # https://blog.csdn.net/tlzhatao/article/details/86555269?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-1.nonecase&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-1.nonecase
    @torch.jit.script_method
    def forward(self, target_image, ref_imgs):
        # if type(ref_imgs) is torch.Tensor:
        #     ref_imgs=ref_imgs.split(3,1)
        # assert(len(ref_imgs) == self.nb_ref_imgs)
        input = [target_image]
        input.extend(ref_imgs)
        input = torch.cat(input, 1)
        # input=target_image_ref_imgs
        # assert (len(ref_imgs) == self.nb_ref_imgs)
        # input = [target_image]
        # input.extend(ref_imgs)
        out_conv1 = self.conv1(input)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)
        out_conv5 = self.conv5(out_conv4)
        out_conv6 = self.conv6(out_conv5)
        out_conv7 = self.conv7(out_conv6)

        pose = self.pose_pred(out_conv7)
        pose = pose.mean(3).mean(2)
        pose = 0.01 * pose.view(pose.size(0), self.nb_ref_imgs, 6)

        # if self.output_exp:
        #     # out_upconv5 = self.upconv5(out_conv5)[:, :, 0:out_conv4.size(2), 0:out_conv4.size(3)]
        #     # out_upconv4 = self.upconv4(out_upconv5)[:, :, 0:out_conv3.size(2), 0:out_conv3.size(3)]
        #     # out_upconv3 = self.upconv3(out_upconv4)[:, :, 0:out_conv2.size(2), 0:out_conv2.size(3)]
        #     # out_upconv2 = self.upconv2(out_upconv3)[:, :, 0:out_conv1.size(2), 0:out_conv1.size(3)]
        #     # out_upconv1 = self.upconv1(out_upconv2)[:, :, 0:input.size(2), 0:input.size(3)]
        #     #
        #     # exp_mask4 = sigmoid(self.predict_mask4(out_upconv4))
        #     # exp_mask3 = sigmoid(self.predict_mask3(out_upconv3))
        #     # exp_mask2 = sigmoid(self.predict_mask2(out_upconv2))
        #     # exp_mask1 = sigmoid(self.predict_mask1(out_upconv1))
        #
        # else:
        # exp_mask4 = None
        # exp_mask3 = None
        # exp_mask2 = None
        exp_mask1 = None

        # if self.training:
        #     return [exp_mask1, exp_mask2, exp_mask3, exp_mask4], pose
        # else:
        return exp_mask1, pose


args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

seq_length=3
weights = torch.load(args.pretrained_posenet)

traced_script_module = PoseExpNetScript(nb_ref_imgs=seq_length - 1).to(device)
traced_script_module.load_state_dict(weights['state_dict'], strict=False)
traced_script_module.eval()
traced_script_module.save('modelScript.pt')
print("Script saved")