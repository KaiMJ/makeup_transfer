import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


####################################################################
# ------------------------- SPL LOSS -------------------------------
####################################################################
class SPLoss(nn.Module):
    def __init__(self):
        super(SPLoss, self).__init__()

    def forward(self, input, reference):
        temp_a = (input / torch.norm(input, dim=2, keepdim=True)) * (reference / torch.norm(reference, dim=2, keepdim=True))
        temp_b = (input / torch.norm(input, dim=3, keepdim=True)) * (reference / torch.norm(reference, dim=3, keepdim=True))
        
        a = torch.sum(temp_a)
        b = torch.sum(temp_b)
        B, c, h, w = input.shape

        return -(a + b) / h
    

class GPLoss(nn.Module):
    def __init__(self):
        super(GPLoss, self).__init__()
        self.trace = SPLoss()  # Please define or import SPLoss() before using

    def get_image_gradients(self, input):
        f_v_1 = input[:, :, :, :-1]
        f_v_2 = input[:, :, :, 1:]
        f_v = f_v_1 - f_v_2

        f_h_1 = input[:, :, :-1, :]
        f_h_2 = input[:, :, 1:, :]
        f_h = f_h_1 - f_h_2

        return f_v, f_h

    def forward(self, input, reference):
        ## comment these lines when your inputs and outputs are in [0,1] range already
        input = (input + 1) / 2
        reference = (reference + 1) / 2

        input_v, input_h = self.get_image_gradients(input)
        ref_v, ref_h = self.get_image_gradients(reference)

        trace_v = self.trace(input_v, ref_v)
        trace_h = self.trace(input_h, ref_h)
        return trace_v + trace_h
    

class CPLoss(nn.Module):
    def __init__(self, rgb=True, yuv=True, yuvgrad=True):
        super(CPLoss, self).__init__()
        self.rgb = rgb
        self.yuv = yuv
        self.yuvgrad = yuvgrad
        self.trace = SPLoss()  # Please define or import SPLoss() before using
        self.trace_YUV = SPLoss()  # Please define or import SPLoss() before using

    def get_image_gradients(self, input):
        f_v_1 = input[:, :, :, :-1]
        f_v_2 = input[:, :, :, 1:]
        f_v = f_v_1 - f_v_2

        f_h_1 = input[:, :, :-1, :]
        f_h_2 = input[:, :, 1:, :]
        f_h = f_h_1 - f_h_2

        return f_v, f_h

    def to_YUV(self, input):
        return torch.cat(((0.299 * input[:, 0, :, :].unsqueeze(1) +
                           0.587 * input[:, 1, :, :].unsqueeze(1) +
                           0.114 * input[:, 2, :, :].unsqueeze(1)), 
                           0.493 * (input[:, 2, :, :].unsqueeze(1) - (
                           0.299 * input[:, 0, :, :].unsqueeze(1) +
                           0.587 * input[:, 1, :, :].unsqueeze(1) +
                           0.114 * input[:, 2, :, :].unsqueeze(1))),
                           0.877 * (input[:, 0, :, :].unsqueeze(1) - (
                           0.299 * input[:, 0, :, :].unsqueeze(1) +
                           0.587 * input[:, 1, :, :].unsqueeze(1) +
                           0.114 * input[:, 2, :, :].unsqueeze(1)))), dim=1)

    def forward(self, input, reference):
        ## comment these lines when you inputs and outputs are in [0,1] range already
        input = (input + 1) / 2
        reference = (reference + 1) / 2
        total_loss = 0
        if self.rgb:
            total_loss += self.trace(input, reference)
        if self.yuv:
            input_yuv = self.to_YUV(input)
            reference_yuv = self.to_YUV(reference)
            total_loss += self.trace(input_yuv, reference_yuv)
        if self.yuvgrad:
            input_yuv = self.to_YUV(input)
            reference_yuv = self.to_YUV(reference)
            input_v, input_h = self.get_image_gradients(input_yuv)
            ref_v, ref_h = self.get_image_gradients(reference_yuv)

            total_loss += self.trace(input_v, ref_v)
            total_loss += self.trace(input_h, ref_h)

        return total_loss

####################################################################
# ------------------------- GANLoss -------------------------------
####################################################################

class GANLoss(nn.Module):
    def __init__(self, mode="lsgan", reduction='mean'):
        super(GANLoss, self).__init__()
        if mode == "lsgan":
            self.loss = nn.MSELoss(reduction=reduction)
        elif mode == "vanilla":
            self.loss = nn.BCEWithLogitsLoss(reduction=reduction)
        else:
            raise NotImplementedError(f'GANLoss {mode} not recognized, we support lsgan and vanilla.')

    def forward(self, predict, target):
        loss = 0
        if type(predict) == list: # multi-scale discriminator
            for predict_i in predict:
                target_square = torch.ones_like(predict_i) * target
                loss += self.loss(predict_i, target_square)
        else:
            target_square = torch.ones_like(predict) * target
            loss = self.loss(predict, target_square)
        return loss

####################################################################
# ------------------------- D_Loss -------------------------------
####################################################################

class SAATDLoss(nn.Module):
    def __init__(self, opts, dis_non_makeup, dis_makeup):
        super(SAATDLoss, self).__init__()
        self.opts = opts
        self.dis_non_makeup = dis_non_makeup
        self.dis_makeup = dis_makeup

        self.dis_loss = GANLoss(opts.gan_mode) # Please define or import GANLoss() before using
        self.false = torch.Tensor([False]).to(opts.device)  # PyTorch Tensor dtype is bool
        self.true = torch.Tensor([True]).to(opts.device)    # PyTorch Tensor dtype is bool

    def forward(self, non_makeup, makeup, z_transfer, z_removal):
        non_makeup_real = self.dis_non_makeup(non_makeup)
        non_makeup_fake = self.dis_non_makeup(z_removal)
        makeup_real = self.dis_makeup(makeup)
        makeup_fake = self.dis_makeup(z_transfer)
        loss_D_non_makeup = self.dis_loss(non_makeup_fake, self.false) + self.dis_loss(non_makeup_real, self.true)
        loss_D_makeup = self.dis_loss(makeup_fake, self.false) + self.dis_loss(makeup_real, self.true)
        loss_D = (loss_D_makeup + loss_D_non_makeup) * 0.5
        return loss_D




class SAATGLoss(nn.Module):
    def __init__(self, opts, generator, dis_non_makeup, dis_makeup):
        super(SAATGLoss, self).__init__()
        self.opts = opts

        self.gen = generator
        self.dis_non_makeup = dis_non_makeup
        self.dis_makeup = dis_makeup

        self.adv_loss = GANLoss(opts.gan_mode)
        self.l1_loss = nn.L1Loss()
        self.GPL = GPLoss()
        self.CPL = CPLoss(rgb=True, yuv=True, yuvgrad=True)

        self.CP_weight = opts.CP_weight
        self.GP_weight = opts.GP_weight
        self.rec_weight = opts.rec_weight
        self.cycle_weight = opts.cycle_weight
        self.semantic_weight = opts.semantic_weight
        self.adv_weight = opts.adv_weight

        self.false = torch.Tensor([False]).to(opts.device)  # PyTorch Tensor dtype is bool
        self.true = torch.Tensor([True]).to(opts.device)    # PyTorch Tensor dtype is bool
        self.softmax = nn.Softmax(dim=2)

    def forward(self, non_makeup, makeup, non_makeup_parse, makeup_parse):
        def nearest_64(input):
            return F.interpolate(input, size=(256 // 4, 256 // 4), mode='nearest')

        z_transfer, z_removal, z_rec_non_makeup, z_rec_makeup, z_cycle_non_makeup, z_cycle_makeup, mapX, mapY = self.gen(
            non_makeup, makeup, non_makeup_parse, makeup_parse)

        # Ladv for generator
        loss_G_GAN_non_makeup = self.adv_loss(self.dis_non_makeup(z_removal), self.true)
        loss_G_GAN_makeup = self.adv_loss(self.dis_makeup(z_transfer), self.true)
        loss_G_GAN = (loss_G_GAN_non_makeup + loss_G_GAN_makeup) * 0.5 * self.adv_weight

        # rec loss
        loss_G_rec_non_makeup = self.l1_loss(non_makeup, z_rec_non_makeup)
        loss_G_rec_makeup = self.l1_loss(makeup, z_rec_makeup)
        loss_G_rec = (loss_G_rec_non_makeup + loss_G_rec_makeup) * 0.5 * self.rec_weight

        # cycle loss
        loss_G_cycle_non_makeup = self.l1_loss(non_makeup, z_cycle_non_makeup)
        loss_G_cycle_makeup = self.l1_loss(makeup, z_cycle_makeup)
        loss_G_cycle = (loss_G_cycle_non_makeup + loss_G_cycle_makeup) * 0.5 * self.cycle_weight

        # semantic loss
        non_makeup_parse_down = nearest_64(non_makeup_parse)
        n, c, h, w = non_makeup_parse_down.shape
        non_makeup_parse_down_warp = torch.bmm(non_makeup_parse_down.reshape(n, c, h * w), mapY)  # n*HW*1
        non_makeup_parse_down_warp = non_makeup_parse_down_warp.reshape(n, c, h, w)

        makeup_parse_down = nearest_64(makeup_parse)
        n, c, h, w = makeup_parse_down.shape
        makeup_parse_down_warp = torch.bmm(makeup_parse_down.reshape(n, c, h * w), mapX)  # n*HW*1
        makeup_parse_down_warp = makeup_parse_down_warp.reshape(n, c, h, w)

        loss_G_semantic_non_makeup = self.l1_loss(non_makeup_parse_down, makeup_parse_down_warp)
        loss_G_semantic_makeup = self.l1_loss(makeup_parse_down, non_makeup_parse_down_warp)
        loss_G_semantic = (loss_G_semantic_makeup + loss_G_semantic_non_makeup) * 0.5 * self.semantic_weight

        # makeup loss
        # loss_G_CP = self.CPL(z_transfer, makeup) + self.CPL(z_removal, non_makeup)
        # loss_G_GP = self.GPL(z_transfer, non_makeup) + self.GPL(z_removal, makeup)

        # print("Loss_G_CP: ", loss_G_CP)
        # print("Loss_G_GP: ", loss_G_GP)
        # loss_G_SPL = loss_G_CP * self.CP_weight + loss_G_GP * self.GP_weight

        # print("Loss G_GAN: ", loss_G_GAN)
        # print("Loss G_rec: ", loss_G_rec)
        # print("Loss G_cycle: ", loss_G_cycle)
        # print("Loss G_semantic: ", loss_G_semantic)
        # print("Loss G_SPL: ", loss_G_SPL)
        loss_G = loss_G_GAN + loss_G_rec + loss_G_cycle + loss_G_semantic #+ loss_G_SPL

        return loss_G, z_transfer, z_removal, z_rec_non_makeup, z_rec_makeup, z_cycle_non_makeup, z_cycle_makeup, mapX, mapY
