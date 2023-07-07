import networks
from networks import init_net
import torch
import torch.nn as nn
import torch.nn.functional as F


class SSAT_G(nn.Module):
    def __init__(self, opts):
        super(SSAT_G, self).__init__()
        self.opts = opts

        # parameters
        self.lr = opts.lr
        self.batch_size = opts.batch_size

        self.gpu = opts.device
        self.input_dim = opts.input_dim
        self.output_dim = opts.output_dim
        self.semantic_dim = opts.semantic_dim

        # encoders
        self.enc_content = init_net(networks.E_content(opts.input_dim), self.gpu, init_type='normal', gain=0.02)
        self.enc_makeup = init_net(networks.E_makeup(opts.input_dim), self.gpu, init_type='normal', gain=0.02)
        self.enc_semantic = init_net(networks.E_semantic(opts.semantic_dim), self.gpu, init_type='normal', gain=0.02)
        self.transformer = init_net(networks.Transformer(), self.gpu, init_type='normal', gain=0.02)
        # generator
        self.gen = init_net(networks.Decoder(opts.output_dim), self.gpu, init_type='normal', gain=0.02)


    def forward(self, non_makeup, makeup, non_makeup_parse, makeup_parse):
        # first transfer and removal
        z_non_makeup_c = self.enc_content(non_makeup)
        z_non_makeup_s = self.enc_semantic(non_makeup_parse)
        z_non_makeup_a = self.enc_makeup(non_makeup)

        z_makeup_c = self.enc_content(makeup)
        z_makeup_s = self.enc_semantic(makeup_parse)
        z_makeup_a = self.enc_makeup(makeup)
        # warp makeup style
        mapX, mapY, z_non_makeup_a_warp, z_makeup_a_warp = self.transformer(z_non_makeup_c,
                                                                            z_makeup_c,
                                                                            z_non_makeup_s,
                                                                            z_makeup_s,
                                                                            z_non_makeup_a,
                                                                            z_makeup_a)
        # makeup transfer and removal
        z_transfer = self.gen(z_non_makeup_c, z_makeup_a_warp)
        z_removal = self.gen(z_makeup_c, z_non_makeup_a_warp)

        # rec
        z_rec_non_makeup = self.gen(z_non_makeup_c, z_non_makeup_a)
        z_rec_makeup = self.gen(z_makeup_c, z_makeup_a)

        # second transfer and removal
        z_transfer_c = self.enc_content(z_transfer)
        # z_non_makeup_s = enc_semantic(non_makeup_parse
        z_transfer_a = self.enc_makeup(z_transfer)

        z_removal_c = self.enc_content(z_removal)
        # z_makeup_s = enc_semantic(makeup_parse)
        z_removal_a = self.enc_makeup(z_removal)

        # warp makeup style
        mapX2, mapY2, z_transfer_a_warp, z_removal_a_warp = self.transformer(z_transfer_c,
                                                                                z_removal_c,
                                                                                z_non_makeup_s,
                                                                                z_makeup_s,
                                                                                z_transfer_a,
                                                                                z_removal_a)
        # makeup transfer and removal
        z_cycle_non_makeup = self.gen(z_transfer_c, z_removal_a_warp)
        z_cycle_makeup = self.gen(z_removal_c, z_transfer_a_warp)

        return z_transfer, z_removal, z_rec_non_makeup, z_rec_makeup, z_cycle_non_makeup, z_cycle_makeup, mapX, mapY 

    def resume(self, model_dir, train=True):
        checkpoint = torch.load(model_dir)
        # weight
        self.enc_content.load_state_dict(checkpoint['enc_c'])
        self.enc_makeup.load_state_dict(checkpoint['enc_a'])
        self.enc_semantic.load_state_dict(checkpoint['enc_s'])
        self.transformer.load_state_dict(checkpoint['enc_trans'])
        self.gen.load_state_dict(checkpoint['gen'])
        return checkpoint['ep'], checkpoint['total_it']


    def normalize_image(self, x):
        return x[:, 0:3, :, :]

    def test_pair(self, data):
        self.non_makeup = data['non_makeup'].to(self.gpu).detach()
        self.makeup = data['makeup'].to(self.gpu).detach()
        self.non_makeup_parse = data['non_makeup_parse'].to(self.gpu).detach()
        self.makeup_parse = data['makeup_parse'].to(self.gpu).detach()
        with torch.no_grad():
            # first transfer and removal
            self.z_non_makeup_c = self.enc_content(self.non_makeup)
            self.z_non_makeup_s = self.enc_semantic(self.non_makeup_parse)
            self.z_non_makeup_a = self.enc_makeup(self.non_makeup)

            self.z_makeup_c = self.enc_content(self.makeup)
            self.z_makeup_s = self.enc_semantic(self.makeup_parse)
            self.z_makeup_a = self.enc_makeup(self.makeup)
            # warp makeup style
            self.mapX, self.mapY, self.z_non_makeup_a_warp, self.z_makeup_a_warp = self.transformer(self.z_non_makeup_c,
                                                                                                    self.z_makeup_c,
                                                                                                    self.z_non_makeup_s,
                                                                                                    self.z_makeup_s,
                                                                                                    self.z_non_makeup_a,
                                                                                                    self.z_makeup_a)
            # makeup transfer and removal
            self.z_transfer = self.gen(self.z_non_makeup_c, self.z_makeup_a_warp)
            self.z_removal = self.gen(self.z_makeup_c, self.z_non_makeup_a_warp)

        non_makeup_down = self.normalize_image(F.interpolate(self.non_makeup, scale_factor=0.25, mode='nearest'))
        n, c, h, w = non_makeup_down.shape
        non_makeup_down_warp = torch.bmm(non_makeup_down.view(n, c, h * w), self.mapY)  # n*HW*1
        non_makeup_down_warp = non_makeup_down_warp.view(n, c, h, w)
        non_makeup_warp = F.interpolate(non_makeup_down_warp, scale_factor=4)

        makeup_down = self.normalize_image(F.interpolate(self.makeup, scale_factor=0.25, mode='nearest'))
        n, c, h, w = makeup_down.shape
        makeup_down_warp = torch.bmm(makeup_down.view(n, c, h * w), self.mapX)  # n*HW*1
        makeup_down_warp = makeup_down_warp.view(n, c, h, w)
        makeup_warp = F.interpolate(makeup_down_warp, scale_factor=4)

        images_non_makeup = self.normalize_image(self.non_makeup).detach()
        images_makeup = self.normalize_image(self.makeup).detach()
        images_z_transfer = self.normalize_image(self.z_transfer).detach()
        row1 = torch.cat((images_non_makeup[0:1, ::],images_makeup[0:1, ::], makeup_warp[0:1, ::], images_z_transfer[0:1, ::]), 3)
        return row1

class SSAT_D_non_makeup(nn.Module):
    def __init__(self, opts):
        super(SSAT_D_non_makeup, self).__init__()
        self.dis_non_makeup = None
        if opts.dis_scale > 1:
            self.dis_non_makeup = networks.MultiScaleDis(opts.input_dim, n_scale=opts.dis_scale, norm=opts.dis_norm)
        else:
            self.dis_non_makeup = networks.Dis(opts.input_dim, norm=opts.dis_norm)

    def forward(self, x):
        return self.dis_non_makeup(x)

class SSAT_D_makeup(nn.Module):
    def __init__(self, opts):
        super(SSAT_D_makeup, self).__init__()
        self.dis_makeup = None
        if opts.dis_scale > 1:
            self.dis_makeup = networks.MultiScaleDis(opts.input_dim, n_scale=opts.dis_scale, norm=opts.dis_norm)
        else:
            self.dis_makeup = networks.Dis(opts.input_dim, norm=opts.dis_norm)

    def forward(self, x):
        return self.dis_makeup(x)

####################################################################
# -------------------------- get_model --------------------------
####################################################################
def get_generator(opts):
    """Return generator by args."""
    net = SSAT_G(opts).to(opts.device)
    networks.init_weights(net, opts.init_type, opts.init_gain)
    return net


def get_dis_non_makeup(opts):
    """Return discriminator by args."""
    net = SSAT_D_non_makeup(opts).to(opts.device)
    networks.init_weights(net, opts.init_type, opts.init_gain)
    return net


def get_dis_makeup(opts):
    """Return discriminator by args."""
    net = SSAT_D_makeup(opts).to(opts.device)
    networks.init_weights(net, opts.init_type, opts.init_gain)
    return net


class LossModule(nn.Module):
    """
    Wrap the network with loss function to return generator loss.
    Args:
        network (Module): The target network to wrap.
    """
    def __init__(self, network):
        super(LossModule, self).__init__(auto_prefix=False)
        self.network = network

    def forward(self, non_makeup, makeup, transfer, removal, non_makeup_parse, makeup_parse):
        lg,z_transfer, z_removal, z_rec_non_makeup, z_rec_makeup, z_cycle_non_makeup, z_cycle_makeup, mapX, mapY = self.network(non_makeup, makeup, transfer, removal, non_makeup_parse, makeup_parse)
        return lg


