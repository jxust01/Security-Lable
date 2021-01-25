import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from utils import bbox_wh_iou
from splat import SplAtConv2d


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, bn=True, activation=True):
        super(Conv, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.activation = activation
        self.bn = bn
        self.layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=stride, padding=0, bias=not bn)
        if bn:
            self.bn = nn.BatchNorm2d(out_channels)
        if activation:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        b, c, h, w = x.size()
        oh = ((h - 1) // self.stride) * self.stride + self.kernel_size
        ow = ((w - 1) // self.stride) * self.stride + self.kernel_size
        oh = max(0, oh - h)
        ow = max(0, ow - w)
        padding = (ow // 2, (ow + 1) // 2, oh // 2, (oh + 1) // 2)
        x = self.layer(F.pad(x, padding))
        if self.bn:
            x = self.bn(x)
        if self.activation:
            x = self.relu(x)

        return x


class Bottleneck(nn.Module):
    def __init__(self, in_places, places, stride=1, expansion=2):
        super(Bottleneck, self).__init__()
        self.downsample = stride > 1 or in_places != places * expansion
        self.b0 = Conv(in_channels=in_places, out_channels=places, kernel_size=1, stride=1)
        self.b1 = SplAtConv2d(in_channels=places, channels=places, kernel_size=3, stride=(stride, stride))
        self.b2 = Conv(in_channels=places, out_channels=places * expansion, kernel_size=1, stride=1,
                       activation=False)

        if self.downsample:
            self.downsample = Conv(in_channels=in_places, out_channels=places * expansion, kernel_size=3, \
                                   stride=stride, activation=False)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.b0(x)
        out = self.b1(out)
        out = self.b2(out)

        if self.downsample:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class Upsample(nn.Module):
    def __init__(self, scale_factor, mode="nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x


class YOLOLayer(nn.Module):
    def __init__(self, anchors, places):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.grid_size = [0, 0]
        self.conv = Conv(in_channels=places, out_channels=self.num_anchors*5, \
                kernel_size=3, stride=1, bn=False, activation=False)

        self.anchor_w = self.anchors[:, 0:1].view((1, 1, 1, self.num_anchors))
        self.anchor_h = self.anchors[:, 1:2].view((1, 1, 1, self.num_anchors))

    def compute_grid_offsets(self, grid_size, cuda=True):
        self.grid_size = grid_size
        gh, gw = self.grid_size
        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.grid_x = torch.arange(gw).repeat(gh, 1).view([1, gh, gw, 1]).type(FloatTensor)
        self.grid_y = torch.arange(gh).repeat(gw, 1).t().view([1, gh, gw, 1]).type(FloatTensor)

    def forward(self, x):
        x = self.conv(x)

        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        num_samples = x.size(0)
        grid_size = [x.size(2), x.size(3)]

        prediction = (
            x.view(num_samples, self.num_anchors, 5, grid_size[0], grid_size[1])
                .permute(0, 3, 4, 1, 2)
                .contiguous()
        )

        x = torch.sigmoid(prediction[..., 0])  # Center x
        y = torch.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        pred_conf = torch.sigmoid(prediction[..., 4:5])  # Conf

        if (grid_size[0] != self.grid_size[0]) or (grid_size[1] != self.grid_size[1]):
            self.compute_grid_offsets(grid_size, cuda=x.is_cuda)

        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = (x + self.grid_x) / self.grid_size[1]
        pred_boxes[..., 1] = (y + self.grid_y) / self.grid_size[0]
        pred_boxes[..., 2] = (torch.exp(w) * self.anchor_w)
        pred_boxes[..., 3] = (torch.exp(h) * self.anchor_h)

        output = FloatTensor(prediction.shape)
        output[..., 0] = x
        output[..., 1] = y
        output[..., 2] = w
        output[..., 3] = h
        output[..., 4:5] = pred_conf

        boxes = torch.cat((pred_boxes, pred_conf), -1)
        return output, boxes.view(num_samples, -1, 5)


class loss_layer(nn.Module):
    def __init__(self, anchors):
        super(loss_layer, self).__init__()
        self.anchors = anchors
        self.ignore_thres = 0.5
        self.obj_scale = 1
        self.noobj_scale = 100
        self.metrics = {}
        print('loss params\nignore_thres:\t\t\t{}\nobj_scale:\t\t\t\t{}\nnoobj_scale:\t\t\t{}\n'.format(self.ignore_thres, self.obj_scale, self.noobj_scale))

    def cross_entropy(self, x, y, alpha=0.25, gamma=2, eps=1e-16):
        cross_entropy = -alpha * y * torch.pow(1 - x, gamma) * torch.log(x + eps) \
                        - (1 - alpha) * (1 - y) * torch.pow(x, gamma) * torch.log(1 - x + eps)
        return cross_entropy

    def forward(self, xs, target):
        FloatTensor = torch.cuda.FloatTensor if xs[0].is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if xs[0].is_cuda else torch.LongTensor

        if target.size(0):
            gwh = target[:, 3:]  # size: N x 2
            anchors = self.anchors.view(-1, 2)  # size: 12 x 2
            anchors_ious = torch.stack([bbox_wh_iou(_an, gwh) for _an in anchors])  # size: 12 x N
            _, best_n = anchors_ious.max(0)  # size: N  value: 0-11

            ll = [l.size(0) for l in self.anchors]  # anchors size: 4 x 3 x 2   ll = [3, 3, 3, 3]
            anchors_ious = torch.split(anchors_ious, ll, 0)  # size: [3 x N, 3 x N, 3 x N, 3 x N]

            best_i = LongTensor(gwh.size(0), 2).fill_(0)  # size: N x 2
            for i, l in enumerate(ll):
                index = (best_n < l) & (best_n >= 0)
                best_i[index, 0] = i  # [:,0] -> feature index
                best_i[index, 1] = best_n[index]  # [:,1] -> anchors index
                best_n -= l

        loss_xy, loss_wh, loss_conf = 0, 0, 0
        for i, xs_ in enumerate(xs):
            x = xs_[..., 0]
            y = xs_[..., 1]
            w = xs_[..., 2]
            h = xs_[..., 3]

            pred_boxes = xs_[..., :4]
            pred_conf = xs_[..., 4]  # Conf

            nB = pred_boxes.size(0)
            nGh = pred_boxes.size(1)
            nGw = pred_boxes.size(2)
            nA = pred_boxes.size(3)

            if target.size(0):
                obj_mask = FloatTensor(nB, nGh, nGw, nA).fill_(0)
                noobj_mask = FloatTensor(nB, nGh, nGw, nA).fill_(1)

                tx = FloatTensor(nB, nGh, nGw, nA).fill_(0)
                ty = FloatTensor(nB, nGh, nGw, nA).fill_(0)
                tw = FloatTensor(nB, nGh, nGw, nA).fill_(0)
                th = FloatTensor(nB, nGh, nGw, nA).fill_(0)

                index = best_i[:, 0] == i  # [:,0] -> feature index
                anchors_iou = anchors_ious[i][:, index]  # size: 12 x N
                best_n = best_i[index, 1]  # [:,1] -> anchors index
                anchor = self.anchors[i]

                target_boxes = target[index, 1:]  # size: N x 4
                gxy = target_boxes[:, :2]  # size: N x 2
                gwh = target_boxes[:, 2:]  # size: N x 2
                b = target[index, 0].long()  # size: N

                gx, gy = gxy.t()  # size: N
                gx, gy = gx * nGw, gy * nGh
                gw, gh = gwh.t()
                gi, gj = gx.long(), gy.long()
                obj_mask[b, gj, gi, best_n] = 1
                noobj_mask[b, gj, gi, best_n] = 0

                for i, iou in enumerate(anchors_iou.t()):
                    noobj_mask[b[i], gj[i], gi[i], iou > self.ignore_thres] = 0

                tx[b, gj, gi, best_n] = gx - gx.floor()
                ty[b, gj, gi, best_n] = gy - gy.floor()
                tw[b, gj, gi, best_n] = torch.log(gw / anchor[best_n][:, 0] + 1e-16)
                th[b, gj, gi, best_n] = torch.log(gh / anchor[best_n][:, 1] + 1e-16)

                tconf = obj_mask
                sum_axis = (1, 2, 3)
                obj_sum = obj_mask.sum(sum_axis) + 1e-6
                noobj_sum = noobj_mask.sum(sum_axis) + 1e-6

                dxy = (x - tx) ** 2 + (y - ty) ** 2
                dwh = (w - tw) ** 2 + (h - th) ** 2

                loss_xy += ((dxy * obj_mask).sum(sum_axis) / obj_sum).mean()
                loss_wh += ((dwh * obj_mask).sum(sum_axis) / obj_sum).mean()
                conf_bce = self.cross_entropy(pred_conf, tconf)
                obj_bce = ((conf_bce * obj_mask).sum(sum_axis) / obj_sum).mean()
                noobj_bce = ((conf_bce * noobj_mask).sum(sum_axis) / noobj_sum).mean()

                loss_conf += self.obj_scale * obj_bce + self.noobj_scale * noobj_bce

                total_loss = loss_xy + loss_wh + loss_conf
            else:
                tconf = FloatTensor(nB, nGh, nGw, nA).fill_(0)
                conf_bce = self.cross_entropy(pred_conf, tconf)
                loss_conf += self.noobj_scale * (conf_bce.mean())
                total_loss = loss_xy + loss_wh + loss_conf

        loss_t = total_loss.detach().cpu().item()
        if target.size(0):
            loss_xy = loss_xy.detach().cpu().item()
            loss_wh = loss_wh.detach().cpu().item()
            loss_conf = loss_conf.detach().cpu().item()

        self.metrics = {
            "loss": loss_t,
            "xy": loss_xy,
            "wh": loss_wh,
            "conf": loss_conf,
        }

        return total_loss


# ResNet50: [3, 4, 6, 3]
# ResNet101: [3, 4, 23, 3]
# ResNet152: [3, 8, 36, 3]
class ResNet(nn.Module):
    def __init__(self, anchors, blocks=[3, 4, 5, 5], expansion=2, Istrain=True):
        super(ResNet, self).__init__()
        self.expansion = expansion
        assert anchors.size(0) == 4
        self.anchors = anchors

        self.conv1 = Conv(in_channels=3, out_channels=16, kernel_size=7, stride=2)

        self.layer1 = self.make_layer(in_places=16, places=16, block=blocks[0], stride=2)
        self.layer2 = self.make_layer(in_places=32, places=32, block=blocks[1], stride=2)
        self.layer3 = self.make_layer(in_places=64, places=64, block=blocks[2], stride=2)
        self.layer4 = self.make_layer(in_places=128, places=128, block=blocks[3], stride=2)

        self.output4 = Bottleneck(in_places=256, places=128, stride=1, expansion=1)
        self.yolo4 = YOLOLayer(self.anchors[3], 128)

        self.output3 = Bottleneck(in_places=256, places=64, stride=1, expansion=1)
        self.yolo3 = YOLOLayer(self.anchors[2], 64)

        self.output2 = Bottleneck(in_places=128, places=32, stride=1, expansion=1)
        self.yolo2 = YOLOLayer(self.anchors[1], 32)

        self.output1 = Bottleneck(in_places=64, places=16, stride=1, expansion=1)
        self.yolo1 = YOLOLayer(self.anchors[0], 16)

        self.upsample = Upsample(2)
        if Istrain:
            self.loss_layers = loss_layer(self.anchors)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layer(self, in_places, places, block, stride):
        layers = []
        layers.append(Bottleneck(in_places, places, stride, self.expansion))
        for i in range(1, block):
            layers.append(Bottleneck(places * self.expansion, places, 1, self.expansion))

        return nn.Sequential(*layers)

    def forward(self, x):
        x0 = self.conv1(x)

        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x = self.output4(x4)
        yolo4, boxes4 = self.yolo4(x)

        x = self.upsample(x)
        x = torch.cat([x, x3], 1)
        x = self.output3(x)
        yolo3, boxes3 = self.yolo3(x)

        x = self.upsample(x)
        x = torch.cat([x, x2], 1)
        x = self.output2(x)
        yolo2, boxes2 = self.yolo2(x)

        x = self.upsample(x)
        x = torch.cat([x, x1], 1)
        x = self.output1(x)
        yolo1, boxes1 = self.yolo1(x)

        yolo_map = [yolo1, yolo2, yolo3, yolo4]
        yolo_outputs = torch.cat([boxes1, boxes2, boxes3, boxes4], 1)

        return yolo_map, yolo_outputs

    def loss(self, f, y):
        total_loss = self.loss_layers(f, y)
        metrics = self.loss_layers.metrics
        return total_loss, metrics

    def regularization_loss(self, weight_decay=5e-5):
        reg_loss = 0
        for name, param in self.named_parameters():
            if 'weight' in name:
                reg_loss += torch.norm(param, p=2)

        return reg_loss * weight_decay
    
    def save_weights(self, save_file):
        state = {'net': self.state_dict(), 'anchors': self.anchors}
        torch.save(state, save_file)
        
    def load_pretrained_weights(self, pretrianed_weights, by_name=True):
        pretrianed_dict = torch.load(pretrianed_weights)['net']
        model_dict = self.state_dict()
        if by_name:
            pretrianed_dict_update = {}
            for k, v in pretrianed_dict.items():
                if k in model_dict:
                    vv = model_dict[k]
                    if v.size() == vv.size():
                        pretrianed_dict_update[k] = v
            model_dict.update(pretrianed_dict_update)
        else:
            model_dict.update(pretrianed_dict)
        self.load_state_dict(model_dict)


if __name__ == '__main__':
    anchors = [[(np.random.rand(), np.random.rand()) for __ in range(3)] for _ in range(4)]
    model = ResNet(anchors=torch.from_numpy(np.array(anchors)))
    print(model)
    model.eval()

    input = torch.randn(1, 3, 224, 224)
    map, outputs = model(input)
    print([o.size() for o in map])
    print()
    torch.save(model.state_dict(), 'model.pth')
