# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ppcls.utils.save_load import load_dygraph_pretrain, load_dygraph_pretrain_from_url


MODEL_URLS = {
    "OctMobileNet_v2_100":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV2_x0_25_pretrained.pdparams",
    "OctMobileNet_v2_1125":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV2_x0_5_pretrained.pdparams",
}

__all__ = list(MODEL_URLS.keys())


class _upsampling(nn.Layer):
    def __init__(self, scales, sample_type='nearest', **kwargs):
        super().__init__(**kwargs)

        assert type(scales) is int or scales[0] == scales[1], \
            "TODO: current upsampling requires all dimensions share the same scale"

        self.scale = scales if type(scales) is int else scales[0]
        self.sample_type = sample_type

    def forward(self, x):
        return F.upsample(x, scale_factor=self.scale, mode=self.sample_type)


class Adapter(nn.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x_h, x_l=None):
        if x_l is not None:
            x_l = F.upsample(x_l, scale_factor=2, mode='nearest')
            x_h = paddle.concat([x_h, x_l], axis=1)
        return x_h

class Activation(nn.Layer):
    def __init__(self, activation='relu', **kwargs):
        # options: {'relu', 'sigmoid', 'softrelu', 'softsign', 'tanh'}
        super().__init__(**kwargs)
        self.activation = activation

    def forward(self, x_h, x_l=None):
        if self.activation == 'relu6':
            # func = (lambda x: paddle.clip(x, 0, 6)) # relu6
            func = F.relu6
        elif self.activation == 'relu':
            func = F.relu
        else:
            func = getattr(F, self.activation)
        x_h = func(x_h) if x_h is not None else None
        x_l = func(x_l) if x_l is not None else None
        return (x_h, x_l)


class SE(nn.Layer):
    def __init__(self, channels, in_channels=0, prefix=None, **kwargs):
        super().__init__(name_scope=prefix, **kwargs)
        # be compatible to conventional convolution
        in_c_h, in_c_l = in_channels if type(in_channels) is tuple else (in_channels, -1)
        assert in_c_l != 0 and in_c_h != 0, \
            "TODO: current version has to specify the `in_channels' to determine the computation graph, but got {}".format(in_channels)

        has_h = in_c_h > 0
        has_l = in_c_l > 0

        self.conv1 = nn.Conv2D(channels, kernel_size=1, padding=0, #name_scope='-conv1_')
        )
        self.relu1 = nn.ReLU()
        self.conv2_h = nn.Conv2D(in_channels=in_c_h, out_channels=in_c_h, kernel_size=1, padding=0,
                            #name_scope='-conv2-h_'
                            ) if has_h else lambda x: None
        self.sigmoid2_h = nn.Sigmoid() if has_h else lambda x: None

        self.conv2_l = nn.Conv2D(in_channels=in_c_l, out_channels=in_c_l, kernel_size=1, padding=0,
                            #name_scope='-conv2-l_'
                            ) if has_l else lambda x: None
        self.sigmoid2_l = nn.Sigmoid() if has_l else lambda x: None

    def _broadcast_mul(self, x, w):
        if x is None or w is None:
            assert x is None and w is None, "x is {} but w {}".format(x, w)
            return None
        else:
            # return F.broadcast_mul(x, w)
            return x * w

    def _concat(self, x1, x2):
        if (x1 is not None) and (x2 is not None):
            return paddle.concat([x1, x2], axis=1)
        else:
            return x1 if x2 is None else x2

    def forward(self, x_h, x_l=None):

        out_h = F.adaptive_avg_pool2d(x_h, output_size=1) if x_h is not None else None
        out_l = F.adaptive_avg_pool2d(x_l, output_size=1) if x_l is not None else None
        out = self._concat(out_h, out_l)

        out = self.relu1(self.conv1(out))

        w_h = self.sigmoid2_h(self.conv2_h(out))
        w_l = self.sigmoid2_l(self.conv2_l(out))

        x_h = self._broadcast_mul(F, x_h, w_h)
        x_l = self._broadcast_mul(F, x_l, w_l)

        return (x_h, x_l)


class BatchNorm(nn.Layer):
    def __init__(self, in_channels=0, gamma_initializer='ones', prefix=None, center=True, scale=True, **kwargs):
        super().__init__(name_scope=prefix, **kwargs)
        # be compatible to conventional convolution
        in_c_h, in_c_l = in_channels if type(in_channels) is tuple else (in_channels, -1)
        assert in_c_l != 0 and in_c_h != 0, \
            "TODO: current version has to specify the `in_channels' to determine the computation graph, but got {}".format(in_channels)

        if scale:
            if gamma_initializer == 'ones':
                init_weight = True # 默认就是 1
            else:
                init_weight = False
        else:
            init_weight = False
        self.bn_h = nn.BatchNorm(in_c_h, param_attr=init_weight, bias_attr=center, #name_scope='-h_', 
                                **kwargs) if in_c_h >= 0 else lambda x: (x)
        if scale:
            if gamma_initializer == 'ones':
                init_weight = True # 默认就是 1
            else:
                init_weight = False
        else:
            init_weight = False
        self.bn_l = nn.BatchNorm(in_c_l, param_attr=init_weight, bias_attr=center, #name_scope='-l_', 
                                **kwargs) if in_c_l >= 0 else lambda x: (x)

    def forward(self, x_h, x_l=None):
        x_h = self.bn_h(x_h) if x_h is not None else None
        x_l = self.bn_l(x_l) if x_l is not None else None
        return (x_h, x_l)


class AvgPool2D(nn.Layer):
    def __init__(self, pool_size, strides, padding=0, in_channels=0, ceil_mode=False, count_include_pad=False, **kwargs):
        super().__init__(**kwargs)
        # be compatible to conventional convolution
        in_c_h, in_c_l = in_channels if type(in_channels) is tuple else (in_channels, -1)
        assert in_c_l != 0 and in_c_h != 0, \
            "TODO: current version has to specify the `in_channels' to determine the computation graph, but got {}".format(in_channels)

        self.pool_h = nn.AvgPool2D(kernel_size=pool_size, stride=strides, padding=padding, ceil_mode=ceil_mode, exclusive=not count_include_pad) if in_c_h >= 0 else lambda x: (x)
        self.pool_l = nn.AvgPool2D(kernel_size=pool_size, stride=strides, padding=padding, ceil_mode=ceil_mode, exclusive=not count_include_pad) if in_c_l >= 0 else lambda x: (x)

    def hybrid_forward(self, x_h, x_l=None):
        x_h = self.pool_h(x_h) if x_h is not None else None
        x_l = self.pool_l(x_l) if x_l is not None else None
        return (x_h, x_l)


class MaxPool2D(nn.Layer):
    def __init__(self, pool_size, strides, padding=0, in_channels=0, ceil_mode=False, count_include_pad=False, **kwargs):
        super().__init__(**kwargs)
        # be compatible to conventional convolution
        in_c_h, in_c_l = in_channels if type(in_channels) is tuple else (in_channels, -1)
        assert in_c_l != 0 and in_c_h != 0, \
            "TODO: current version has to specify the `in_channels' to determine the computation graph, but got {}".format(in_channels)

        self.pool_h = nn.MaxPool2D(kernel_size=pool_size, stride=strides, padding=padding, ceil_mode=ceil_mode, exclusive=not count_include_pad) if in_c_h >= 0 else lambda x: (x)
        self.pool_l = nn.MaxPool2D(kernel_size=pool_size, stride=strides, padding=padding, ceil_mode=ceil_mode, exclusive=not count_include_pad) if in_c_l >= 0 else lambda x: (x)

    def hybrid_forward(self, F, x_h, x_l=None):
        x_h = self.pool_h(x_h) if x_h is not None else None
        x_l = self.pool_l(x_l) if x_l is not None else None
        return (x_h, x_l)


class Conv2D(nn.Layer):
    def __init__(self, channels, kernel_size, strides=(1, 1), use_bias=True,
                 in_channels=0, enable_path=((0, 0), (0, 0)), padding=0,
                 groups=1, sample_type='nearest', prefix=None, **kwargs):
        super().__init__(name_scope=prefix, **kwargs)
        # be compatible to conventional convolution
        (h2l, h2h), (l2l, l2h) = enable_path
        c_h, c_l = channels if type(channels) is tuple else (channels, 0)
        in_c_h, in_c_l = in_channels if type(in_channels) is tuple else (in_channels, -1)

        assert (in_c_h + in_c_l) == groups or ((in_c_h < 0 or in_c_h/groups >= 1) \
                and (in_c_l < 0 or in_c_l/groups >= 1)), \
            "Constains are not satisfied: (%d+%d)==%d, %d/%d>1, %d/%d>1" % ( \
            in_c_h, in_c_l, groups, in_c_h, groups, in_c_l, groups )
        assert in_c_l != 0 and in_c_h != 0, \
            "TODO: current version has to specify the `in_channels' to determine the computation graph"
        assert strides == 1 or strides == 2 or all((s <= 2 for s in strides)), \
            "TODO: current version only support strides({}) <= 2".format(strides)

        is_dw = False
        # computational graph will be automatic or manually defined
        self.enable_l2l = True if l2l != -1 and (in_c_l >= 0 and c_l > 0) else False
        self.enable_l2h = True if l2h != -1 and (in_c_l >= 0 and c_h > 0) else False
        self.enable_h2l = True if h2l != -1 and (in_c_h >= 0 and c_l > 0) else False
        self.enable_h2h = True if h2h != -1 and (in_c_h >= 0 and c_h > 0) else False
        if groups == (in_c_h + in_c_l): # depthwise convolution
            assert c_l == in_c_l and c_h == in_c_h
            self.enable_l2h, self.enable_h2l = False, False
            is_dw = True
        use_bias_l2l, use_bias_h2l = (False, use_bias) if self.enable_h2l else (use_bias, False)
        use_bias_l2h, use_bias_h2h = (False, use_bias) if self.enable_h2h else (use_bias, False)

        # deal with stride with resizing (here, implemented by pooling)
        s = (strides, strides) if type(strides) is int else strides
        do_stride2 = s[0] > 1 or s[1] > 1

        self.conv_l2l = None if not self.enable_l2l else nn.Conv2D(
                        out_channels=c_l, kernel_size=kernel_size, stride=1,
                        padding=padding, groups=groups if not is_dw else in_c_l,
                        bias_attr=use_bias_l2l, in_channels=in_c_l,
                        # name_scope='-l2l_', 
                        **kwargs)

        self.conv_l2h = None if not self.enable_l2h else nn.Conv2D(
                        out_channels=c_h, kernel_size=kernel_size, stride=1,
                        padding=padding, groups=groups,
                        bias_attr=use_bias_l2h, in_channels=in_c_l,
                        # name_scope='-l2h_', 
                        **kwargs)

        self.conv_h2l = None if not self.enable_h2l else nn.Conv2D(
                        out_channels=c_l, kernel_size=kernel_size, stride=1,
                        padding=padding, groups=groups,
                        bias_attr=use_bias_h2l, in_channels=in_c_h,
                        # name_scope='-h2l_', 
                        **kwargs)

        self.conv_h2h = None if not self.enable_h2h else nn.Conv2D(
                        out_channels=c_h, kernel_size=kernel_size, stride=1,
                        padding=padding, groups=groups if not is_dw else in_c_h,
                        bias_attr=use_bias_h2h, in_channels=in_c_h,
                        # name_scope='-h2h_',
                        **kwargs)

        self.l2l_down = (lambda x: (x)) if not self.enable_l2l or not do_stride2 else \
                        nn.AvgPool2D(kernel_size=strides, stride=strides, \
                                        ceil_mode=True, exclusive=True)

        self.l2h_up = (lambda x: (x)) if not self.enable_l2h or do_stride2 else \
                        _upsampling(scales=(2, 2), sample_type=sample_type)

        self.h2h_down = (lambda x: (x)) if not self.enable_h2h or not do_stride2 else \
                        nn.AvgPool2D(kernel_size=strides, stride=strides, \
                                        ceil_mode=True, exclusive=True)

        self.h2l_down = (lambda x: (x)) if not self.enable_h2l else \
                        nn.AvgPool2D(kernel_size=(2*s[0], 2*s[1]), \
                                        stride=(2*s[0], 2*s[1]), \
                                        ceil_mode=True, exclusive=True)

    def _sum(self, x1, x2):
        if (x1 is not None) and (x2 is not None):
            return x1 + x2
        else:
            return x1 if x2 is None else x2

    def forward(self, x_high, x_low=None):

        x_h2h = self.conv_h2h(self.h2h_down(x_high)) if self.enable_h2h else None
        x_h2l = self.conv_h2l(self.h2l_down(x_high)) if self.enable_h2l else None

        x_l2h = self.l2h_up(self.conv_l2h(x_low)) if self.enable_l2h else None
        x_l2l = self.conv_l2l(self.l2l_down(x_low)) if self.enable_l2l else None

        x_h = self._sum(x_l2h, x_h2h)
        x_l = self._sum(x_l2l, x_h2l)
        return (x_h, x_l)


class RELU6(nn.Layer):
    """Relu6 used in MobileNetV2."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x):
        # return F.clip(x, 0, 6, name="relu6")
        return F.relu6(x)

# use RELU6 if gluon.nn not support 'relu6'
_op_act = Activation

class _BottleneckV1(nn.Layer):
    """ResNetV1 BottleneckV1
    """
    # pylint: disable=unused-argument
    def __init__(self, in_planes, mid_planes, out_planes, strides=1,
                 norm_kwargs=None, last_gamma=False, name_prefix=None,
                 **kwargs):
        super().__init__(name_scope=name_prefix)

        self.use_shortcut = strides == 1 and in_planes == out_planes

        num_group = sum((c if c > 0 else 0 for c in mid_planes))

        # extract information
        self.conv1 = Conv2D(channels=mid_planes, in_channels=in_planes,
                                kernel_size=1, use_bias=False, prefix='conv1')
        self.bn1 = BatchNorm(in_channels=mid_planes, prefix='bn1',
                                **({} if norm_kwargs is None else norm_kwargs))
        self.relu1 = _op_act('relu6')
        # capture spatial relations
        self.conv2 = Conv2D(channels=mid_planes, in_channels=mid_planes,
                                kernel_size=3, padding=1, groups=num_group,
                                strides=strides, use_bias=False, prefix='conv2')
        self.bn2 = BatchNorm(in_channels=mid_planes, prefix='bn2',
                                **({} if norm_kwargs is None else norm_kwargs))
        self.relu2 = _op_act('relu6')
        # embeding back to information highway
        self.conv3 = Conv2D(channels=out_planes, in_channels=mid_planes,
                                kernel_size=1, use_bias=False, prefix='conv3')
        self.bn3 = BatchNorm(in_channels=out_planes, prefix='bn3',
                                gamma_initializer='zeros' if (last_gamma and \
                                self.use_shortcut) else 'ones',
                                **({} if norm_kwargs is None else norm_kwargs))

    def _sum(self, x1, x2):
        if (x1 is not None) and (x2 is not None):
            return x1 + x2
        else:
            return x1 if x2 is None else x2

    def forward(self, x1, x2=None):
        x = (x1, x2)
        shortcut = x

        out = self.relu1(*self.bn1(*self.conv1(*x)))
        out = self.relu2(*self.bn2(*self.conv2(*out)))
        out = self.bn3(*self.conv3(*out))

        if self.use_shortcut:
            out = (self._sum(out[0], shortcut[0]), self._sum(out[1], shortcut[1]))

        return out


class OctMobileNetV2(nn.Layer):
    r"""MobileNet model from the
    `"MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
    <https://arxiv.org/abs/1704.04861>`_ paper.

    Parameters
    ----------
    multiplier : float, default 1.0
        The width multiplier for controlling the model size. Only multipliers that are no
        less than 0.25 are supported. The actual number of channels is equal to the original
        channel size multiplied by this multiplier.
    classes : int, default 1000
        Number of classes for the output layer.
    """

    def __init__(self, multiplier=1.0, classes=1000, ratio=0.,
                 norm_kwargs=None, final_drop=0., last_gamma=False,
                 name_prefix=None, **kwargs):
        super().__init__(name_scope=name_prefix)
        # reference:
        # - Howard, Andrew G., et al.
        #   "Mobilenets: Efficient convolutional neural networks for mobile vision applications."
        #   arXiv preprint arXiv:1704.04861 (2017).
        in_channels = [int(multiplier * x) for x in
                        [32] + [16] + [24] * 2 + [32] * 3 + [64] * 4 + [96] * 3 + [160] * 3]
        mid_channels = [int(t * x) for t, x in zip([1] + [6] * 16, in_channels)]
        out_channels = [int(multiplier * x) for t, x in zip([1] + [6] * 16,
                        [16] + [24] * 2 + [32] * 3 + [64] * 4 + [96] * 3 + [160] * 3 + [320])]
        strides = [1, 2] * 2 + [1, 1, 2] + [1] * 6 + [2] + [1] * 3
        in_ratios = [0.] + [ratio] * 13 + [0.] * 3
        ratios = [ratio] * 13 + [0.] * 4
        last_channels = int(1280 * multiplier) if multiplier > 1.0 else 1280

        self.conv1 = nn.Sequential()
        self.conv1.add_sublayer('conv', nn.Conv2D(in_channels=3, out_channels=int(32 * multiplier),
                        kernel_size=3, padding=1, stride=2, bias_attr=False,
                        # name_scope='conv1'))
                        ))
        self.conv1.add_sublayer('bn', nn.BatchNorm(#name_scope='bn1_',
                                                    int(32 * multiplier),
                        **({} if norm_kwargs is None else norm_kwargs)))
        self.conv1.add_sublayer('ac', RELU6())
        # ------------------------------------------------------------------
        stage_index, i = 1, 0
        for k, (in_c, mid_c, out_c, s, ir, r) in enumerate(zip(in_channels, mid_channels, out_channels, strides, in_ratios, ratios)):
            stage_index += 1 if s > 1 else 0
            i = 0 if s > 1 else (i + 1)
            name = 'L%d_B%d' % (stage_index, i)
            # -------------------------------------
            in_c = (in_c, -1)
            mid_c = self._get_channles(mid_c, r)
            out_c = (out_c, -1)
            # -------------------------------------
            setattr(self, name, _BottleneckV1(in_c, mid_c, out_c,
                                            strides=s,
                                            norm_kwargs=None,
                                            last_gamma=last_gamma,
                                            name_prefix="%s_" % name))
        # ------------------------------------------------------------------
        self.tail = nn.Sequential()
        self.tail.add_sublayer('conv', nn.Conv2D(out_channels=last_channels, in_channels=out_channels[-1],
                                kernel_size=1, bias_attr=False, 
                                #name_scope='tail-conv_'))
                                ))
        self.tail.add_sublayer('bn', nn.BatchNorm(#name_scope='tail-bn_',
                                                   last_channels,
                                **({} if norm_kwargs is None else norm_kwargs)))
        self.tail.add_sublayer('ac', RELU6())
        # ------------------------------------------------------------------
        self.avgpool = nn.AdaptiveAvgPool2D(1)
        self.drop = nn.Dropout(final_drop) if final_drop > 0. else lambda x: (x)
        self.classifer = nn.Conv2D(in_channels=last_channels, out_channels=classes,
                                    kernel_size=1, #name_scope='classifier_')
        )
        self.flat = nn.Flatten()

    def _get_channles(self, width, ratio):
        width = (width - int(ratio * width), int(ratio * width))
        width = tuple(c if c != 0 else -1 for c in width)
        return width

    def _concat(self, x1, x2):
        if (x1 is not None) and (x2 is not None):
            return paddle.concat(x1, x2, axis=1)
        else:
            return x1 if x2 is None else x2

    def forward(self, x):

        x = self.conv1(x)

        x = (x, None)
        for iy in range(1, 10):
            # assume the max number of blocks is 50 per stage
            for ib in range(0, 50):
                name = 'L%d_B%d' % (iy, ib)
                if hasattr(self, name):
                    x = getattr(self, name)(*x)

        x = self.tail(x[0])
        x = self.avgpool(x)
        x = self.drop(x)
        x = self.classifer(x)
        x = self.flat(x)
        return x

def _load_pretrained(pretrained, model, model_url, use_ssld=False):
    if pretrained is False:
        pass
    elif pretrained is True:
        load_dygraph_pretrain_from_url(model, model_url, use_ssld=use_ssld)
    elif isinstance(pretrained, str):
        load_dygraph_pretrain(model, pretrained)
    else:
        raise RuntimeError(
            "pretrained type is not available. Please use `string` or `boolean` type."
        )

def OctMobileNet_v2_100(pretrained=False, use_ssld=False, **kwargs):
    model = OctMobileNetV2(multiplier=1.0, ratio=0.5,
                       name_prefix='M100_', norm_kwargs=None, **kwargs)
    _load_pretrained(
    pretrained, model, MODEL_URLS["OctMobileNet_v2_100"], use_ssld=use_ssld)
    return model
    
def OctMobileNet_v2_1125(pretrained=False, use_ssld=False, **kwargs):
    model = OctMobileNetV2(multiplier=1.125, ratio=0.5,
                       name_prefix='M1125_', norm_kwargs=None, **kwargs)
    _load_pretrained(
    pretrained, model, MODEL_URLS["OctMobileNet_v2_1125"], use_ssld=use_ssld)
    return model

