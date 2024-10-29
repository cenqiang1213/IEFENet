import torch
import torch.nn as nn
from models.common import Conv

############################## =====1.MUIE===== ##############################
#clahe
def clahe(image,clipLimit=2.0, tileGridSize=(8, 8)):
    B,G,R = cv2.split(image)
    clahe = cv2.createCLAHE(clipLimit, tileGridSize)
    clahe_B = clahe.apply(B)
    clahe_G = clahe.apply(G)
    clahe_R = clahe.apply(R)
    result = cv2.merge((clahe_B,clahe_G,clahe_R))
    return result

#gamma
def adjust_gamma(image, gamma):
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(image, gamma_table)


# defog
def zmMinFilterGray(src, r=7):
    return cv2.erode(src, np.ones((2*r+1, 2*r+1)))

def guidedfilter(I, p, r, eps):
    '''引导滤波，直接参考网上的matlab代码'''
    height, width = I.shape
    m_I = cv2.boxFilter(I, -1, (r,r))
    m_p = cv2.boxFilter(p, -1, (r,r))
    m_Ip = cv2.boxFilter(I*p, -1, (r,r))
    cov_Ip = m_Ip-m_I*m_p
    m_II = cv2.boxFilter(I*I, -1, (r,r))
    var_I = m_II-m_I*m_I
    a = cov_Ip/(var_I+eps)
    b = m_p-a*m_I
    m_a = cv2.boxFilter(a, -1, (r,r))
    m_b = cv2.boxFilter(b, -1, (r,r))
    return m_a*I+m_b


def getV1(m, r, eps, w, maxV1):  #输入rgb图像，值范围[0,1]
    '''计算大气遮罩图像V1和光照值A, V1 = 1-t/A'''
    V1 = np.min(m,2)                                         #得到暗通道图像
    V1 = guidedfilter(V1, zmMinFilterGray(V1,7), r, eps)     #使用引导滤波优化
    bins = 2000
    ht = np.histogram(V1, bins)                              #计算大气光照A
    d = np.cumsum(ht[0])/float(V1.size)
    for lmax in range(bins-1, 0, -1):
        if d[lmax]<=0.999:
            break
    A  = np.mean(m,2)[V1>=ht[1][lmax]].max()
    V1 = np.minimum(V1*w, maxV1)                   #对值范围进行限制
    return V1,A


def defog(m, r=81, eps=0.001, w=0.95, maxV1=0.80, bGamma=False):
    m = m/255.0
    Y = np.zeros(m.shape)
    V1,A = getV1(m, r, eps, w, maxV1)               #得到遮罩图像和大气光照
    for k in range(3):
        Y[:,:,k] = (m[:,:,k]-V1)/(1-V1/A)           #颜色校正
    Y =  np.clip(Y, 0, 1)
    if bGamma:
        Y = Y**(np.log(0.5)/np.log(Y.mean()))       #gamma校正,默认不进行该操作
    return Y*255
### defog

#  Gaussian filtering
def GaussianBlur(img):
    image= cv2.GaussianBlur(img, (5, 5), 0)  # 可以更改核大小
    return image
############################## =====1.MUIE===== ##############################


############################## =====2.HCFF===== ##############################

class HCFF(nn.Module):
    def __init__(self, c1, c2):
        super(HCFF, self).__init__()
        self.layer1_1 = Conv(c1, c2//4, 3, 1)
        self.layer1_2 = Conv(c2//4, c2//4, 3, 1)
        self.cv1 = Conv(c1+c2//2, c2//2, 3, 2)
        self.cv2 = Conv(c2//4, c2//4, 3, 1)
        self.cv3 = Conv(c2//4, c2//4, 1, 1)
        self.cv4 = Conv(c2, c2, 1, 1)

    def forward(self, x):
        l1_1 = self.layer1_1(x)    # c
        l1_2 = self.layer1_2(l1_1) # c =
        input_l2 = torch.cat((x, l1_1, l1_2), 1) # c =
        y = list(self.cv1(input_l2).chunk(2, 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))
############################## =====2.HCF===== ##############################


############################## =====3.SFE===== ##############################

class SFE(nn.Module):
    def __init__(self, c1, c2):
        super(SFE, self).__init__()
        c_ = c2//2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_*4, c2, 1, 1)
        self.CBAM = CBAM(c_)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        x3 = self.CBAM(x2)
        x4 = self.cv3(x3)
        return self.cv4(torch.cat((x1, x2, x3, x4), 1))


class CBAM(nn.Module):
    def __init__(self, input):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(input)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 平均池化，是取整个channel所有元素的均值 [3,5,5] => [3,1,1]
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 最大池化，是取整个channel所有元素的最大值[3,5,5] => [3,1,1]

        # shared MLP
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):                               # 输入x = [b, c, 56, 56]
        avg_out = torch.mean(x, dim=1, keepdim=True)    # avg_out = [b, 1, 56, 56]  求x的每个像素在所有channel相同位置上的平均值
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # max_out = [b, 1, 56, 56]  求x的每个像素在所有channel相同位置上的最大值
        x = torch.cat([avg_out, max_out], dim=1)        # x = [b, 2, 56, 56]  concat操作
        x = self.conv1(x)                               # x = [b, 1, 56, 56]  卷积操作，融合avg和max的信息，全方面考虑
        return self.sigmoid(x)
############################## =====3.SFE===== ##############################


############################## =====4.CIoU, MDPIoU===== ##############################

def bbox_iou(box1, box2, xywh=True, CIoU=False, MDPIoU=True, feat_h=640, feat_w=640, eps=1e-7):
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps
    # IoU
    iou = inter / union
    if CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
        rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center dist ** innovations

        v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
        with torch.no_grad():
            alpha = v / (v - iou + (1 + eps))
            return iou - (rho2 / c2 + v * alpha)  # CIoU
    elif MDPIoU:
        d1 = (b2_x1 - b1_x1) ** 2 + (b2_y1 - b1_y1) ** 2
        d2 = (b2_x2 - b1_x2) ** 2 + (b2_y2 - b1_y2) ** 2
        mpdiou_hw_pow = feat_h ** 2 + feat_w ** 2
        return iou - d1 / mpdiou_hw_pow - d2 / mpdiou_hw_pow  # MPDIoU
############################## =====4.CIoU, MDPIoU===== ##############################