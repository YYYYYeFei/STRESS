import numpy as np
import scanpy as sc
import torch
from torch import nn

class LeidenLoss(nn.Module):
    def __init__(self,args, cluster_type='lsgan', loss_weight=1.0):
        super(LeidenLoss, self).__init__()
        self.cluster_type = cluster_type
        self.loss_weight = loss_weight
        self.args=args

        if self.cluster_type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.cluster_type == 'lsgan':
            self.loss = nn.MSELoss()
        else:
            raise NotImplementedError(
                f'GAN type {self.gan_type} is not implemented.')

    def get_target_label(self, input):
        print("对预测结果进行聚类.......")
        """Get target label.

        Args:
            input (Tensor): Input tensor.
            target_is_real (bool): Whether the target is real or fake.

        Returns:
            (bool | Tensor): Target tensor. Return bool for wgan, otherwise,
                return Tensor.
        """
        data=input
        print('data.shape：',data.shape)
        # data = np.load(file_path)
        cutoff_value=-1
        visualize=1
        # 2. 归一化和标准化数据
        # 找到非空细胞
        data[data < cutoff_value] = 0
        # non_empty_cells = np.any(data > 0, axis=2)
        non_empty_cells = torch.nonzero(data > 0, as_tuple=False)
        print('non_empty_cells.shape:',non_empty_cells.shape)
        data_non_empty = np.transpose(data[non_empty_cells].cpu().numpy(),(0,1,2))
        # 构建AnnData对象
        adata = sc.AnnData(data_non_empty.reshape(-1, data.shape[2]))

        # 3. 保存标准化后的数据
        # filename = os.path.splitext(os.path.basename(file_path))[0]
        # standardized_output_path = os.path.join(output_folder, f'{filename}_standardized.npy')
        # Ensure output shape matches 64*64*2000 and fill with zeros if necessary
        standardized_data = np.zeros_like(data)
        standardized_data[non_empty_cells] = adata.X.reshape(data_non_empty.shape)
        # np.save(standardized_output_path, standardized_data)

        # 4. 进行leiden聚类
        sc.pp.neighbors(adata, use_rep='X')
        sc.tl.leiden(adata)
        non_empty_cells_coords = np.argwhere(non_empty_cells.reshape(-1))
        x_coords, y_coords = non_empty_cells_coords[:, 0] % 64, non_empty_cells_coords[:, 0] // 64
        clusters = adata.obs['leiden'].astype(int)
        print(clusters.shape)

        # 6. 生成新的npy文件
        cluster_matrix = np.full((64, 64), -1, dtype=int)
        for i, (x, y) in enumerate(zip(x_coords, y_coords)):
            cluster_matrix[y, x] = clusters[i]
        print(cluster_matrix.shape)
        cluster_matrix=torch.FloatTensor(cluster_matrix).to(self.args.device).requires_grad(True)
        return cluster_matrix

    def forward(self, input, target_label):
        """
        Args:
            input (Tensor): The input for the loss module, i.e., the network
                prediction.
            target_is_real (bool): Whether the targe is real or fake.
            is_disc (bool): Whether the loss for discriminators or not.
                Default: False.

        Returns:
            Tensor: GAN loss value.
        """

        loss=0
        for i in range(input.shape[0]):
            predict_label = self.get_target_label(input[i])
            loss += self.loss(predict_label, target_label[i])
        # loss_weight is always 1.0 for discriminators
        # return loss if is_disc else loss * self.loss_weight
        return loss/input.shape[0]

        # loss=self.loss(input, target_label)
        #
        # return loss


def ntxent_loss(args, features, temp):
    """
       NT-Xent Loss.

       Args:
           z1: The learned representations from first branch of projection head
           z2: The learned representations from second branch of projection head
       Returns:
           Loss
       """
    LABELS = torch.cat([torch.arange(args.batch_size) for i in range(2)], dim=0)
    LABELS = (LABELS.unsqueeze(0) == LABELS.unsqueeze(1)).float()  # one-hot representations
    LABELS = LABELS.to(args.device)

    similarity_matrix = torch.matmul(features, features.T)
    mask = torch.eye(LABELS.shape[0], dtype=torch.bool).to(args.device)

    # print('LABELS.shape[0]',LABELS.shape[0])
    # print('similarity_matrix.shape[0]', similarity_matrix.shape[0])
    # print('mask.shape[0]', mask.shape)

    labels = LABELS[~mask].view(LABELS.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(args.device)
    logits = logits / temp
    return logits, labels


import torchvision.models as models
class PerceptualVGG1(nn.Module):

    def __init__(self,
                 layer_name_list,
                 vgg_type='vgg19',
                 use_input_norm=False):
        super().__init__()
        self.layer_name_list = layer_name_list
        self.use_input_norm = use_input_norm

        # get vgg model and load pretrained vgg weight
        # remove _vgg from attributes to avoid `find_unused_parameters` bug

        assert vgg_type in ('vgg16', 'vgg19')
        if vgg_type == 'vgg16':
            _vgg = models.vgg16(pretrained=True)
        else:
            _vgg = models.vgg19(pretrained=True)

        num_layers = max(map(int, layer_name_list)) + 1
        assert len(_vgg.features) >= num_layers
        # only borrow layers that will be used from _vgg to avoid unused params
        self.vgg_layers = _vgg.features[:num_layers]

        if self.use_input_norm:
            # the mean is for image with range [0, 1]
            self.register_buffer(
                'mean',
                torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            # the std is for image with range [-1, 1]
            self.register_buffer(
                'std',
                torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        for v in self.vgg_layers.parameters():
            v.requires_grad = False

        self.inits = nn.Conv2d(2048, 3, kernel_size=3, stride=2, padding=1, bias=False)

    def forward(self, x):
        x=self.inits(x.type(torch.float32))
        """Forward function.
        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
        Returns:
            Tensor: Forward results.
        """

        if self.use_input_norm:
            x = (x - self.mean) / self.std
        output = {}

        for name, module in self.vgg_layers.named_children():
            x = module(x)
            if name in self.layer_name_list:
                output[name] = x.clone()
        return output


class PerceptualLoss1(nn.Module):
    """Perceptual loss with commonly used style loss.

    Args:
        layers_weights (dict): The weight for each layer of vgg feature.
            Here is an example: {'4': 1., '9': 1., '18': 1.}, which means the
            5th, 10th and 18th feature layer will be extracted with weight 1.0
            in calculating losses.
    """

    def __init__(self,
                 layer_weights,
                 vgg_type='vgg19',
                 use_input_norm=False,
                 perceptual_weight=1.0,
                 style_weight=0.,
                 norm_img=False,
                 criterion='l1'):
        super().__init__()
        self.norm_img = norm_img
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.layer_weights = layer_weights
        self.vgg = PerceptualVGG1(
            layer_name_list=list(layer_weights.keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm)

        criterion = criterion.lower()
        if criterion == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif criterion == 'mse':
            self.criterion = torch.nn.MSELoss()
        else:
            raise NotImplementedError(
                f'{criterion} criterion has not been supported in'
                ' this version.')

    def forward(self, x, gt):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """

        if self.norm_img:
            x = (x + 1.) * 0.5
            gt = (gt + 1.) * 0.5
        # extract vgg features
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())

        # calculate perceptual loss
        if self.perceptual_weight > 0:
            percep_loss = 0
            for k in x_features.keys():
                percep_loss += self.criterion(
                    x_features[k], gt_features[k]) * self.layer_weights[k]
            percep_loss *= self.perceptual_weight
        else:
            percep_loss = None

        # calculate style loss
        if self.style_weight > 0:
            style_loss = 0
            for k in x_features.keys():
                style_loss += self.criterion(
                    self._gram_mat(x_features[k]),
                    self._gram_mat(gt_features[k])) * self.layer_weights[k]
            style_loss *= self.style_weight
        else:
            style_loss = None

        return percep_loss

    def _gram_mat(self, x):
        """Calculate Gram matrix.

        Args:
            x (torch.Tensor): Tensor with shape of (n, c, h, w).

        Returns:
            torch.Tensor: Gram matrix.
        """
        (n, c, h, w) = x.size()
        features = x.view(n, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram

class L1_Charbonnier_loss(nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6 # already use square root

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss