import os


def _get_env_var(varname):
    value = os.getenv(varname)
    if not value:
        raise EnvironmentError("Required environment variable '{}' is not set.".format(varname))
    return value


class ModelPaths(object):
    def __init__(self):
        pass

    @staticmethod
    def checkpoint_base_dir():
        path = os.path.abspath(os.getcwd())
        if 'kaggle' in path:
            return os.path.join(_get_env_var('/kaggle/working/'), 'checkpoints')
        else:
            return os.path.join(_get_env_var('/home/kasaei2/FarnooshArefi/VIS/'), 'checkpoints')

    @staticmethod
    def pretrained_backbones_dir():
        path = os.path.abspath(os.getcwd())
        if 'kaggle' in path:
            return os.path.join(_get_env_var('/kaggle/working/'), 'pretrained')
        else:
            return os.path.join(_get_env_var('/home/kasaei2/FarnooshArefi/VIS/'), 'pretrained')

    @staticmethod
    def pretrained_maskrcnn_x101_fpn():
        return os.path.join(ModelPaths.pretrained_backbones_dir(), 'e2e_mask_rcnn_X_101_32x8d_FPN_1x.pth')

    @staticmethod
    def pretrained_maskrcnn_r50_fpn():
        return os.path.join(ModelPaths.pretrained_backbones_dir(), 'e2e_mask_rcnn_R_50_FPN_1x.pth')

    @staticmethod
    def pretrained_maskrcnn_r101_fpn():
        return os.path.join(ModelPaths.pretrained_backbones_dir(), 'e2e_mask_rcnn_R_101_FPN_1x.pth')
