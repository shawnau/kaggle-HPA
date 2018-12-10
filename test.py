import torch
import cv2
from dl_backbone.config import cfg


def test_train_loader():
    cfg.SOLVER.IMS_PER_BATCH = 10
    from dl_backbone.data.build import make_data_loader
    data_loader = make_data_loader(cfg, is_train=False)
    for iteration, (images, targets, indices) in enumerate(data_loader):
        print("Image Batch Size: ", images.size())
        print("Target Batch Size: ", targets.size())
        print("Indices: ", indices)
        for idx in range(len(images)):
            np_img = images[idx][[2, 1, 0], :, :].permute(1, 2, 0).numpy()*255
            cv2.imwrite('train_output_%d.jpg'%idx, np_img)
        break


def test_test_loader():
    cfg.TEST.IMS_PER_BATCH = 10
    from dl_backbone.data.build import make_data_loader
    data_loader = make_data_loader(cfg, is_train=False)
    for iteration, (images, targets, indices) in enumerate(data_loader):
        print("Image Batch Size: ", images.size())
        print("Target Batch Size: ", targets.size())
        print("Indices: ", indices)
        for idx in range(len(images)):
            np_img = images[idx][[2, 1, 0], :, :].permute(1, 2, 0).numpy()*255
            cv2.imwrite('test_output_%d.jpg'%idx, np_img)
        break


def test_lr_scheduler():
    from dl_backbone.solver.build import make_optimizer, make_lr_scheduler
    import torch.nn as nn

    class TestModule(nn.Module):
        def __init__(self):
            super(TestModule, self).__init__()
            self.param = nn.Parameter(torch.Tensor(5))

        def forward(self, x):
            return x

    model = TestModule()
    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)
    for i in range(cfg.SOLVER.MAX_ITER):
        scheduler.step()
        if ((i < cfg.SOLVER.WARMUP_ITERS) and (i % 100 == 0)) or (i % 1000 == 0):
            print("iter: %d, lr: %.6f" % (i, scheduler.get_lr()[0]))


if __name__ == "__main__":
    test_train_loader()
    test_test_loader()
    test_lr_scheduler()
