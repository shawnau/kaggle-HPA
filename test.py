import cv2


def test_train_loader():
    from dl_backbone.config import cfg
    from dl_backbone.data.build import make_data_loader
    data_loader = make_data_loader(cfg)
    for iteration, (images, targets, indices) in enumerate(data_loader):
        print("Image Batch Size: ", images.tensors.size())
        print("Target Batch Size: ", targets.size())
        print("Indices: ", indices)
        for idx in range(len(images.tensors)):
            if idx % 4 == 0:
                np_img = images.tensors[idx][[0, 1, 2], :, :].permute(1, 2, 0).numpy()*255
                cv2.imwrite('train_output_%d.jpg'%idx, np_img)
        break


def test_test_loader():
    from dl_backbone.config import cfg
    from dl_backbone.data.build import make_data_loader
    data_loader = make_data_loader(cfg, is_train=False)
    for iteration, (images, targets, indices) in enumerate(data_loader):
        print("Image Batch Size: ", images.tensors.size())
        print("Target Batch Size: ", targets.size())
        print("Indices: ", indices)
        for idx in range(len(images.tensors)):
            if idx % 2 == 0:
                np_img = images.tensors[idx][[0, 1, 2], :, :].permute(1, 2, 0).numpy()*255
                cv2.imwrite('test_output_%d.jpg'%idx, np_img)
        break


if __name__ == "__main__":
    test_train_loader()
    test_test_loader()
