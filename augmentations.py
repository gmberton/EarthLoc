
import torchvision.transforms as tfm


INTERPOLATION = tfm.functional.InterpolationMode.BILINEAR


class ImprovedRandomPerspective(tfm.RandomPerspective):
    def __init__(self, distortion_scale=0.5, p=1):
        super().__init__(distortion_scale=distortion_scale, p=p)
    @staticmethod
    def crop_background(endpoints, tr_img):
        left_y = max(endpoints[0][1], endpoints[1][1])
        left_x = max(endpoints[0][0], endpoints[3][0])
        highest_point = min(endpoints[2][1], endpoints[3][1])
        lowest_point = left_y
        rightmost_point = min(endpoints[1][0], endpoints[2][0])
        leftmost_point = left_x
        return tfm.functional.crop(tr_img, left_y, left_x,
                                   highest_point - lowest_point,
                                   rightmost_point - leftmost_point)
    def forward(self, images):
        assert len(images.shape) == 4, f"images should be a batch of images, but it has shape {images.shape}"
        B, C, H, W = images.shape
        startpoints, endpoints = self.get_params(W, H, distortion_scale=self.distortion_scale)
        images = tfm.functional.perspective(images, startpoints, endpoints, interpolation=INTERPOLATION)
        images = self.crop_background(endpoints, images)
        images = tfm.functional.center_crop(images, min(images.shape[-2:]))
        return images


def get_my_augment(distortion_scale=0.8, crop_size=700, final_size=400, rand_rot=45,
                   brightness=0.9, contrast=0.8, saturation=0.9, hue=0.05):
    return tfm.Compose([
        tfm.RandomRotation(rand_rot, interpolation=INTERPOLATION),
        tfm.RandomCrop(crop_size),
        ImprovedRandomPerspective(distortion_scale=distortion_scale),
        tfm.Resize([final_size, final_size], antialias=True, interpolation=INTERPOLATION),
        tfm.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue),
        tfm.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
