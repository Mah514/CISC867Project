from PIL import ImageFilter
import random
import torchvision.transforms as transforms
from PIL import Image
import copy
import numpy as np
import albumentations as A


class GaussianBlur(object):
    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class Custom_Distoration(object):
    def __init__(self, input_rows=224, input_cols=224):
        self.input_rows = input_rows
        self.input_cols = input_cols
    def __call__(self, org_img):
        org_img = np.array(org_img)
        r = random.random()
        if r <= 0.3:  #cut-out
            cnt = 10
            while cnt > 0:
                block_noise_size_x, block_noise_size_y = random.randint(10, 70), random.randint(10, 70)
                noise_x, noise_y = random.randint(3, self.input_rows - block_noise_size_x - 3), random.randint(3,self.input_cols - block_noise_size_y - 3)
                org_img[noise_x:noise_x + block_noise_size_x, noise_y:noise_y + block_noise_size_y,:] = 0
                cnt = cnt - 1
                if random.random() < 0.1:
                    break
        elif 0.3 < r <= 0.35:  #cut-out
            image_temp = copy.deepcopy(org_img)
            org_img[:, :,:] = 0
            cnt = 10
            while cnt > 0:
                block_noise_size_x, block_noise_size_y = self.input_rows - random.randint(50,70), self.input_cols - random.randint(50, 70)
                noise_x, noise_y = random.randint(3, self.input_rows - block_noise_size_x - 3), random.randint(3,self.input_cols - block_noise_size_y - 3)
                org_img[noise_x:noise_x + block_noise_size_x,noise_y:noise_y + block_noise_size_y,:] = image_temp[noise_x:noise_x + block_noise_size_x, noise_y:noise_y + block_noise_size_y,:]
                cnt = cnt - 1
                if random.random() < 0.1:
                    break
        elif 0.35 < r <= 0.65:  #shuffling
            cnt = 10
            image_temp = copy.deepcopy(org_img)
            while cnt > 0:
                while True:
                    block_noise_size_x, block_noise_size_y = random.randint(10, 15), random.randint(10, 15)
                    noise_x1, noise_y1 = random.randint(3, self.input_rows - block_noise_size_x - 3), random.randint(3,self.input_cols - block_noise_size_y - 3)
                    noise_x2, noise_y2 = random.randint(3, self.input_rows - block_noise_size_x - 3), random.randint(3,self.input_cols - block_noise_size_y - 3)
                    if ((noise_x1 > noise_x2 + block_noise_size_x) or (noise_x2 > noise_x1 + block_noise_size_x) or (noise_y1 < noise_y2 + block_noise_size_y) or (noise_y2 < noise_y1 + block_noise_size_y)):
                         break

                org_img[noise_x1:noise_x1 + block_noise_size_x, noise_y1:noise_y1 + block_noise_size_y,:] = image_temp[noise_x2:noise_x2 + block_noise_size_x,noise_y2:noise_y2 + block_noise_size_y,:]
                org_img[noise_x2:noise_x2 + block_noise_size_x, noise_y2:noise_y2 + block_noise_size_y,:] = image_temp[noise_x1:noise_x1 + block_noise_size_x,noise_y1:noise_y1 + block_noise_size_y,:]
                cnt = cnt - 1
        return Image.fromarray(org_img)

class Transform:
    def __init__(self,mode):
        self.mode=mode
        self.crop_transform=transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.))])
        
#        self.transform = transforms.Compose([transforms.RandomApply([
#            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
#        ], p=0.8),
#        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
#        transforms.RandomHorizontalFlip()])
        
        self.transform = A.Compose(
            [#A.CLAHE(p=0.6),
             #A.Emboss(p=0.6),
             #A.FancyPCA(p=0.6),
             #A.RandomToneCurve(p=0.6),
             A.ColorJitter (brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, always_apply=False, p=0.8),
             #A.ChannelShuffle(p=0.5),
             #A.Sharpen(p=0.6),
             random.choice(A.GaussianBlur (blur_limit=0, sigma_limit=[0.1, 2.0], always_apply=False, p=0.6),
                           A.GlassBlur(p=0.6),
                           A.RandomFog(p=0.6),
                           A.ZoomBlur(max_factor=1.05,p=0.6)),
             A.HorizontalFlip(p=0.5)])
        
        self.more_transforms = A.Compose(
            [
                random.choice(A.GaussNoise(var_limit=(5,15), p=0.6),
                              A.ISONose(intensity=(1,1), p=0.6),
                              A.Spatter(p=0.6)),
                A.RandomGamma(gamma_limit=(150,250),p=0.8),
                A.RandomGravel(gravel_roi=(0, 0, 1, 1), number_of_patches=50,p=0.5),
                A.RandomRain(p=0.5),
                A.RandomSnow(brightness_coeff=1.5, p=0.5),
                A.RandomShadow(p=0.5),
                A.Defocus(radius=(5,10), p=0.8)
                
            ]
        )
        
        self.reconstruction_transform=transforms.Compose([
            Custom_Distoration(224,224),
            transforms.ToTensor(),
        ])
        self.to_tensor = transforms.Compose([
            transforms.ToTensor()
        ])

    def __call__(self, x):
        y1_orig = self.crop_transform(Image.fromarray(x))
        y2_orig = self.crop_transform(Image.fromarray(x))
        y1 = self.transform(image=np.asarray(y1_orig))['image']
        y2 = self.transform(image=np.asarray(y2_orig))['image']
        if self.mode.lower() == 'di':
            y1=self.to_tensor(y1)
            y2=self.to_tensor(y2)
            return [y1,y2]
        else:
            #y1 = self.more_transforms(y1)['image']
            #y2 = self.more_transforms(y2)['image']
            y1 = self.reconstruction_transform(Image.fromarray(y1))
            y2 = self.reconstruction_transform(Image.fromarray(y2))
            y1_orig_1c = y1_orig.convert('L')
            y1_orig_1c = np.array(y1_orig_1c) / 255.0
            y1_orig_1c = np.expand_dims(y1_orig_1c, axis=0).astype('float32')
            return [y1, y2, y1_orig_1c]

