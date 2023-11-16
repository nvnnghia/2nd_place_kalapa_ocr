import cv2
import numpy as np
from glob import glob 
from tqdm import tqdm  
import json
from PIL import ImageFont, ImageDraw, Image, ImageFilter
import random  
import math 
import unicodedata
import pandas as  pd 
import os  

def extract_boxes(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Remove horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25,1))
    detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(thresh, [c], -1, 0, -1)

    # Dilate to merge into a single contour
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30,1))
    dilate = cv2.dilate(thresh, vertical_kernel, iterations=3)

    # Find contours, sort for largest contour and extract ROI
    cnts, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:-1]
    x1,y1,x2,y2 = 0,0,0,0
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        x1 = x 
        y1 = y 
        x2 = x + w  
        y2 = y + h  
        break
    return x1,y1,x2,y2

class BackgroundGenerator(object):
    @classmethod
    def gaussian_noise(cls, height, width):
        """
            Create a background with Gaussian noise (to mimic paper)
        """

        # We create an all white image
        image1 = np.ones((height, width)) * 255
        image2 = np.ones((height, width)) * 255
        image3 = np.ones((height, width)) * 255

        # We add gaussian noise
        cv2.randn(image1, 235, 10)
        cv2.randn(image2, 235, 10)
        cv2.randn(image3, 235, 10)
        image = np.stack([image1, image2, image3], -1).astype(np.uint8)
        # print(image.shape)
        return Image.fromarray(image)#.convert('RGB')

    @classmethod
    def plain_white(cls, height, width):
        """
            Create a plain white background
        """
        bg_img = random.randint(150,255)*np.ones((height,width,3)).astype(np.uint8)
        bg_img = Image.fromarray(bg_img)

        return bg_img

    @classmethod
    def quasicrystal(cls, height, width):
        """
            Create a background with quasicrystal (https://en.wikipedia.org/wiki/Quasicrystal)
        """

        image = Image.new("L", (width, height))
        pixels = image.load()

        frequency = random.random() * 30 + 20  # frequency
        phase = random.random() * 2 * math.pi  # phase
        rotation_count = random.randint(10, 20)  # of rotations

        for kw in range(width):
            y = float(kw) / (width - 1) * 4 * math.pi - 2 * math.pi
            for kh in range(height):
                x = float(kh) / (height - 1) * 4 * math.pi - 2 * math.pi
                z = 0.0
                for i in range(rotation_count):
                    r = math.hypot(x, y)
                    a = math.atan2(y, x) + i * math.pi * 2.0 / rotation_count
                    z += math.cos(r * math.sin(a) * frequency + phase)
                c = int(255 - round(255 * z / rotation_count))
                pixels[kw, kh] = c  # grayscale
        return image.convert('RGB')

    @classmethod
    def canvas(cls, height, width):
        # Create white canvas and get drawing context
        # w, h = 250, 50
        img  = Image.new('RGB', (width, height), color = 'white')
        draw = ImageDraw.Draw(img)

        # Let's have repeatable, deterministic randomness
        # seed(37)

        # Generate a basic random colour, random RGB values 10-245
        R, G, B = random.randint(10,245), random.randint(10,245), random.randint(10,245),

        # Draw a random number of circles between 40-120
        cmin = random.randint(50, 70)
        cmax = random.randint(90,120)
        for _ in range(cmin,cmax):
           # Choose RGB values for this circle, somewhat close (+/-10) to basic RGB
           r = R + random.randint(-10,10)
           g = G + random.randint(-10,10)
           b = B + random.randint(-10,10)
           diam = random.randint(5,11)
           x, y = random.randint(0,width), random.randint(0,height)
           draw.ellipse([x,y,x+diam,y+diam], fill=(r,g,b))

        # Blur the background a bit
        res = img.filter(ImageFilter.BoxBlur(3))
        return img.convert('RGB')


import albumentations as A
transform = A.Compose([
    A.OneOf([
       A.GridDistortion(always_apply=False, p=1.0, num_steps=1, distort_limit=(-0.05, 0.05), interpolation=1, border_mode=0, value=(0, 0, 0), mask_value=None, normalized=1),
       A.Perspective(always_apply=False, p=1.0, scale=(0.01, 0.05), keep_size=1, pad_mode=0, pad_val=(0, 0, 0), mask_pad_val=0, fit_output=1, interpolation=1),
       A.OpticalDistortion(always_apply=False, p=1.0, distort_limit=(-0.2, 0.2), shift_limit=(-0.00, 0.00), interpolation=0, border_mode=0, value=(0, 0, 0), mask_value=None)
       ], p = 1.0
       )
    ])

# all_paths = glob('test_images/*/*.jpg')
df = pd.read_csv('../train_folds.csv')

os.makedirs('aug', exist_ok = True)

ignore = [[46,0], [46,1], [67,2], [67,8], [67,9], [67,13], [67,19], [67,24], [68,-1],[73,1], [73,16], [73,17], [73,23], [73,24],
                        [76,3], [76,4], [76,5], [77,0], [77,1], [77,2], [77,3], [77,25], [87,4], [87,6],[87,8],[87,13],[87,21],
                        [87,26],[148,6],[148,22],[149,12],[167,-1],[195,-1]]

# print(len(all_paths))
# for path in tqdm(all_paths):
for cc, row in tqdm(df.iterrows()):
    # if cc < 1462:
    #     continue
    path = '../training_data/images/' + row['img_path']
    label = row['text']

    writer = path.split("/")[-2]
    name = path.split("/")[-1]

    is_skip = False
    for ig in ignore:
        if writer == ig[0]:
            if name == ig[1] or ig[1] == -1:
                is_skip = True 
    if is_skip:
        print(path)
        continue

    image = cv2.imread(path)
    im_h, im_w = image.shape[:2]
    if im_w > 1280:
        ratio = 1280/im_w 
        new_h = int(im_h*ratio)
        image = cv2.resize(image, (1280, new_h))
    x1,y1,x2,y2 = extract_boxes(image)

    if x2 - x1 < 10 or y2 - y1 < 10:
        print(cc, x1, y1, x2, y2)
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    image = thresh[y1:y2, x1:x2]
    for i in range(10):
        img = transform(image=image.copy())["image"]

        im_h, im_w = img.shape[:2]
        ratio_h = 0.01*random.randint(10,50)
        ratio_w = 0.01*random.randint(10,70)
        new_h = int(im_h*(1+ratio_h))
        new_w = int(im_w*(1+ratio_w))
        temp = np.zeros((new_h, new_w))
        start_h = random.randint(1,new_h - im_h)
        start_w = random.randint(1,(new_w - im_w)//2)

        temp[start_h:start_h+im_h, start_w:start_w+im_w] = img
        img = temp.astype(np.uint8) 

        height = random.randint(50,70)
        width = random.randint(1100,1300)
        background_type = random.randint(0, 3)
        if background_type == 0:
            pil_img = BackgroundGenerator.gaussian_noise(height, width)
        elif background_type == 1:
            pil_img = BackgroundGenerator.plain_white(height, width)
        elif background_type == 2:
            pil_img = BackgroundGenerator.quasicrystal(height, width)
        else:
            pil_img = BackgroundGenerator.canvas(height, width)

        c1 = img*random.randint(10, 170)
        c2 = img*random.randint(10, 170)
        c3 = img*random.randint(10, 170)
        bg_img = np.stack([c1,c2,c3], -1)
        bg_img = Image.fromarray(bg_img).convert('RGB')

        bg_img = bg_img.resize((width, height))
        r, g, b = bg_img.split()
        def apply_function(channel):
            return channel.point(lambda x: 0 if x > 250 or x < 5 else 255, '1')

        mask = apply_function(r)
        pil_img.paste(bg_img, (5, 5), mask=mask)

        # name = fontpath.split('/')[-1][:-4]
        pil_img.save(f'aug/{writer}_{name}_{i:06d}.jpg')

        with open(f'aug/{writer}_{name}_{i:06d}.txt', 'w') as f:
            f.write(label)


    # break