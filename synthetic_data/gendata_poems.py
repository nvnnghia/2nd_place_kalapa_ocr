import cv2
import numpy as np
from tqdm import tqdm  
import json
from PIL import ImageFont, ImageDraw, Image, ImageFilter
from glob import glob
import random  
import math 
import unicodedata
import pandas as pd 
import albumentations as A
import os  

transform = A.Compose([
    A.OneOf([
       A.GridDistortion(always_apply=False, p=1.0, num_steps=1, distort_limit=(-0.05, 0.05), interpolation=1, border_mode=0, value=(0, 0, 0), mask_value=None, normalized=1),
       A.Perspective(always_apply=False, p=1.0, scale=(0.01, 0.05), keep_size=1, pad_mode=0, pad_val=(0, 0, 0), mask_pad_val=0, fit_output=1, interpolation=1),
       A.OpticalDistortion(always_apply=False, p=1.0, distort_limit=(-0.2, 0.2), shift_limit=(-0.00, 0.00), interpolation=0, border_mode=0, value=(0, 0, 0), mask_value=None)
       ], p = 1.0
       )
    ])

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

def gen_text(text):
    text = unicodedata.normalize('NFC', text).strip()

    text = text.replace('óa', 'oá').replace('òa', 'oà')
    # print(text)

    fontpath = random.choice(all_fonts)
    font = ImageFont.truetype(fontpath, random.randint(25,35))

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

    text_w, text_h = font.getsize(text)
    # print(text_w, text_h)

    ratio_h = 0.01*random.randint(10,70)
    ratio_w = 0.01*random.randint(10,70)
    text_h = int(text_h*(1+ratio_h))
    text_w = int(text_w*(1+ratio_w))

    try:
        bg_img = 255*np.ones((text_h, text_w,3)).astype(np.uint8)
        bg_img = Image.fromarray(bg_img)
        draw = ImageDraw.Draw (bg_img)
        draw.text((random.randint(1,6) , random.randint(1,6)) , text, font=font, fill=(random.randint(10, 170), random.randint(10, 170), random.randint(10, 170)))
    except:
        return

    use_aug = 0
    #use augmentation
    if random.random()<0.4:
        use_aug = 1
        try:
            open_cv_image = np.array(bg_img) 
            img = transform(image=open_cv_image)["image"]
            bg_img = Image.fromarray(img)
        except:
            return
    #~use augmentation

    bg_img = bg_img.resize((width, height))

    r, g, b = bg_img.split()
    def apply_function(channel):
        return channel.point(lambda x: 0 if x > 250 or x < 5 else 255, '1')

    mask = apply_function(r)
    pil_img.paste(bg_img, (5, 5), mask=mask)

    name = fontpath.split('/')[-1][:-4]
    pil_img.save(f'poems/{count:06d}_{use_aug}_{name}.jpg')

    with open(f'poems/{count:06d}_{use_aug}_{name}.txt', 'w') as f:
        f.write(text)

import re
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub('\s+', ' ', text)
    text = text.strip()
    # text = '[start] ' + text + ' [end]'
    return text

os.makedirs('poems', exist_ok = True)

if __name__ == '__main__':
    all_fonts = glob('fonts/*.*')
    count = 1
    df = pd.read_csv('poems_dataset.csv')
    df = df.sample(frac=0.2).reset_index(drop=True)
    # df = df.head(10)
    for i, row in tqdm(df.iterrows()):
        text = row['content']
        text = text.split('<\n>')
        if len(text) < 2:
            continue
        text = random.sample(text, 2)
        for t in text:
            t = preprocess(t)
            if len(t) < 10 or len(t) > 70:
                continue
            gen_text(t)

            count += 1
            print(count, end='\r')

        if count > 25000: #generate ~25k for now
            break



