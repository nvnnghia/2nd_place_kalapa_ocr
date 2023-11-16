import cv2
import numpy as np
from tqdm import tqdm  
import json
from PIL import ImageFont, ImageDraw, Image, ImageFilter
from glob import glob
import random  
import math 
import unicodedata
import os  

import albumentations as A
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

def gen_text(st, wa):
    prob = random.random()
    code = data['code']
    if random.random()<0.1:
        code = code.lower()
    if prob < 0.4:
        text = wa['name'] + ' ' + d['name'] + ' ' + data['name']
    elif prob < 0.8:
        text = wa['pre'] + ' ' + wa['name'] + ' ' + d['pre'] + ' ' + d['name'] + ' ' + data['name']
    elif prob < 0.9:
        text = wa['pre'] + ' ' + wa['name'] + ' ' + d['pre'] + ' ' + d['name'] + ' ' + code
    else:
        text = wa['pre'] + ' ' + wa['name'] + ' ' + d['name'] + ' ' + code

    text = unicodedata.normalize('NFC', text).strip()

    tp_prob = random.random()
    if tp_prob < 0.2:
        text = text.replace('Thành phố', 'TP')
    elif tp_prob < 0.4:
        text = text.replace('Thành phố', 'Tp')
    elif tp_prob < 0.5:
        text = text.replace('Thành phố', 'tp')

    p_prob = random.random()
    if p_prob < 0.2:
        text = text.replace('Phường', 'P')
    elif p_prob < 0.4:
        text = text.replace('Phường', 'p')

    p_prob = random.random()
    if p_prob < 0.2:
        text = text.replace('Huyện', 'H')
    elif p_prob < 0.3:
        text = text.replace('Huyện', 'h')

    p_prob = random.random()
    if p_prob < 0.2:
        text = text.replace('Quận', 'Q')
    elif p_prob < 0.3:
        text = text.replace('Quận', 'q')

    p_prob = random.random()
    if p_prob < 0.2:
        text = text.replace('Thị trấn', 'TT')
    elif p_prob < 0.4:
        text = text.replace('Thị trấn', 'Tt')

    p_prob = random.random()
    if p_prob < 0.2:
        text = text.replace('Tỉnh Lộ ', 'TL')

    if len(st)>0:
        if random.random()<0.5:
            num = random.randint(1,500)
            num = f'{num}'
        else:
            num1 = random.randint(1,500)
            num2 = random.randint(1,500)
            num = f'{num1}/{num2}'

        ab_prob = random.random()
        if ab_prob < 0.1:
            num += 'a'
        elif ab_prob < 0.2: 
            num += 'b'

        text = num + ' ' +  st + ' ' + text

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

    bg_img = 255*np.ones((text_h, text_w,3)).astype(np.uint8)
    bg_img = Image.fromarray(bg_img)
    draw = ImageDraw.Draw (bg_img)
    draw.text((random.randint(1,6) , random.randint(1,6)) , text, font=font, fill=(random.randint(10, 170), random.randint(10, 170), random.randint(10, 170)))

    #use augmentation
    if random.random()<0.5:
        open_cv_image = np.array(bg_img) 
        img = transform(image=open_cv_image)["image"]
        bg_img = Image.fromarray(img)
    #~use augmentation

    bg_img = bg_img.resize((width, height))

    r, g, b = bg_img.split()
    def apply_function(channel):
        return channel.point(lambda x: 0 if x > 250 or x < 5 else 255, '1')

    mask = apply_function(r)
    pil_img.paste(bg_img, (5, 5), mask=mask)

    name = fontpath.split('/')[-1][:-4]
    pil_img.save(f'address/{count:06d}_{name}.jpg')

    with open(f'address/{count:06d}_{name}.txt', 'w') as f:
        f.write(text)

os.makedirs('address', exist_ok = True) 
if __name__ == '__main__':
    all_fonts = glob('fonts/*.*')

    count = 1
    for i in [0,1]:
        print('=====', i)
        all_json = glob('vietnam_dataset/data/*.json')
        for js in all_json:
            print(js)
            json_string = open(js)
            data = json.load(json_string)
            districts = data['district']
            name = data['name']
            for d in districts:
                streets = d['street']

                if i == 0:
                    # use all wawrd
                    for wa in d['ward']:
                        st = ''
                        if len(streets)>0:
                            st = random.choice(streets)

                        if random.random()<0.5:
                            st = ''
                        
                        text = gen_text(st, wa)
                        count += 1
                        print(count, end='\r')

                #use all street
                for st in streets:
                    wa = random.choice(d['ward'])
                    text = gen_text(st, wa)
                    count += 1
                    print(count, end='\r')

