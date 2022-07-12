import csv
import random
import pandas as pd
from fastai.text import *
from fastai.vision import *
import PIL
from PIL import Image
import cv2
from tqdm import tqdm
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf

from hydra import compose, initialize
from omegaconf import OmegaConf
hydra.core.global_hydra.GlobalHydra.instance().clear()

initialize(config_path="conf", job_name="test_app")
cfg = compose(config_name="config")
 
 
src_path=cfg.paths.src_path
synth_path=cfg.paths.synth_path
generated_data=cfg.paths.initial_data
edited_lines = pd.read_csv(generated_data+'train.csv',delimiter='\t\t')

def standardize_imgs(imgs, baseheight):
    resized_imgs = []
    for img in imgs:
        hpercent = (baseheight / float(img.size[1]))
        wsize = int((float(img.size[0]) * float(hpercent)))
        img = img.resize((wsize, baseheight), PIL.Image.ANTIALIAS)
        resized_imgs.append(img)
    return resized_imgs


def resize_max(im, size=1000):
    "Resize an image so that the largest dimension is of specified size"
    r,c = im.size
    ratio = size/max(r,c)
    return im.resize((int(r*ratio), int(c*ratio)), PIL.Image.ANTIALIAS)
def show_sample(df, path, row=2, col=2, show_files=False):
    fig, axes = plt.subplots(row,col, figsize=(20, 20))
    samp = df.sample(row*col).values #=> outputs as an array [[filename, labels]]
    for i,ax in enumerate(axes.flat):
        row = samp[i]
        ax.imshow(PIL.Image.open(path/row[0]))
        title = row[1]+f"\n{row[2]}" if show_files else row[1]
        ax.set_title(title)

#     plt.tight_layout(pad=0.2)
def vconcat_resize(img_list, interpolation  
                   = cv2.INTER_CUBIC): 
      # take minimum width 
    w_min = min(img.shape[1]  
                for img in img_list) 
      
    # resizing images 
    im_list_resize = [cv2.resize(img, 
                      (w_min, int(img.shape[0] * w_min / img.shape[1])), 
                                 interpolation = interpolation) 
                      for img in img_list] 
    # return final image 
    return cv2.vconcat(im_list_resize) 
def create_img(imgs, targ_path, num_lines, max_size=None, pad=50):
    w = 1
    h = num_lines
        
    widths, heights = zip(*(i.size for i in imgs))
    median_height = int(np.median(heights))
    
    stzd_imgs = standardize_imgs(imgs, median_height)
    #imgs=stzd_imgs
    lines = [imgs[i:i + w] for i in range(0, len(imgs), w)]
    
    total_width = max([np.sum([word.size[0] for word in line]) for line in lines]) + (pad*(w+1))   
    total_height = (median_height * h) + (pad*(h+1)) #sum(heights)

    new_im = PIL.Image.new('RGB', (total_width, total_height), color=(255,255,255))

    y_offset = pad
    x_offset = pad
    
    for line in lines:
        x_offset = pad
        for word in line:
            new_im.paste(word, (x_offset,y_offset))
            x_offset += word.size[0] + pad
        y_offset += median_height + pad
    
    if max_size: 
        resize_max(new_im, max_size).save(targ_path)
    else:
        new_im.save(targ_path)
# number of words/image
def create_synth_data(df, num, num_lines, src_path, targ_path, max_size=None, offset=0):
    d={}
    for i in tqdm(range(num)):
        samp = df.sample(num_lines)
        files = samp.filename.values
        print(files)
        imgs  = [PIL.Image.open(src_path+f) for f in files]
         
        # split into rows with \n
        label = '\n'.join([' '.join(row) for row in np.array_split(samp.text.values, num_lines)])
#         label = ' '.join(samp.text.values)
        
        fname = 'train_'+str(num_lines)+'_'+'{:04d}'.format(i+offset)+'.png'
        create_img(imgs, targ_path+fname, num_lines, max_size)
        [f.close() for f in imgs]
        d[fname] = label
    return pd.DataFrame({'filename': list(d.keys()), 'label': list(d.values())}),d

 
 
for num_lines in tqdm(range(1,9)):
    synth,d = create_synth_data(edited_lines, 10000 , num_lines, src_path, synth_path)  
    synth=synth.replace(r'\n','|',regex=True) 
    CSV = str(synth_path)+'_'+str(num_lines)+'.csv'

    #synth.to_csv(CSV, columns=['filename', 'label'], index=False)
    with open(f"{generated_data}gt_blocs_from_lines_iam_8_lines_train.csv", 'a+') as csv_file:  
        writer = csv.writer(csv_file)
        for key, value in d.items():
            value=value.replace('\n',' | ')
            csv_file.write(key+',128+'+' value'+'\n')