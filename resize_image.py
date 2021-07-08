import argparse
import os
from PIL import Image
from tqdm.auto import tqdm

def resize_image(image, size):
    """Resize an image to the given size."""
    return image.resize(size, Image.ANTIALIAS)

def resize_images(image_dir, output_dir, size):
    """Resize the images in 'image_dir' and save into 'output_dir'."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    images = os.listdir(image_dir)
    num_images = len(images)

    for i, image in enumerate(tqdm(images, total=len(images))):
        with open(os.path.join(image_dir, image), 'r+b') as f: # open each image file
            with Image.open(f) as img: # open loaded file to Image module
                img = resize_image(img, size) # resize each image to given size
                img.save(os.path.join(output_dir, image), img.format)

def main(args):
    image_dir = args.image_dir
    output_dir = args.output_dir
    image_size = [args.image_size, args.image_size] # resized image will be square size
    resize_images(image_dir, output_dir, image_size)
    print("Saved resized images to '{}'".format(output_dir))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='./dataset/train2017/',
                        help='directory for train images')
    parser.add_argument('--output_dir', type=str, default='./dataset/resized_train2017/',
                        help='directory for saving resized images')
    parser.add_argument('--image_size', type=int, default=256,
                        help='size for image after processing')
    args = parser.parse_args()
    main(args)