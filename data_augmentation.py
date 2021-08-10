from PIL import Image, ImageOps

"""According to Wikipedia, HSL (for hue, saturation, lightness) and HSV
(for hue, saturation, value; also known as HSB, for hue, saturation, brightness) are 
alternative representations of the RGB color model"""

img = Image.open("./tulips.jpeg")
mirror_img = ImageOps.mirror(img)
mirror_img.save("./tulips_mirror.jpeg")

HSV= img.convert('HSV')

# Split into separate channels
H, S, V = HSV.split()
S = S.point(lambda p: p//2)
H = H.point(lambda p: p//3)

HSVr = Image.merge('HSV', (H,S,V))

RGBr = HSVr.convert('RGB')

RGBr.save('tulips_hsv.jpeg')
with Image.open("tulips.jpeg") as im:
    im.rotate(90).save("./tulips_rotated.jpeg")


with Image.open("tulips.jpeg") as im:
    

    width, height = im.size
    
    # Setting the points for cropped image
    left = 5
    top = height / 4
    right = 164
    bottom = 3 * height / 4
    
    # Cropped image of above dimension
    # (It will not change original image)
    im1 = im.crop((left, top, right, bottom))
    
    
    im1.save("./tulips_cropped.jpeg")
    #im1.show()
