from PIL import Image
from PIL.ExifTags import TAGS
import os


# path to the image or video
# imagename = "./mchar_train/004863.png"

def showImageInfo(imagename):
	# read the image data using PIL
	image = Image.open(imagename)

	# extract other basic metadata
	info_dict = {
		"Filename"         : image.filename,
		"Image Size"       : image.size,
		"Image Height"     : image.height,
		"Image Width"      : image.width,
		"Image Format"     : image.format,
		"Image Mode"       : image.mode,
		"Image is Animated": getattr(image, "is_animated", False),
		"Frames in Image"  : getattr(image, "n_frames", 1)
	}

	for label, value in info_dict.items():
		print(f"{label:25}: {value}")

	print(os.stat(imagename))
	print(os.path.getsize(imagename))
	print(os.path.getmtime(imagename))
	print(os.path.getctime(imagename))