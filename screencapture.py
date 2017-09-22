import numpy as np
import Quartz.CoreGraphics as CG
from PIL import Image

# Arguments:
# region - CGRect object detailing what part of the screen should be captured.
#		   Default captures the entire screen.
# Returns:
# image - RGBA PIL Image of the region of the screen specified as an argument.
def capture_screen(region=CG.CGRectInfinite):
	image = CG.CGWindowListCreateImage(
	   	region,
	   	CG.kCGWindowListOptionOnScreenOnly,
	   	CG.kCGNullWindowID,
	   	CG.kCGWindowImageDefault)

	width = CG.CGImageGetWidth(image)
	height = CG.CGImageGetHeight(image)
	bytes_per_row = CG.CGImageGetBytesPerRow(image)

	pixeldata = CG.CGDataProviderCopyData(CG.CGImageGetDataProvider(image))
	image = np.frombuffer(pixeldata, dtype=np.uint8)

	image = image.reshape((height, bytes_per_row//4, 4))
	image = image[:,:width]

	return Image.fromarray(image, 'RGBA')
