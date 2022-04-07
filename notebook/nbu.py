from IPython.display import HTML, Image


def _src_from_data(data):
	"""Base64 encodes image bytes for inclusion in an HTML img element"""
	img_obj = Image(data = data)
	for bundle in img_obj._repr_mimebundle_():
		for mimetype, b64value in bundle.items():
			if mimetype.startswith('image/'):
				return f'data:{mimetype};base64,{b64value}'


def gallery(images, row_height = 'auto'):
	"""Shows a set of images in a gallery that flexes with the width of the notebook.

	Parameters
	----------
	images: list of str or bytes
		URLs or bytes of images to display

	row_height: str
		CSS height value to assign to all images. Set to 'auto' by default to show images
		with their native dimensions. Set to a value like '250px' to make all rows
		in the gallery equal height.
	"""
	figures = []
	for image in images:
		if isinstance(image, bytes):
			src = _src_from_data(image)
			caption = ''
		else:
			src = image
			caption = f'<figcaption style="font-size: 0.6em">{image}</figcaption>'
		figures.append(f'''
            <figure style="margin: 5px !important;">
              <img src="{src}" style="height: {row_height}">
              {caption}
            </figure>
        ''')
	return HTML(data = f'''
        <div style="display: flex; flex-flow: row wrap; text-align: center;">
        {''.join(figures)}
        </div>
    ''')