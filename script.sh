magick peppers.tiff -resize 256x256! -colorspace Gray -depth 8 BMP3:peppers.bmp
convert peppers.bmp \
	-gaussian-blur 0.8 \
	-define convolve:scale='!' \
	-define morphology:compose=Lighten \
	-morphology Convolve  'Sobel:>' \
	-type Bilevel \
	BMP3:peppers-sobel.bmp

convert peppers-sobel.bmp -morphology Thinning:-1 Skeleton:1 -negate BMP3:peppers-sobel-thin.bmp
