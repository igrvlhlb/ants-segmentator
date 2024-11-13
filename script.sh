function dst_name() {
	echo "${1%.*}-sobel-thin.bmp"
}

magick peppers.tiff \
	-resize 256x256! \
	-colorspace Gray \
	-depth 8 BMP3:- \
	| magick - -define convolve:scale='!' \
	-define morphology:compose=Lighten \
	-morphology Convolve  'Sobel:>' \
	-threshold 30% \
	-type Bilevel BMP3:- \
	| magick - -morphology Thinning:-1 Skeleton:1 \
	-negate \
	BMP3:$(dst_name $1)
