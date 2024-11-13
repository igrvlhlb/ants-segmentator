#!/usr/bin/env sh

function dst_name() {
	echo "${1%.*}-$2.bmp"
}

out1=$(dst_name $1 grayscale)
out2=$(dst_name $1 sobel-thin)

magick $1 \
	-resize 256x256! \
	-colorspace Gray \
	-depth 8 \
	BMP3:$out1

magick $out1 -define convolve:scale='!' \
	-define morphology:compose=Lighten \
	-morphology Convolve  'Sobel:>' \
	-threshold 30% \
	-type Bilevel BMP3:- \
	| magick - -morphology Thinning:-1 Skeleton:1 \
	-negate \
	BMP3:$out2
