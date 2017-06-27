filename=$(basename "$1")
filename="${filename%.*}"
pdflatex "$filename.tex"
pdfcrop "$filename.pdf"
convert -density 900 "$filename-crop.pdf" "$filename.png"
rm *.aux *.pdf *.log
