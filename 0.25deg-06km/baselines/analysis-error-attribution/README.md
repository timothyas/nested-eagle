```
pandoc gfs_vs_hrrr_writeup.md \
    -o gfs_vs_hrrr_writeup.pdf \
    --pdf-engine=tectonic -V geometry:margin=1in -V colorlinks=true \
    -V mainfont="Liberation Serif" -V sansfont="Liberation Sans" -V monofont="Liberation Mono"
```
