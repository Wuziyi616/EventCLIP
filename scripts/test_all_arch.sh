#!/bin/bash

# run `test.py` on all CLIP arches
CMD=$1

for arch in 'RN50' 'RN101' 'RN50x4' 'RN50x16' 'RN50x64' 'ViT-B/32' 'ViT-B/16' 'ViT-L/14'
do
    if [ "$arch" = "RN50x64" ]; then
        bs=32
    else
        bs=64
    fi

    echo "Testing $arch"
    cmd="$CMD --arch $arch --bs $bs"
    echo $cmd
    eval $cmd
done
