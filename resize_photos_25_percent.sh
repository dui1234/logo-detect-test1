#!/usr/bin/env bash

set -euo pipefail

# Usage:
#   ./resize_photos_25_percent.sh /path/to/folder_name
#
# Example:
#   ./resize_photos_25_percent.sh ./folder_name
#
# Output:
#   ./folder_name_0.25
#
# Note:
#   "Reduce by 25%" means the new width and height are 75% of the originals.

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 /path/to/photo_folder"
    exit 1
fi

src_dir="${1%/}"

if [[ ! -d "$src_dir" ]]; then
    echo "Error: folder does not exist: $src_dir" >&2
    exit 1
fi

parent_dir="$(dirname "$src_dir")"
folder_name="$(basename "$src_dir")"
out_dir="$parent_dir/${folder_name}_0.25"

# Support both newer and older ImageMagick commands.
if command -v magick >/dev/null 2>&1; then
    image_cmd=(magick)
elif command -v convert >/dev/null 2>&1; then
    image_cmd=(convert)
else
    echo "Error: ImageMagick is not installed." >&2
    echo "Install it with: sudo apt update && sudo apt install imagemagick" >&2
    exit 1
fi

mkdir -p "$out_dir"

count=0

while IFS= read -r -d '' file; do
    filename="$(basename "$file")"
    echo "Resizing: $filename"

    "${image_cmd[@]}" "$file" -auto-orient -resize 75% "$out_dir/$filename"
    ((count += 1))
done < <(
    find "$src_dir" -maxdepth 1 -type f \
        \( -iname '*.jpg' -o \
           -iname '*.jpeg' -o \
           -iname '*.png' -o \
           -iname '*.webp' -o \
           -iname '*.bmp' -o \
           -iname '*.tif' -o \
           -iname '*.tiff' \) \
        -print0
)

if [[ $count -eq 0 ]]; then
    echo "No supported image files were found in: $src_dir"
else
    echo "Done. Resized $count image(s)."
    echo "Saved to: $out_dir"
fi
