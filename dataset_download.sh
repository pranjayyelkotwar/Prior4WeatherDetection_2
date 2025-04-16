#!/bin/bash

set -e  # Exit if any command fails

# Create target directory
mkdir -p dataset/cityscapes

# Step 1: Login and save cookies
echo "🔐 Logging in to Cityscapes..."
wget --keep-session-cookies --save-cookies=cookies.txt \
     --post-data 'username=PranjayYelkotwar&password=DL@Assignment123&submit=Login' \
     https://www.cityscapes-dataset.com/login/ -O /dev/null

# Step 2: Download URLs
URLS=(
  "https://www.cityscapes-dataset.com/file-handling/?packageID=29"
  "https://www.cityscapes-dataset.com/file-handling/?packageID=3"
  "https://www.cityscapes-dataset.com/file-handling/?packageID=1"
)

# Step 3: Download packages in parallel
echo "⬇️  Starting downloads..."
for url in "${URLS[@]}"; do
  wget --load-cookies cookies.txt --content-disposition "$url" &
done

# Step 4: Wait for all downloads
wait
echo "✅ All downloads complete."

# Step 5: Unzip all .zip files into dataset/cityscapes/
echo "📦 Extracting zip files..."
for zip in *.zip; do
  echo "📂 Extracting $zip"
  unzip -q "$zip" -d dataset/cityscapes/
done

# Step 6: Remove zip files to save space
echo "🧹 Cleaning up zip files..."
rm -f *.zip

echo "🎉 All done! Data is in dataset/cityscapes/"
