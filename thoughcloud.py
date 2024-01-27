import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageChops
from wordcloud import WordCloud
import cv2
import time
import os

# Read text
subGenre = pd.read_csv(r'goodreads_Data.csv', encoding='ISO-8859-1', usecols=['Sub-Genre'], sep=',')
start_time = time.time()

# Convert DataFrame to string and make uppercase
a1 = pd.DataFrame(subGenre)
a3 = pd.DataFrame.to_string(a1)
a0 = a3.upper()

# Load the mask for the word cloud
mask_image_path = r'images\thoughts2.webp'
mask = np.array(Image.open(mask_image_path))

# Invert the mask: white becomes black (0) and black becomes white (255)
inverted_mask = 255 - mask[:, :, 0]  # Assuming the mask is in grayscale or RGB where all channels are the same

# Convert the mask to grayscale (necessary for edge detection)
mask_gray = inverted_mask

# Detect edges in the mask to get the border of the thought bubble
edges = cv2.Canny(mask_gray, threshold1=100, threshold2=200)

# Create the WordCloud object
wordcloud = WordCloud(prefer_horizontal=.7,
                      colormap='Blues',
                      min_font_size=5,
                      max_font_size=70,
                      background_color="rgba(255, 255, 255, 0)", mode="RGBA",
                      width=7680,
                      height=4320,
                      margin=2,
                      collocations=False,
                      mask=mask,
                      repeat=True,
                      relative_scaling=0,
                      scale=1,
                      min_word_length=3,
                      include_numbers=False,
                      normalize_plurals=False,
                      font_step=1).generate(a0)

# Convert edges to have three channels so it can be colored
edges_colored = np.stack((edges, edges, edges),axis=-1)

# Ensure the overlay is applied over the RGBA image by adding an alpha channel to edges
alpha_channel = np.ones(edges.shape, dtype=edges.dtype) * 255  # Fully opaque
edges_colored = np.concatenate((edges_colored, alpha_channel[..., None]), axis=-1)


# Overlay the edges on the word cloud image
wordcloud_image = wordcloud.to_image()
wordcloud_array = np.array(wordcloud.to_image().convert('RGBA'))
overlay = cv2.addWeighted(wordcloud_array, 1, edges_colored, 1, 0)

# Convert back to Image for display
overlay_image = Image.fromarray(overlay)

# Display the overlay image
plt.figure(figsize=(30,30))
plt.imshow(overlay_image, interpolation='bilinear')
plt.axis("on")  # You might want to set this to "off" after checking the alignment
plt.show()

# Display the word cloud
plt.figure(figsize=(30,30))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("on")
plt.show()

# Create and show the border image
border_image = ImageChops.difference(wordcloud.to_image().convert('RGB'), Image.new('RGB', wordcloud.to_image().size))
border_image.show()

print(' ')
print('Time elapsed: {:.2f}s'.format(time.time() - start_time))

# Save the generated word cloud
wordcloud.to_file('ImagessubGenre_wordcloud.png')
