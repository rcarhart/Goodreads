import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from stop_words import get_stop_words
import time
from io import BytesIO
from wordcloud import WordCloud, ImageColorGenerator, get_single_color_func
import requests
import io


csv_file_path = r'Data/subGenre_coordinates.csv'
directory = os.path.dirname(csv_file_path)
png_file_path = r'Images/subGenre_wordcloud.png'

#read text
subGenre = pd.read_csv(r'goodreads_Data.csv', encoding='ISO-8859-1', usecols=['Sub-Genre'], sep=',')
start_time = time.time()

response = requests.get("https://raw.githubusercontent.com/R-CoderDotCom/samples/main/wordcloud-mask.jpg")
mask = np.array(Image.open(BytesIO(response.content)))
a1 = pd.DataFrame(subGenre)
a3 = pd.DataFrame.to_string(a1)
a0 = a3.upper()

wordcloudImage = np.array(Image.open(r'images\heart2.webp'))
WordCloud = WordCloud(prefer_horizontal=.7,
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
print(WordCloud.layout_)

df = pd.DataFrame(WordCloud.layout_, columns=['Name','Size','Cord','Direction','Color'])
df.to_csv(csv_file_path)
WordCloud.to_file(png_file_path)

print(' ')
print ('time elapsed: {:.2f}s'.format(time.time() - start_time))

# plt.figure(figsize=(30,30))
plt.imshow(WordCloud,interpolation='bilinear')
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()

