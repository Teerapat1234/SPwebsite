import datetime
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import sys
import praw
import scipy
import pandas
import sklearn
import json
import webbrowser
from PIL import Image
import requests
from io import BytesIO
from io import StringIO
import urllib.request
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from skimage import io
import seaborn as sns
from skimage.io import imread, imshow
from skimage.transform import resize
from skimage.util import img_as_ubyte
import glob
import argparse
import imutils
from skimage.metrics import structural_similarity as ssim
from datetime import datetime, timedelta


##---------------------------------------------------------------------------------------------------------
def BDcheck(url):
    if "gif" not in url:  # Check type of image
        mode_to_bpp = {'1': 1, 'L': 8, 'P': 8, 'RGB': 24, 'RGBA': 32, 'CMYK': 32, 'YCbCr': 24, 'I': 32, 'F': 32}
        response = requests.get(url)
        data = Image.open(BytesIO(response.content))
        bpp = mode_to_bpp[data.mode]
    else:  # if image has gif type bpp = 0
        bpp = 0
    return bpp
##---------------------------------------------------------------------------------------------------------
def storePost(formatted_posts, TopPost, subname):
    arr = []
    current_date = datetime.today().strftime('%Y-%m-%d-%H')
    current_date = str(current_date).split("-")
    fodder = current_date[2][:2]
    now_str = ""
    if int(current_date[3]) > 12:
        now_str = "-" + "12" + "-" + fodder + "-" + current_date[1] + "-" + current_date[0]
    else:
        now_str = "-" + "00" + "-" + fodder + "-" + current_date[1] + "-" + current_date[0]
    for i in formatted_posts:
        data = {"up": i.up, "down": i.down, "time": i.time, "url": i.url}     #Time that the post was made
        arr.append(data)
    log_file = subname + './post' + now_str + '.json'
    with open("C:/Users/psyon/Desktop/test/json/" + log_file, "w") as file:
        json.dump(arr, file, indent=4)

    arr = []
    for i in TopPost:
        data = {"post": i}
        arr.append(data)
    log_file = subname + './pop' + now_str + '.json'
    with open("C:/Users/psyon/Desktop/test/json/" + log_file, "w") as file:
        json.dump(arr, file, indent=4)
# --------------------------------------------------------------------------------------------------------

def compare(image, template):
    image = cv2.resize(image, (500, 500))
    template = cv2.resize(template, (250, 250))
    template2 = template.copy()
    template = cv2.Canny(template, 100, 150)
    (tH, tW) = template.shape[:2]
    found = None
    for scale in np.linspace(0.2, 1.0, 45)[::-1]:
        resized = imutils.resize(image, width=int(image.shape[1] * scale))
        r = image.shape[1] / float(resized.shape[1])
        if resized.shape[0] < tH or resized.shape[1] < tW:
            break
        edged = cv2.Canny(resized, 100, 150)  # The 2 numbers behind is the lower, upper threshold of our edge.
        result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
        (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
        if found is None or maxVal > found[0]:
            found = (maxVal, maxLoc, r)
    (_, maxLoc, r) = found
    (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
    (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))

    resizedTemplate = cv2.resize(template2, (endX - startX, endY - startY), interpolation=cv2.INTER_AREA)
    roi = image[startY:endY, startX:endX]
    image[startY:endY, startX:endX] = roi

    sim = ssim(resizedTemplate, roi)  # similarity is here instead
    return sim
##---------------------------------------------------------------------------------------------------------
def scrape():
    #############################################################################
    for run in range(3):
        class post:
            up, down, time, url, check = 0, 0, 0, "", True  # This will be used to test whether this particular post already have similar img to other posts.

        reddit = praw.Reddit(client_id="ByFeBxzm-VNXVw",
                             username="LukeKunG",
                             client_secret="uBuaT8KB5nH4RfxFzaZ-WeafRSE",
                             user_agent="Senior")
        subname = ""
        if run == 0:
            subname = "dankmemes"
        elif run == 1:
            subname = "memes"
        else:
            subname = "funny"
        sub = reddit.subreddit(subname)
        hot_posts = sub.new(limit=100)  # Change value here
        formatted_posts, URLarray = [], []
        for submission in hot_posts:
            x = post()
            ratio = submission.upvote_ratio
            x.up = int(round((ratio * submission.ups) / (2 * ratio - 1)) if ratio != 0.5 else round(submission.ups / 2))
            x.down = x.up - submission.ups           #I've changed this "score" to "up" here and above
            x.time = submission.created_utc
            submission_imgURL = str(submission.url.encode('utf-8'))
            ArrayToken = submission_imgURL.split("'")
            x.url = ArrayToken[1]
            URLarray.append(ArrayToken[1])
            formatted_posts.append(x)

        #############################################################################
        TopPost, TopPostContestant = [], []
        for i in range(len(formatted_posts)):
            if formatted_posts[i].check:            #Only run below if the POST is already checked to have similarity with other prior posts.
                formatted_posts[i].check = False    #It will never be compare again
                allow, bpp = True, 0
                try:
                    bpp = BDcheck(formatted_posts[i].url)  # bit depth of picture
                except:
                    allow = False
                if "gif" in formatted_posts[i].url or bpp != 24:
                    allow = False
                try:
                    template = imread(formatted_posts[i].url)
                except:
                    allow = False
                if allow:                      #Only proceed if the img use to compare doesn't violate the bit-thingy in the first place.
                    template = imread(formatted_posts[i].url)
                    for post in range(len(formatted_posts)):
                        if formatted_posts[post].check:  #The 2 comparing posts aren't the same post. Or the other post wasn't compared before.
                            addin, bpp = True, 0
                            try:
                                bpp = BDcheck(formatted_posts[post].url)  # bit depth of picture
                            except:
                                addin = False
                            if "gif" in formatted_posts[post].url or bpp != 24:
                                addin = False
                            if addin:
                                try:
                                    image = imread(formatted_posts[post].url)
                                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                                    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
                                except:
                                    break
                                sim = compare(image, template)
                                if sim >= 0.3:    #Change the value of similarity here
                                    formatted_posts[post].check = False
                                    if len(TopPost) == 0:   #the first instance of this similar posts
                                        TopPost.append(formatted_posts[post].url)
                                    else:
                                        TopPostContestant.append(formatted_posts[post].url)

                    if len(TopPostContestant) > len(TopPost):  #This means that there's an array that has more similar posts
                        TopPost = TopPostContestant

        #Now we will have a lists of post with certain similarity (TopPost) and the lists of overall post (formatted_post)
        print(len(formatted_posts))
        storePost(formatted_posts, TopPost, subname)

    return TopPost[0]

#########################################################################################################
template = scrape()
