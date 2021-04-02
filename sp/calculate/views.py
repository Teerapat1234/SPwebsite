# //////////////////////////////////
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.template import loader
from django.db import models
from django.shortcuts import render
from .forms import ImageForm
import json
import time
import math
import imutils
import requests
import itertools
import numpy as np
import praw
import matplotlib.pyplot as plt
import cv2
import bokeh
from PIL import Image
import io
import os
import urllib, base64
from io import BytesIO, StringIO
from skimage.io import imread, imshow
from skimage.transform import resize
from skimage.util import img_as_ubyte
from skimage.metrics import structural_similarity as ssim
from datetime import datetime, timedelta, date
from .forms import ImageForm
from pathlib import Path
from scipy.integrate import odeint
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import json

##---------------------------------------------------------------------------------------------------------
from .models import Carousel


def compare_img(imgA, imgB):
    s = ssim(imgA, imgB)  # comparison with structural similarity index
    # y.TPic = imgA  Uncomment these if you want to see the comparison
    # y.Memeroi = imgB  Uncomment these if you want to see the comparison
    return s

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
BASE_DIR = Path(__file__).resolve().parent.parent

##---------------------------------------------------------------------------------------------------------
def Regression(postNumPerDay):
    i = 2  # Start at the position where the number of verified-posts/day are stored
    x = 0  # dimension x
    n = len(postNumPerDay) / 3
    SUMxy, SUMx, SUMx2, SUMy = 0, (n + 1) * n / 2, 0, 0
    try:
        while i < len(postNumPerDay):
            y = postNumPerDay[i]
            x += 1
            SUMy, SUMx2, SUMxy = SUMy + y, SUMx2 + (x * x), SUMxy + x * y
            i = i + 3
        b = (n * SUMxy - (SUMx * SUMy)) / (n * SUMx2 - (SUMx * SUMx))
    except:
        b = 1.0
    print("b", b)
    slope = b * (n - 1) / n  # a+b*n - a+b  || thus, we have no use for a
    print("slope", slope)
    return slope  # P

##---------------------------------------------------------------------------------------------------------
def CalInfectRecoveryRate(UpDown, PostNumPerDays):  # PostNumPerDays = [day1, 100, num of posts]
    x = 0  # Infectionrate
    y = 0  # Recoveryrate
    i, n = 2, len(PostNumPerDays) / 3
    print("post per num", PostNumPerDays)
    while i < len(PostNumPerDays):  # Calculate infection rate using the combined number similar posts over 100 over n days
        x = x + int(PostNumPerDays[i]) / int(PostNumPerDays[i - 1])  # Number of meme posts over all posts
        i = i + 3

    x, i, n = x/n, 0, 0
    while i < len(UpDown):  # Calculate recovery rate using the combined number of downvote percentage over similar posts
        ########################
        # divide = UpDown[i] + UpDown[i+1]
        # if divide == 0:
        #     divide = 1
        # x = x + UpDown[i]/divide
        # y = y + UpDown[i+1]/divide
        #######################
        # print("The types", type(UpDown[i]), type(UpDown[i+1]))
        if int(UpDown[i + 1]) > int(UpDown[i]) / 4:  # If downvote is higher than quarter percentile of upvote
            n += 1
            if int(UpDown[i]) == 0:
                UpDown[i] = 1
            y = y + int(UpDown[i + 1]) / int(UpDown[i])
        i += 2

    if n == 0:  # Thre's no instance of posts with less than 1/4 downvote rate.
        y = 0.1
    else:
        y = y / n
    print("x, y", x, y)
    return x, y

##---------------------------------------------------------------------------------------------------------
def Calsir(postNumPerDay):
    I0 = postNumPerDay[len(postNumPerDay) - 1]
    S0 = postNumPerDay[len(postNumPerDay) - 2] - I0
    R0 = 0
    i, MaxNum = 1, 0
    while i < len(postNumPerDay):
        if postNumPerDay[i] > MaxNum:
            MaxNum = postNumPerDay[i]
        else:
            R0 = R0 + (MaxNum - postNumPerDay[i])
        i += 3
    n = S0 + I0 + R0
    return n, S0, I0, R0

##---------------------------------------------------------------------------------------------------------
def deriv(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

##---------------------------------------------------------------------------------------------------------
def informative(I, input_days):
    maxI, pos, once, plus, day = 0, [0, -1], True, (120 / input_days) - 1, 1  # [pos_maxI, pos_start of I less than 0.2]
    for i in range(len(I)):
        if i % plus == 0 and i != 0:
            day += 1
        if maxI < I[i]:  # Finding the point where the maximum infected cases is presented
            pos[0], maxI = day / 2, I[i]
        if I[i] < 0.2 and once:  # might need to change this number at some point
            pos[1], once = day, False
    # i = plus
    # day = 1
    # while i <= len(I):
    #     if pos[0] <= i:
    #         pos[0] = day
    #     if pos[1] <= i:
    #         pos[1] = day
    #     i = i + plus
    #     day += 1
    return pos

##---------------------------------------------------------------------------------------------------------
def plot_data(postNumPerDay):
    t, num, i = [], [], 0
    while i < len(postNumPerDay):
        t.append(postNumPerDay[i])
        num.append(postNumPerDay[i + 2])
        i += 3
    fig = plt.figure(facecolor='w')  # Plot the data on three separate curves for S(t), I(t) and R(t)
    ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
    ax.axis([0.5, 0.5 * (len(t) - 1), 0, 100])
    ax.plot(t, num)
    ax.set_xlabel('days')
    ax.set_ylabel('posts with detected meme format')
    # ax.yaxis.set_tick_params(length=0)
    # ax.xaxis.set_tick_params(length=0)
    ax.grid(b=True, which='major', c='w', lw=5, ls='-')
    img_path = os.path.join(BASE_DIR, "static/data.jpg")
    plt.savefig(img_path)

##---------------------------------------------------------------------------------------------------------
def readjson(year, month, day, date_back):
    date_list, i = [], 0  # Store all dates leading to the current date   ['-day-month-year', '-da.....]
    while i < date_back:  # Including the first day. so if starts from 1 scrape for 2 days then [1, 2]
        # current_date = datetime.today() - timedelta(days=date_back)
        if i == 0:
            current_date = datetime(year, month, day, 00, 00, 00)
        else:
            current_date = datetime(year, month, day, 00, 00, 00) + timedelta(days=i)
        current_date = str(current_date).split("-")
        fodder = current_date[2][:2]
        start_date = "-" + "00" + "-" + fodder + "-" + current_date[1] + "-" + current_date[0] + ".json"
        date_list.append(start_date)
        start_date = "-" + "12" + "-" + fodder + "-" + current_date[1] + "-" + current_date[0] + ".json"
        date_list.append(start_date)
        i += 1
    print("Here's the date list", date_list)

    # #############################################################################
    class Rpost:
        url, MostSim, up, down = "", 0.0, 0, 0
    x = Rpost()
    # #############################################################################
    Posts = []
    for i in date_list:
        with open('/home/oem/Desktop/crontabProj/json/post' + i) as file:
            data = json.load(file)
        count = 0
        for q in data:
            count += 1
            arr = str(q).split(",")
            x.up, x.down = arr[0][7:], arr[1][9:]
            if count < len(data):
                x.url = arr[9:-2]
            else:
                x.url = arr[9:-3]
            Posts.append(x)  # 100 posts per json(smaller loop) upto date_back number of days(bigger loop)

    return Posts
##---------------------------------------------------------------------------------------------------------
def scale(Posts, original, percentage):     #Scale the image down and calculates the similarity between them
    template = original
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    template = cv2.resize(template, (250, 250))
    template2 = template.copy()
    template = cv2.Canny(template, 100, 150)
    (tH, tW) = template.shape[:2]

    count_day, day_count, results, num, postNumPerDay, UpDown = 0, 0, [], 0, [], []  # For every 100 we cut to the next day
    for post in Posts:
        count_day += 1
        addin, bpp = True, 0
        try:
            bpp = BDcheck(post.url)  # bit depth of picture
        except:
            addin = False
        if "gif" in post.url:
            addin = False
        if bpp != 24:  # check Is it 24?
            addin = False
        if addin:
            # template = original
            image = imread(post.url)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (500, 500))
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
            post.MostSim = compare_img(resizedTemplate, roi)

        if addin and post.MostSim > percentage:
            print("Most sim and percentage", post.MostSim, percentage)
            results.append(post)
            UpDown.append(post.up)
            UpDown.append(post.down)

        if count_day == 100:
            postNumPerDay.append(day_count)
            postNumPerDay.append(100)
            postNumPerDay.append(len(results) - num)
            num = len(results)  # value of num is always behind x by a day
            day_count += 0.5
            count_day = 0  # set number of days so it'll reach 100 again

    return results, postNumPerDay, UpDown
##---------------------------------------------------------------------------------------------------------
def mathmatical(UpDown, postNumPerDay):
    beta, gamma = CalInfectRecoveryRate(UpDown, postNumPerDay)
    N, S0, I0, R0 = Calsir(postNumPerDay)
    t = np.linspace(0, 7, 160)  # int(len(postNumPerDay) / 3)
    y0 = [S0, I0, R0]  # Initial conditions vector
    ret = odeint(deriv, y0, t, args=(N, beta, gamma))  # Integrate the SIR equations over the time grid, t.
    S, I, R = ret.T

    fig = plt.figure(facecolor='w')  # Plot the data on three separate curves for S(t), I(t) and R(t)
    ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
    ax.plot(t, S / N, 'b', alpha=0.5, lw=2, label='regular users')
    ax.plot(t, I / N, 'r', alpha=0.5, lw=2, label='meme creators')
    ax.plot(t, R / N, 'g', alpha=0.5, lw=2, label='bored meme creators')
    ax.set_xlabel('Time /days')
    ax.set_ylabel('Number (1000s)')
    ax.set_ylim(0, 1.0)
    ax.set_xlim(0, 7)  #
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    ax.grid(b=True, which='major', c='w', lw=2, ls='-')
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
        ax.spines[spine].set_visible(False)
    img_path = os.path.join(BASE_DIR, "static/graph.jpg")
    plt.savefig(img_path, bbox_inches="tight")

    return S, I, R, beta, gamma
##---------------------------------------------------------------------------------------------------------
def popular_prediction(request):
    percentage, date_back, current_date = 0.1, 7, str(datetime.today())
    current_date = current_date.split("-")
    current_date[2] = int(current_date[2][:2])
    current = "pop-00-" + current_date[0] + "-" + current_date[1] + "-" + current_date[2] + ".json"
    with open('/Users/psyon/Desktop/crontabProj/json/' + current) as file:  # Open the first link in the pop-.... to set as the template
        data = json.load(file)
    data = json.dumps(data)
    data = data.split("{")
    data.pop(0)  # removes the first array box that only contains the "["
    url = data[0][9:-3]
    original = cv2.imread(url)  #########################not sure if this works or not ususally it's just imread()
    #############################################################################
    Posts = readjson(current_date[0], current_date[1], current_date[2], date_back)  # read the database and retrieve the stored info
    results, postNumPerDay, UpDown = scale(Posts, original, percentage)  # The picture comparison part
    plot_data(postNumPerDay)  # Set up the scale of the graph
    S, I, R, beta, gamma = mathmatical(UpDown, postNumPerDay)  # The graph part
    pos = informative(I, date_back)
    date_back = current_date[2] + date_back
    return render(request, 'output.html',
                  {"results": results, "year": current_date[0], "month": current_date[1], "day": current_date[2], "days": date_back, "sim": percentage,
                   "maxI": pos[0], "trend_over": pos[1]
                      , "slope": beta, "recov": gamma})
##---------------------------------------------------------------------------------------------------------
def scrape(request):
    #############################################################################
    date = str(request.POST.get('input_date'))
    dateArr = date.split("-")
    uploaded_img = Image.open(request.FILES['image'])
    img_path = os.path.join(BASE_DIR, "storeimg/1.jpg")
    if uploaded_img.mode in ("RGBA", "P"):
        uploaded_img = uploaded_img.convert("RGB")
    try:
        date_back = int(request.POST.get('input_days'))
        year, month, day = int(dateArr[0]), int(dateArr[1]), int(dateArr[2])
        percentage = request.POST.get('percent')
        percentage = float(percentage) / 100.0
        uploaded_img.save(img_path)
    except:
        return render(request, 'error.html')
    ############################################################################
    original = cv2.imread(img_path)  # This value is the img used to compare
    # ///////////////////////////////////////////////////////////////////////////////////////////////////////
    Posts = readjson(year, month, day, date_back)  # read the database and retrieve the stored info
    results, postNumPerDay, UpDown = scale(Posts, original, percentage)  # The picture comparison part
    # ///////////////////////////////////////////////////////////////////////////////////////////////////////

    # ///////////////////////////////////////////////////////////////////////////////////////////////////////
    plot_data(postNumPerDay)  # Set up the scale of the graph
    S, I, R, beta, gamma = mathmatical(UpDown, postNumPerDay)  # The graph part
    # ///////////////////////////////////////////////////////////////////////////////////////////////////////

    pos = informative(I, date_back)
    date_back = day + date_back
    return render(request, 'output.html',
                  {"results": results, "year": year, "month": month, "day": day, "days": date_back, "sim": percentage,
                   "maxI": pos[0], "trend_over": pos[1]
                      , "slope": beta, "recov": gamma})
##---------------------------------------------------------------------------------------------------------
def index(request):
    form = ImageForm()
    return render(request, 'Home.html', {'form': form})
    # return render(request, 'index.html', {'form': form})
    # return render(request, "index.html")
def home(request):
    form = ImageForm()
    return render(request, 'Home.html', {'form': form})
def today(request):
    form = ImageForm()
    return render(request, 'TodayMost.html', {'form': form})
def analysis(request):
    form = ImageForm()
    return render(request, 'index.html', {'form': form})
def HomeView(request):
    obj = Carousel.objects.all()
    context = {
        'obj': obj
    }
    return render(request, 'Home.html', context)
##---------------------------------------------------------------------------------------------------------
def submitquery(request):
    q = request.POST.get['query', False]
    try:
        ans = eval(q)
        mydict = {
            "q": q,
            "ans": ans,
            "error": False
        }
        return render(request, 'index.html', context=mydict)
    except:
        pass
##---------------------------------------------------------------------------------------------------------
def test(request):
    date = str(request.POST.get('idate'))
    days = request.POST.get('idays')
    option = request.POST.get('option')
    per = request.POST.get('per')
    per = float(per) / 100.0
    if option == "on":
        option = "lul"
    else:
        option = "no"
    dateArr = date.split("-")
    year = int(dateArr[0])
    month = int(dateArr[1])
    day = int(dateArr[2])
    days = int(days) + day
    # 2017, 6, 29, 00, 00, 00
    return render(request, 'output.html', {"results": per, "year": year, "month": month, "day": day, "days": days})
##---------------------------------------------------------------------------------------------------------
def image_upload_view(request):  # process img uploaded by users
    if request.method == 'POST':
        uploaded_img = Image.open(request.FILES['image'])
        img_path = os.path.join(BASE_DIR, "storeimg/1.jpg")
        uploaded_img.save(img_path)
        original = cv2.imread(img_path)  # This value is the img used to compare

        return render(request, 'index.html', {'img_obj': img_path})
    else:  # essentially saying request.method==GET, meaning that we're request seeign the img
        form = ImageForm()
    return render(request, 'index.html', {'form': form})


##---------------------------------------------------------------------------------------------------------

def try_this(request):
    return render(request, 'output.html')
