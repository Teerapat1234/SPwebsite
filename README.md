DESCRIPTION:<br/>
This project is one of the 2 main part project, called "Using SIR to predict and visualize the rise and fall of meme trends on Reddit".<br/>
This repositry only contains the code that takes formattedly named Json files, and use the data stroed in those files to calculate and visualize an SIR model, complete with other trivial explanations derived from such graph.

WORKING FOLDER:<br/>
The main primary working code (Python 3.8) of this project is located in SPwebsite/sp/calculate/views.py<br/>
While the main primary frontend part that will be used by the code (IE. html, csss) is located in SPwebsite/sp/templates/calculate/ (all the html files in this directory is used one way or the other.)<br/>
Lastly, the main resoureces that will be used by both ends (IE. images, user upload images) is located in SPwebsite/sp/static/

HOW TO RUN:<br/>
-This project incorporates the use of Django framework, thus you need to have it installed on your computer first. Fortunate enough, Django has an indept documentation that will take you through the installation easily.<br/>
-Once you have installed Django, simply open this project, using command line navigating yourself into SPwebsite/sp/ there should be a python file named "manage.py" there. <br/>
-In the same directory, open up "views.py", you will see the necessary imports that you have to manually make.<br/>
-You need to create a folder on your computer with 3 smaller sub-folders inside, named "dankmemes", "memes", and "funny". Then use the json files found in "json/dankmemes" folder that is included in the repo to fill into the dankmemes sub-folder.
-Once everything has finished, go back to the command line, type in "python manage.py runserver", this will open up post:8000 on your webbrowser. And you're all set!

DISCLAIMER:<br/>
-Upon running the site, the only date that you can get any data from is 27/04/2021 due to the lack of any other data from any other date (Remember, the Json files are saved locally, and I wouldn't want to upload all 100 of them into Github) and only from R/dankmemes.<br/>
-For the same reason as above, do not expect the images that should normally be shown on the site to work properly.

And the rest are just me having no fucking clue how django works.
