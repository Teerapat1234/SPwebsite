from django.db import models

class Allcal(models.Model):
    name = models.CharField(max_length=200)
    insname = models.CharField(max_length=100)
    def __str__(self):
        return self.coursename

class details(models.Model):
    course = models.ForeignKey(Allcal, on_delete=models.CASCADE)
    x = models.CharField(max_length=500)
    i = models.CharField(max_length=500)
    def __str__(self):
        return self.x

class Image(models.Model):
    title = models.CharField(max_length=200)
    # image = models.ImageField(upload_to='users/%Y/%m/%d/', blank=True)  Real time display option
    image = models.ImageField(upload_to="storeimg")

    def __str__(self):
        return self.image

# Create your models here.
