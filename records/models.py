# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from datetime import datetime
from django.db import models

# Create your models here.
class Records(models.Model):
    id = models.CharField(max_length=100, primary_key=True)
    first_name = models.CharField(max_length=50, null=True)
    last_name = models.CharField(max_length=50, null=True)
    education = models.CharField(max_length=150,default="Baddi University")
    occupation = models.CharField(max_length=150, default="Student")
    image= models.ImageField(upload_to="static/img", default="static/img/default.jpg")
    bio = models.TextField()
    recorded_at = models.DateTimeField(default=datetime.now, blank=True)
    present = models.IntegerField( null=True)
    absent = models.IntegerField(null=True)
    total = models.IntegerField(null=True)


    def __str__(self):
        return self.first_name
    class Meta:
        verbose_name_plural = "Records"


#class Attendence(models.Model):
#      user_id = models.ForeignKey(Records)
#     present=models.AutoFields()
#     absent= models.AutoFields()
#     total= models.AutoFields()
