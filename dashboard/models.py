from django.db import models

# Create your models here.

from django.db import models
from django.contrib.auth.models import User

class Wishlist(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    brand = models.CharField(max_length=100)
    price = models.FloatField()
    year = models.IntegerField()

    def __str__(self):
        return f"{self.brand} - ${self.price} - {self.year}"
