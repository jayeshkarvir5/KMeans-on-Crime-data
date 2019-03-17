from django.db import models

# Create your models here.


class Item(models.Model):
    State = models.CharField(max_length=50)
    District = models.CharField(max_length=50)
    Murder = models.IntegerField(default=0)
    Attempt_to_commit_Murder = models.IntegerField(default=0)
    Rape = models.IntegerField(default=0)
    Kidnapping_Abduction = models.IntegerField(default=0)
    Robbery = models.IntegerField(default=0)
    Burglary = models.IntegerField(default=0)
    Theft = models.IntegerField(default=0)
    Riots = models.IntegerField(default=0)
    Cheating = models.IntegerField(default=0)
    Hurt = models.IntegerField(default=0)
    Dowry_Deaths = models.IntegerField(default=0)
    Assault_on_Women = models.IntegerField(default=0)
    Sexual_Harassment = models.IntegerField(default=0)
    Stalking = models.IntegerField(default=0)
    Death_by_Negligence = models.IntegerField(default=0)
    Extortion = models.IntegerField(default=0)
    Incidence_of_Rash_Driving = models.IntegerField(default=0)
    Total = models.IntegerField(default=0)

    def __str__(self):
        return self.District + ' - ' + str(self.Total)
