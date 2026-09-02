from django.db import models

class PredictionHistory(models.Model):

    age = models.IntegerField()
    gender = models.CharField(max_length=20)
    job_role = models.CharField(max_length=100)
    experience = models.IntegerField()

    work_hours = models.IntegerField()
    overtime = models.IntegerField()
    sleep = models.FloatField()

    work_life = models.IntegerField()
    satisfaction = models.IntegerField()
    manager = models.IntegerField()

    prediction = models.CharField(max_length=20)
    recommendation = models.TextField()

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.job_role} - {self.prediction}"
