# Generated by Django 2.1.7 on 2019-03-17 07:44

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='dataset',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('State', models.CharField(max_length=50)),
                ('District', models.CharField(max_length=50)),
                ('Murder', models.IntegerField(default=0)),
                ('Attempt_to_commit_Murder', models.IntegerField(default=0)),
                ('Rape', models.IntegerField(default=0)),
                ('Kidnapping_Abduction', models.IntegerField(default=0)),
                ('Robbery', models.IntegerField(default=0)),
                ('Burglary', models.IntegerField(default=0)),
                ('Theft', models.IntegerField(default=0)),
                ('Riots', models.IntegerField(default=0)),
                ('Cheating', models.IntegerField(default=0)),
                ('Hurt', models.IntegerField(default=0)),
                ('Dowry_Deaths', models.IntegerField(default=0)),
                ('Assault_on_Women', models.IntegerField(default=0)),
                ('Sexual_Harassment', models.IntegerField(default=0)),
                ('Stalking', models.IntegerField(default=0)),
                ('Death_by_Negligence', models.IntegerField(default=0)),
                ('Extortion', models.IntegerField(default=0)),
                ('Incidence_of_Rash_Driving', models.IntegerField(default=0)),
                ('Total', models.IntegerField(default=0)),
            ],
        ),
    ]