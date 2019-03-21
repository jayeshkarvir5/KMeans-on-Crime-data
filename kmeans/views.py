from django.http import HttpResponse
from django.shortcuts import render
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from numpy import random, float, array
import numpy as np
import seaborn as sns
from .models import Item
from matplotlib import pylab
from pylab import *
from io import BytesIO
import base64


def index(request):
    df = pd.read_csv("D:/Projects/DM_miniproject/kmeans/data_for_mini-project.csv")
    # entries = []
    # for obj in df.T.to_dict().values():
    #     entries.append(Item(**obj))
    # Item.objects.bulk_create(entries)
    context = {
        'df': df.loc[0:10].to_html(),
        # 'x': X.to_html(),
        # 'qs': Item.objects.all(),
        # 'graphic': graphic,
    }
    return render(request, 'kmeans/index.html', context)


def selected_option(request):
    state = request.POST.get('option')
    qs = Item.objects.filter(State__startswith=state)
    df = pd.read_csv("D:/Projects/DM_miniproject/kmeans/data_for_mini-project.csv")
    df = df[(df.State == state) & (df.District != "Total")]
    X = df[['Murder', 'Attempt_to_commit_Murder', 'Rape', 'Kidnapping_Abduction', 'Robbery', 'Burglary',
            'Theft', 'Riots', 'Cheating', 'Hurt', 'Dowry_Deaths', 'Assault_on_Women', 'Sexual_Harassment',
            'Stalking', 'Death_by_Negligence', 'Extortion', 'Incidence_of_Rash_Driving']]
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    cluster_range = range(1, 20)  # take care of the size for different states
    cluster_errors = []

    for num_clusters in cluster_range:
        clusters = KMeans(num_clusters)
        clusters.fit(X_scaled)
        cluster_errors.append(clusters.inertia_)

    clusters_df = pd.DataFrame({"num_clusters": cluster_range, "cluster_errors": cluster_errors})
    plt.figure(figsize=(10, 4))
    plt.plot(clusters_df.num_clusters, clusters_df.cluster_errors, marker="o")
    # Store image in a string buffer
    plt.tight_layout()
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    graphic = base64.b64encode(image_png)
    graphic = graphic.decode('utf-8')
    pylab.close()
    context = {
        'df': df.to_html(),
        'qs': qs,
        'graphic': graphic,
    }
    return render(request, 'kmeans/selected_option.html', context)


def clustering(request):
    size = request.POST.get('k')
    context = {
        'k': size,
    }
    return render(request, 'kmeans/cluster_formation.html', context)

# x = arange(0, 2*pi, 0.01)
# s = cos(x)**2
# plot(x, s)
# xlabel('xlabel(X)')
# ylabel('ylabel(Y)')
# title('Simple Graph!')
# grid(True)
