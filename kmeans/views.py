from __future__ import print_function
from django.http import HttpResponse
from django.shortcuts import render
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale, StandardScaler
from numpy import random, float, array
import numpy as np
import seaborn as sns
from .models import Item
from matplotlib import pylab
from pylab import *
from io import BytesIO
import base64


def index(request):
    context = {
        'title': "Index",
        # 'df': df.loc[0:10].to_html(),
        # 'x': X.to_html(),
        # 'qs': Item.objects.all(),
        # 'graphic': graphic,
    }
    return render(request, 'kmeans/index.html', context)


def selected_option(request):
    state = request.POST.get('option')
    request.session['state'] = state
    qs = Item.objects.filter(State__startswith=state)
    df = pd.read_csv("D:/Projects/DM_miniproject/kmeans/data_for_mini-project.csv")
    df = df[(df.State == state) & (df.District != "Total")]
    # X = df[['Murder', 'Attempt_to_commit_Murder', 'Rape', 'Kidnapping_Abduction', 'Robbery', 'Burglary',
    #         'Theft', 'Riots', 'Cheating', 'Hurt', 'Dowry_Deaths', 'Assault_on_Women', 'Sexual_Harassment',
    #         'Stalking', 'Death_by_Negligence', 'Extortion', 'Incidence_of_Rash_Driving']]
    X = df[['Murder', 'Rape', 'Kidnapping_Abduction', 'Burglary']]
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
        'title': "Statistics for State",
        'df': df.to_html(classes=["table-bordered", "table-striped", "table-hover", "text-center"]),
        'qs': qs,
        'graphic': graphic,
    }
    return render(request, 'kmeans/selected_option.html', context)


def clustering(request):
    size = request.POST.get('k')
    state = request.session['state']
    request.session['size'] = size
    df = pd.read_csv("D:/Projects/DM_miniproject/kmeans/data_for_mini-project.csv")
    df = df[['State', 'District', 'Murder', 'Rape', 'Kidnapping_Abduction', 'Burglary', 'Total']]
    qs = df[(df.District == "Total")]
    df = df[(df.State == state) & (df.District != "Total")]
    # X = df[['Murder', 'Attempt_to_commit_Murder', 'Rape', 'Kidnapping_Abduction', 'Robbery', 'Burglary',
    #         'Theft', 'Riots', 'Cheating', 'Hurt', 'Dowry_Deaths', 'Assault_on_Women', 'Sexual_Harassment',
    #         'Stalking', 'Death_by_Negligence', 'Extortion', 'Incidence_of_Rash_Driving']]
    X = df[['Murder', 'Rape', 'Kidnapping_Abduction', 'Burglary']]

    clusters = KMeans(int(size))
    clusters.fit(X)
    df['Crime_clusters'] = clusters.labels_
    centers = np.array(clusters.cluster_centers_)
    plt.figure(figsize=(7, 5))
    plt.scatter(centers[:, 0], centers[:, 1], marker="x", color='r')
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=[
                plt.cm.get_cmap("Spectral")(float(i) / (int(size)+1)) for i in clusters.labels_])
    plt.tight_layout()
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    graphic = base64.b64encode(image_png)
    graphic = graphic.decode('utf-8')
    pylab.close()

    qs = qs[['State', 'Total']]
    arr_state = np.array(qs.iloc[:, 0])
    index = np.arange(36)
    arr_total = np.array(qs.iloc[:, 1])
    plt.figure(figsize=(13, 5))
    plt.bar(index, arr_total)
    plt.xlabel('State', fontsize=10)
    plt.ylabel('Total No of crimes', fontsize=10)
    plt.xticks(index, arr_state, fontsize=7, rotation=30)
    plt.title('Total Crime Vs State')
    plt.tight_layout()
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    graphic1 = base64.b64encode(image_png)
    graphic1 = graphic1.decode('utf-8')
    pylab.close()
    df = df.sort_values(by=['Crime_clusters'], ascending=True)
    context = {
        'title': "Applied K-Means",
        'k': size,
        'df': df.to_html(classes=["table-bordered", "table-striped", "table-hover", "text-center"]),
        'qs': qs.to_html(classes=["table-bordered", "table-striped", "table-hover", "text-center"]),
        'graphic': graphic,
        'graphic1': graphic1,
    }
    return render(request, 'kmeans/cluster_formation.html', context)

# x = arange(0, 2*pi, 0.01)
# s = cos(x)**2
# plot(x, s)
# xlabel('xlabel(X)')
# ylabel('ylabel(Y)')
# title('Simple Graph!')
# grid(True)


# Populating database
# df = pd.read_csv("D:/Projects/DM_miniproject/kmeans/data_for_mini-project.csv")
# entries = []
# for obj in df.T.to_dict().values():
#     entries.append(Item(**obj))
# Item.objects.bulk_create(entries)
