import folium
import random
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer, VectorAssembler

def cluster_stations(data, k=10, seed=1):
    num_stations = data.select('name').distinct().count()
    print(f'Clustering {num_stations} different stations')

    assembler = VectorAssembler(inputCols=['latitude', 'longitude'], outputCol='features')
    data_with_features = assembler.transform(data)

    kmeans = KMeans().setK(k).setSeed(seed)
    model = kmeans.fit(data_with_features)
    clusters = model.transform(data_with_features)

    return clusters.drop('features')


def map_stations(predictions, centers):
    Montreal = [45.508154, -73.587450]
    montreal_map = folium.Map(
        location = Montreal,
        zoom_start = 12,
        tiles = 'CartoDB positron'
    )

    r = lambda: random.randint(0, 255)
    colors = ['#%02X%02X%02X' % (r(),r(),r()) for c in centers]

    # Map cluster centroids
    for center, color in zip(centers, colors):
        folium.Circle(
            location=center,
            radius=10,
            color=color,
            fill=True
        ).add_to(montreal_map)

    montreal_map.save('centroids.html')

    stations = predictions.rdd.collect()
    for station in stations:
        folium.Circle(
            location=[station.latitude, station.longitude],
            radius=2,
            color=colors[station.prediction],
            fill=True
        ).add_to(montreal_map)

    montreal_map.save('clusters.html')
