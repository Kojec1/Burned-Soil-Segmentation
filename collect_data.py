import json
import ee
import config

# Initiate Earth Engine
ee.Initialize()

# Load coordinates from a JSON file
with open(config.COORDINATES_PATH) as f:
    coords = json.load(f)

# Loop over a set of coordinates
for coord in coords:
    # Extract the region name
    region = coord['region']

    # Extract the point coordinates
    point = coord['point']
    poi = ee.Geometry.Point(point)

    # Extract the region of interest coordinates
    roi = coord['coordinates']
    roi = ee.Geometry.Polygon(roi)

    # Filter the Sentinel-2 images by region and cloud coverage
    collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(poi).filter(
        ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10)).sort('CLOUDY_PIXEL_PERCENTAGE')

    # Get the collection of images before the event
    collection_pre = collection.filterDate('2019-03-01', '2019-05-30')
    image_pre = collection_pre.first().select('B.+')

    # Get the collection of images after the event
    collection_post = collection.filterDate('2020-03-01', '2020-05-30')
    image_post = collection_post.first().select('B.+')

    # Export pre and post-event images to Google Drive
    task_pre = ee.batch.Export.image.toDrive(image_pre,
                                             description='{}_pre'.format(region),
                                             folder='SIAME_Project',
                                             fileNamePrefix='roi_{}_pre'.format(region),
                                             scale=10,
                                             maxPixels=1e11,
                                             region=roi)
    task_pre.start()

    task_post = ee.batch.Export.image.toDrive(image_post,
                                              description='{}_post'.format(region),
                                              folder='SIAME_Project',
                                              fileNamePrefix='roi_{}_post'.format(region),
                                              scale=10,
                                              maxPixels=1e11,
                                              region=roi)
    task_post.start()
