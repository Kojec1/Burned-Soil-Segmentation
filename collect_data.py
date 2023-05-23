import json
import ee
import config

ee.Initialize()

with open(config.COORDINATES_PATH) as f:
    coords = json.load(f)

for coord in coords:
    region = coord['region']

    point = coord['point']
    poi = ee.Geometry.Point(point)

    roi = coord['coordinates']
    roi = ee.Geometry.Polygon(roi)

    collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(poi).filter(
        ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10)).sort('CLOUDY_PIXEL_PERCENTAGE')

    collection_pre = collection.filterDate('2019-03-01', '2019-05-30')
    image_pre = collection_pre.first().select('B.+')

    collection_post = collection.filterDate('2020-03-01', '2020-05-30')
    image_post = collection_post.first().select('B.+')

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
