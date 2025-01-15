
import pandas as pd
import ee
import geemap


Map=geemap.Map()

roi = ee.FeatureCollection("users/Molly_HK/daymet_area")

# Define the start and end dates for the time series
# startDate = '2002-01-01'
# endDate = '2002-12-31'
vegIndex = 'EVI2'
th = 0.2
threshMin = 0.1
# scale = 50

year1 = "2002" ###c
lag = 20
startDate = year1+'-01-01'
endDate = year1+'-12-31'



dataset = (ee.ImageCollection("MODIS/061/MOD09Q1").filterBounds(roi).filterDate(startDate, endDate))  #MOD09Q1 250m

lc = ee.ImageCollection('MODIS/061/MCD12Q1').filterBounds(roi).filter(ee.Filter.date(startDate, endDate)).first()
lc_unique = lc.select('LC_Type1').eq(12)  #cropland

def clip_mask(image):
  lc = image.updateMask(lc_unique).clip(roi)
  return lc

filter_lc = dataset.map(clip_mask)

def EVI2(image):
  evi2 = image.expression(
    '2.5*(NIR-RED) / (NIR+2.4*RED+1)',{
      'RED': image.select('sur_refl_b01').multiply(0.0001), #620–670 nm
      'NIR': image.select('sur_refl_b02').multiply(0.0001),  #near-infrared band, 841–876 nm
    })

  return image.addBands((evi2).rename('EVI2'))

withEVI2 = filter_lc.map(EVI2)
styleParams = {
  "fillColor": 'b5ffb4',
  "color": '00909F',
  "width": 3.0,
};

Map.addLayer(withEVI2,styleParams,"withEVI2",False)
Map


# SECTION 4 - Generate composites

precipitation = withEVI2.select('EVI2')

ft = ee.FeatureCollection(ee.List([]))
listDates = ee.List.sequence(ee.Date(startDate).millis(), ee.Date(endDate).millis(), 86400000*lag) # days to mills. 24hx60minx60sx1000ms

#  map the dates
def dds(dd):
  date_window = ee.Date(ee.Number(dd))
  date_startW = date_window.advance(-lag/2, 'days')
  date_endW = date_window.advance(lag/2, 'days')
  col_window = precipitation.filterDate(date_startW,date_endW)
  out = col_window.reduce(ee.Reducer.mean())
  # output the day_times, means, or value is empty.
  return out.set('system:time_start',date_window.millis()).set('empty', col_window.size().eq(0))

colDekadsRaw = ee.ImageCollection(listDates.map(dds).flatten())


# colDekadsRaw
def fill_evi2_min(im):
    return ee.Image(threshMin).rename('EVI2_mean').copyProperties(im, ['system:time_start'])

# Using min to fill the empty value
colDekadsEmpty = colDekadsRaw.filterMetadata('empty', 'equals', 1).map(fill_evi2_min)

colDekads = colDekadsRaw.filterMetadata('empty', 'equals', 0).merge(colDekadsEmpty)

# Using the moving average value to fit the data
def moving_average_gap_filling(im):
    lag = 20  #Setting the day is 20
    date_window = ee.Date(im.get('system:time_start'))
    date_startW = date_window.advance(-lag*2, 'days')
    date_endW = date_window.advance(lag*2, 'days')


    meanIm1 = colDekads.filterDate(date_startW,date_window.advance(1, 'days')).reduce(ee.Reducer.mean())
    meanIm2 = colDekads.filterDate(date_window.advance(-1, 'days'),date_endW).reduce(ee.Reducer.mean())

    meanIm = (meanIm1.add(meanIm2)).divide(2)

    return im.unmask(meanIm).copyProperties(im, ['system:time_start'])
    # return im.unmask(meanIm).copyProperties(im, ['system:time_start'])

colDekads = colDekads.map(moving_average_gap_filling)
colDekads = colDekads.sort('system:time_start')

# SECTION 2 - Define function for cubic interpolation

def cubicInterpolation(collection,step):
    listDekads = ee.List.sequence(1,collection.size().subtract(3),1)


    def col_inter(ii):
        ii = ee.Number(ii)
        p0 = ee.Image(collection.toList(10000).get(ee.Number(ii).subtract(1)))
        p1 = ee.Image(collection.toList(10000).get(ii))
        p2 = ee.Image(collection.toList(10000).get(ee.Number(ii).add(1)))
        p3 = ee.Image(collection.toList(10000).get(ee.Number(ii).add(2)))


        diff01 = ee.Date(p1.get('system:time_start')).difference(ee.Date(p0.get('system:time_start')), 'day')
        diff12 = ee.Date(p2.get('system:time_start')).difference(ee.Date(p1.get('system:time_start')), 'day')
        diff23 = ee.Date(p3.get('system:time_start')).difference(ee.Date(p2.get('system:time_start')), 'day')


        diff01nor = diff01.divide(diff12)
        diff12nor = diff12.divide(diff12)
        diff23nor = diff23.divide(diff12)


        f0 = p1
        f1 = p2
        f0p = (p2.subtract(p0)).divide(diff01nor.add(diff12nor))
        f1p = (p3.subtract(p1)).divide(diff12nor.add(diff23nor))


        a = (f0.multiply(2)).subtract(f1.multiply(2)).add(f0p).add(f1p)
        b = (f0.multiply(-3)).add(f1.multiply(3)).subtract(f0p.multiply(2)).subtract(f1p)
        c = f0p
        d = f0

        xValues = ee.List.sequence(0,diff12.subtract(1),step)
        xDates = ee.List.sequence(p1.get('system:time_start'),p2.get('system:time_start'),86400000)


        def divide_fun(x):
            im = ee.Image(ee.Number(x).divide(diff12))
            return (im.pow(3)).multiply(a).add((im.pow(2)).multiply(b)).add(im.multiply(c)).add(d).set('system:time_start',ee.Number(xDates.get(x)))

        interp = xValues.map(divide_fun)
        return interp


    # colInterp = listDekads.map(col_inter)
    colInterp = listDekads.map(col_inter)
    print(type(colInterp))
    colInterp = ee.ImageCollection(colInterp.flatten())

    return colInterp


# SECTION 6 - Interpolate time series

interp = cubicInterpolation(colDekads,1)

# SECTION 7 - Estimate phenology metrics


init = ee.Image(ee.Date(str((int(year1)-1))+'-12-31').millis());

def EVI2_interp(im):
  return im.rename('EVI2_interp').addBands(im.metadata('system:time_start','date1')).set('system:time_start', im.get('system:time_start'))

interp = interp.map(EVI2_interp)  #add time band
print(type(interp))

minND = ee.Image(threshMin)
maxND = colDekads.max()
amplitude = maxND.subtract(minND)

thresh = amplitude.multiply(th).add(minND).rename('EVI2_interp')

def th_interp(im):
  out = im.select('EVI2_interp').gt(thresh);
  return im.updateMask(out).copyProperties(im,['system:time_start'])


col_aboveThresh = interp.map(th_interp)
# print(col_aboveThresh)
print(type(col_aboveThresh))


SoS = col_aboveThresh.reduce(ee.Reducer.firstNonNull()).select('date1_first').rename('SoS')
SoS_doy = SoS.subtract(init).divide(86400000);

EoS = col_aboveThresh.reduce(ee.Reducer.lastNonNull()).select('date1_last').rename('EoS')
EoS_doy = EoS.subtract(init).divide(86400000);

doy = EoS_doy.subtract(SoS_doy).rename('DOY')

print(type(SoS_doy))
print(type(EoS_doy))
print(type(doy))


################################### Map AddLayer ############################
# palettes = ee.require('users/gena/packages:palettes')
# phenoPalette = palettes.colorbrewer.RdYlGn[9]

# RdYlGn:{3:["fc8d59","ffffbf","91cf60"],4:["d7191c","fdae61","a6d96a","1a9641"],5:["d7191c","fdae61","ffffbf","a6d96a","1a9641"],6:["d73027","fc8d59","fee08b","d9ef8b","91cf60","1a9850"],7:["d73027","fc8d59","fee08b","ffffbf","d9ef8b","91cf60","1a9850"],8:["d73027","f46d43","fdae61","fee08b","d9ef8b","a6d96a","66bd63","1a9850"],9:["d73027","f46d43","fdae61","fee08b","ffffbf","d9ef8b","a6d96a","66bd63","1a9850"],10:["a50026","d73027","f46d43","fdae61","fee08b","d9ef8b","a6d96a","66bd63","1a9850","006837"],11:["a50026","d73027","f46d43","fdae61","fee08b","ffffbf","d9ef8b","a6d96a","66bd63","1a9850","006837"]},
phenoPalette1 = ['ffffff',"76ee00","7fff00","76ee00","66cd00","458b00","22bb22",'4aff00','008000',"006400"]
phenoPalette2 = ['ffffff',"ffffe0","fffacd","fafad2",'ffff00',"ffff00","ffd700","ffa500",'ffbc00','ff4500']
# phenoPalette1 = ["FF4040","458B00","FFFF00"]
visSoS = {min:0,max:200,"palette":phenoPalette1}
visEoS = {min:180,max:300,"palette":phenoPalette2}


# Map.addLayer(s.clip(roi),visSoS,'SoS')

Map.addLayer(SoS_doy.clip(roi),visSoS,'SoS')
Map.addLayer(EoS_doy.clip(roi),visEoS,'EoS')
Map.addLayer(doy.clip(roi),visEoS,'doy')
Map.addLayer(roi,{},'ROI')
Map.addLayerControl()
# Map.plot_colormap('SOS', width=8.0, height=0.4, orientation='horizontal')
Map.add_colorbar(visSoS, label="SOS", layer_name="SOS")
Map.add_colorbar(visEoS, label="EOS", layer_name="EOS")
Map.add_colorbar(visEoS, label="EOS", layer_name="EOS")
# Map.addlengend()

Map
