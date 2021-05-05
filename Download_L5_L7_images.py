"""
This code obtains Landsat 5 TM and Landsat 7 ETM+ image pairs, and filters out cloud then splits the image into 128x128 pixel boxes for the
30m bands and 256x256 pixel boxes for the 15m pansharpened band
"""
import numpy as np
import os
import sys
import ee
from google.colab import drive

ee.Authenticate()
ee.Initialize()

drive.mount('/content/gdrive')
os.chdir('/content/gdrive/MyDrive/Colab_data')

#The bands of interest
bands_use_L5=['B1','B2','B3','B4','B5','B6','B7']
bands_use_L7=['B1','B2','B3','B4','B5','B6_VCID_2','B7']

band_len_5=len(bands_use_L5)
band_len_7=len(bands_use_L7)

L7_codes=["065015_20040909", "078013_20050721"] #Landsat 7 tile codes to download
L5_codes=["064016_20040910", "077014_20050722"] #Landsat 5 tile codes to download

#Define region of images to download
min_x = [-141.65508291710566,-159.2120129564438]
max_x = [-142.6856003611202,-160.4864270189438]
min_y = [63.386101452689466,66.12977287716515]
max_y = [63.85857693568399,66.58794070628201]

def find_area(image,geometry,band):
  """
  Determines the number of pixels in each sub-image that is downloaded.
  Enables verification that both images are complete and can be saved.
  """
  image2=image.select(band)
  sumDictionarypolygon1 = (image2.reduceRegion(
  reducer= ee.Reducer.count(),
  geometry= geometry,
  scale= 30,
  maxPixels= 1e9))
  return np.array((ee.Array(sumDictionarypolygon1.get(band)).getInfo()))

def CLOUD_MASK(image):
  scored = ee.Algorithms.Landsat.simpleCloudScore(image);
  mask = scored.select(['cloud']).lte(20);
  masked = image.updateMask(mask);
  return masked

def LatLonImg(img,num_bands,scale,geometry):
  """
  Obtains the latitude and longitude information from each image
  """
  img = img.addBands(ee.Image.pixelLonLat())

  img = img.reduceRegion(reducer=ee.Reducer.toList(),\
                                      geometry=geometry,\
                                      maxPixels=1e7,\
                                      scale=scale);

  lats = np.array((ee.Array(img.get("latitude")).getInfo()))
  lons = np.array((ee.Array(img.get("longitude")).getInfo()))

  all_data_return=[]
  for oo in range(1,num_bands+1):
    result_str='result'+str(oo)
    all_data_return.append(np.array((ee.Array(img.get(result_str)).getInfo())))
  
  return lats, lons, all_data_return
        
# covert the lat, lon and array into an image
def toImage(lats,lons,all_data_bands,band_len):
  """
  Converts the ee.Image Earth Engine object into a NumPy array
  """

    # get the unique coordinates
    uniqueLats = np.unique(lats)
    uniqueLons = np.unique(lons)

    # get number of columns and rows from coordinates
    ncols = len(uniqueLons)
    nrows = len(uniqueLats)

    # determine pixelsizes
    ys = uniqueLats[1] - uniqueLats[0]
    xs = uniqueLons[1] - uniqueLons[0]

    # create an array with dimensions of image
    arr = np.zeros([nrows, ncols,band_len], np.float32) #-9999

    # fill the array with values
    counter =0
    for y in range(0,len(arr),1):
        for x in range(0,len(arr[0]),1):
            if lats[counter] == uniqueLats[y] and lons[counter] == uniqueLons[x] and counter < len(lats)-1:
                counter+=1
                for pp in range(0,len(all_data_bands)):
                  arr[len(uniqueLats)-1-y,x,pp] = all_data_bands[pp][counter] # we start from lower left corner
    return arr


step_size=0.0348 #Size of each sub-image (with units of degrees). ).0348 is the optimised value for RAM limitations on Google Colab
size_scene=128 #Pixel size of the scene to save
ALL_im1=[]
ALL_im2=[]
ALL_im_pan=[]

#Iterating through each of the images
for shape in range(len(L7_codes)):
  tot_x_iter=int((max_x[shape]-min_x[shape])/step_size)
  tot_y_iter=int((max_y[shape]-min_y[shape])/step_size)
  
  for x_iter in range(0,tot_x_iter):
    for y_iter in range(0,tot_y_iter):
      min_x_scene=min_x[shape]+x_iter*step_size
      max_x_scene=min_x_scene+step_size
      min_y_scene=min_y[shape] + y_iter*step_size
      max_y_scene=min_y_scene+step_size

      min_geom=ee.Geometry.Rectangle(
                [min_x_scene, min_y_scene,max_x_scene,max_y_scene], None, False);

      L7_codee='LE07_'+L7_codes[shape]
      L5_codee='1_LT05_'+L5_codes[shape]
      
      #Retrieve the L7 and L5 images from Earth Engine
      L7_image = (ee.ImageCollection("LANDSAT/LE07/C01/T1_TOA")
      .filterMetadata('system:index','equals',L7_codee)    
      .map(CLOUD_MASK)
      .median()
      .select(bands_use_L7)
      .clip(min_geom))

      L7_pan = (ee.ImageCollection("LANDSAT/LE07/C01/T1_TOA")
      .filterMetadata('system:index','equals',L7_codee)    
      .map(CLOUD_MASK)
      .median()
      .select('B8')
      .clip(min_geom))

      L5_image = (ee.ImageCollection('LANDSAT/LT05/C01/T1_TOA').merge(ee.ImageCollection('LANDSAT/LT05/C01/T2_TOA'))
      .filterMetadata('system:index','equals',L5_codee)    
      .map(CLOUD_MASK)
      .median()
      .select(bands_use_L5)
      .clip(min_geom))
      
      
      area7=find_area(L7_image,min_geom,'B1')
      area5=find_area(L5_image,min_geom,'B1')
      areapan=find_area(L7_pan,min_geom,'B8')

      if area7==area5 and areapan==(area7) and area5>0:  #Test that images are equivalent sizes
        #Convert into NumPy arrays
        L5_band_result_names=['result'+str(y) for y in range(1,band_len_5+1)]
        result5 = L5_image.rename(L5_band_result_names)
        lat5, lon5, all_L5_data = LatLonImg(result5,band_len_5,30, min_geom)

        L7_band_result_names=['result'+str(y) for y in range(1,band_len_7+1)]
        result7 = L7_image.rename(L7_band_result_names)
        lat7, lon7, all_L7_data = LatLonImg(result7,band_len_7,30,min_geom)

        result_pan=L7_pan.rename('result1')
        latpan, lonpan, all_pan_data = LatLonImg(result_pan,1,15,min_geom)

        if len(lat5)==len(all_L5_data[0]) and len(lat7)==len(all_L7_data[0]) and len(latpan)==len(all_pan_data[0]): #Testing that images are same size
          
          #Convert into NumPy arrays
          im1_out5 = toImage(lat5,lon5,all_L5_data,band_len_5)
          im2_out7  = toImage(lat7,lon7,all_L7_data,band_len_7)
          im3_outpan=toImage(latpan, lonpan, all_pan_data,1)

          im1_y, im1_x=np.shape(im1_out5)[:2]
          ALL_im1.append(im1_out5[:size_scene,:size_scene,:])

          im2_y, im2_x=np.shape(im2_out7)[:2]
          ALL_im2.append( im2_out7[:size_scene,:size_scene,:])

          ALL_im_pan.append(im3_outpan[:size_scene*2,:size_scene*2,:])
          print('obtained images: ',str(len(ALL_im1)))
          
          if len(ALL_im1)%1000==0:   
            #Save collection of images to cloud storage when large enough
            im1_save=np.stack([ALL_im1[q] for q in range(len(ALL_im1))],axis=0)[:,:,:,:len(bands_use_L5)]
            ALL_im1=[]
            im2_save=np.stack([ALL_im2[q] for q in range(len(ALL_im2))],axis=0)[:,:,:,:len(bands_use_L7)]
            ALL_im2=[]
            impan_save=np.stack([ALL_im_pan[q] for q in range(len(ALL_im_pan))],axis=0)[:,:,:,:1]
            ALL_im_pan=[]

            string_save_1='IM1_FINAL_APR'+str(count_saved_pickle)
            np.save(string_save_1, im1_save)

            string_save_2='IM2_FINAL_APR'+str(count_saved_pickle)
            np.save(string_save_2, im2_save)

            string_save_pan='IMpan_FINAL_APR'+str(count_saved_pickle)
            np.save(string_save_pan, impan_save)

            count_saved_pickle+=1
            im1_save,im2_save,impan_save=[],[],[]

            print('SAVED images!')
        else:
          pass

#Save last collection of images to cloud
im1_save=np.stack([ALL_im1[q] for q in range(len(ALL_im1))],axis=0)[:,:,:,:len(bands_use_L5)]
ALL_im1=[]
im2_save=np.stack([ALL_im2[q] for q in range(len(ALL_im2))],axis=0)[:,:,:,:len(bands_use_L7)]
ALL_im2=[]
impan_save=np.stack([ALL_im_pan[q] for q in range(len(ALL_im_pan))],axis=0)[:,:,:,:1]
ALL_im_pan=[]

string_save_1='IM1_FINAL_APR'+str(count_saved_pickle)
np.save(string_save_1, im1_save)

string_save_2='IM2_FINAL_APR'+str(count_saved_pickle)
np.save(string_save_2, im2_save)

string_save_pan='IMpan_FINAL_APR'+str(count_saved_pickle)
np.save(string_save_pan, impan_save)
print('SAVED images!')
