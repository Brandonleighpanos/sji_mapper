# IRIS sji mapper

*by Brandon Panos*   

This repository contains a set of functions for the production of centroid and feature SJI animations. 

***get_features.py*** - Scrapes a number of features associated with the Mg II h\&k lines (see documentation).  
***movie_utils.py*** - Creates an object from an IRISreader observation that can be quantized with the ***k-means*** method.  
The resulting features or centroids are then mapped onto the individual SJIs and woven into animations.  
The code is still in test phase and will be updated to handle all IRIS observation modes as well as include line features from different spectra. The code is fully adaptable to any IRIS line.  

 <img src="cover.png" width="1000">
 
###### The above images show a centroid mask (upper left) as well as core intensities, triplet emission intensities and k/h integrated line ratio masks.  
