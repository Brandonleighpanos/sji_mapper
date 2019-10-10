from irisreader.utils.date import from_Tformat, to_Tformat, to_epoch
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial.distance import cdist
from irisreader import observation
from sklearn import preprocessing
from matplotlib import animation
from IPython.display import HTML
import matplotlib.pyplot as plt
import numpy as np
import pickle

class quantize_spectra:
    '''
     A class for generating centroid masked SJI and feature animations.
     
        Attributes
        ---------
        data: array
            IRIS observation data in the form [steps, rasters, y, lambda].
        raster: array
            IRISreader object in the form [exposure, y, lambda] obtained from obs.raster("line").
        sji: array
            IRISreader object in the form [xpix, ypix, frames].
        hdrs: array
            Time dependent headers in the form [step position, raster] with hdrs[0][1] containing time info of the
            zeroth step position on the second raster sweep.
        mode: str
            Whether to generate a centroid or feature mask.
        hmap: str
            Heat map used for the feature mask.
            
        Methods
        -------
        mini_batch_k_means(self, data, n_clusters=10, batch_size=10, n_init=10, verbose=0):
            Performs kmeans and returns the datas centroids, labels and inertia.
        label(self, centroids, data):
            Label the data with custom centroids.
        raster_x_coord( self, sjiind ):
            Finds the locations of the slit position for a giving SJI, taking into account
            the PZT driven secondary mirror.
        get_raster_centroid_mask(self, step, clr_dic, labels):
            Returns a "centroid mask" or grid for each raster containing the positions and
            colors for that frames animation.
        clean_mask(self, step, slpos_grid, y_grid, color_grid):
            Returns a centroid mask of only the centroids under investigation.
         sji_raster(self, steps):
             Returns the step that each SJI should be taken closest to in time. For instance, aligning a SJI in time
             with the first step would mean that the 64'th step would be highly disicosiated with the chosen SJI.
        animate(self, clr_dic, labels, gamma=0.2, cutoff_percentile=99.9, figsize=(15,15)):
            Iterates over each raster creating multiple SJI centroid masks that are sown together to form an animation.
            Incase mode has been set to features, movies of the evolving feature are produced.
    '''
    def __init__(self, data, raster, sji, hdrs, mode='cluster', hmap='Blues_r'):
        # sji and raster data from observation
        self.data = data
        self.raster = raster
        self.sji = sji
        self.hdrs = hdrs
        self.mode = mode
        self.hmap = hmap
        # define data parameters
        self.y = data.shape[2]
        self.lam = data.shape[3]
        self.steps = data.shape[0]
        self.n_rasters = data.shape[1]
    def mini_batch_k_means(self, data, n_clusters=10, batch_size=10, n_init=10, verbose=0):
        '''
        Return centroids and labels of a data set using the k-means algorithm in mini-batch style
        Input   - X: data in the form [m_examples, lambda]
                - n_clusters: number of groups to find in the data
                - batch_size: number of data points used for a single update step of the centroids
                  (these add together in a running stream)
                - n_init: number of convergences tested, the iteration with the lowest inertia is retained
                - verbose: output statistics
        Returns: - centroids: form [n_clusters, lambda], mean points of groups
                - labels: list of assignments indicating which centroid each data point belongs to
                - inertia: measure of performance, sum of all inter-cluster distances
        '''
        mbk = MiniBatchKMeans(init='k-means++', n_clusters=n_clusters, batch_size=batch_size,
                              n_init=n_init, max_no_improvement=10, verbose=verbose)
        # transpose for consistency
        data_transposed = np.transpose(data, (1,0,2,3))
        # put data in the form [m,lambda]
        x = data_transposed.reshape( self.steps * self.n_rasters * self.y, self.lam, order='C' )
        # replace al nan values with ones (because k-means cant handle missing values)
        x = np.nan_to_num(x, 1)
        # apply the k-means algorithm
        mbk.fit(x)
        centroids = mbk.cluster_centers_
        labels = mbk.labels_
        inertia = mbk.inertia_
        kmeans = {'centroids':centroids,
                  'labels':labels,
                  'inertia':inertia}
        return kmeans
    def label(self, centroids, data):
        '''
        encase one wants to use custom centroids, then instead of labeling via kmeans we use a fast scipy implementation.
        '''
        # transpose for consistency
        data_transposed = np.transpose(data, (1,0,2,3))
        # put data in the form [m,lambda]
        nprof = data_transposed.reshape( self.steps * self.n_rasters * self.y, self.lam, order='C' )
        dist_mat = cdist(nprof, centroids, 'euclidean')
        labels = np.argmin(dist_mat, axis=1)
        return labels
    def raster_x_coord( self, sjiind ):
        '''
        Returns n_raster slit positions ito pixels for a given SJI index.
        '''
        # work in solar coordinates
        slitcoord = self.sji.pix2coords( sjiind, np.array([self.sji.get_slit_pos(sjiind)-1,0]) ) # any y
        slitcoord_primary_y = slitcoord[1]
        slitcoord_primary_x = slitcoord[0]
        slit_offset_secondary = self.sji.headers[sjiind]['PZTX']
        # This assumes that the PZT offsets remain constant for the entire observation
        raster_offsets_secondary = np.asarray( [ self.raster.headers[i]['PZTX'] for i in range(self.steps) ] )
        # slit x position for each raster pos = position of primary - wedge tilt + fine scale secondary pztx
        slpos = slitcoord_primary_x - slit_offset_secondary + raster_offsets_secondary
        # convert back to pixels after the pztx subtraction
        slpos = [ self.sji.coords2pix(sjiind,[slpos[i],slitcoord_primary_y])[0] for i in range(self.steps) ]
        return slpos
    def get_raster_centroid_mask(self, step, clr_dic, labels):
        '''
        Returns a grid of positions and colors for a single raster centroid mask.
        '''
        slpos = self.raster_x_coord(step)
        slpos_grid = [ [sl]*self.y for sl in slpos ] # slit x-coords for each spectra for a single raster
        xcoords = np.asarray( [item for sublist in slpos_grid for item in sublist] ) # flatten nested list
        ycoords = np.asarray( list(range(self.y))*self.steps ) # y-coord for each spectra for a single raster
        if self.mode == 'cluster':
            colors = np.asarray( [ clr_dic.get(l, 'grey') for l in labels ] )
        if self.mode == 'features':
            scaler = preprocessing.StandardScaler()
            colors = scaler.fit_transform(labels.reshape(-1, 1))
        colors = colors.reshape(self.y*self.steps, self.n_rasters, order='F')
        centroid_mask = {'x':xcoords,
                         'y':ycoords,
                         'c':colors}
        return centroid_mask
    def clean_mask(self, step, slpos_grid, y_grid, color_grid):
        '''
        Returns a centroid mask with only the intresnting (non grey) centroids for a given raster step.
        '''
        if self.mode == 'cluster':
            mask = color_grid[:, step] != 'grey'
        if self.mode == 'features':
            mask = ~np.isnan(color_grid[:, step])
        centroid_mask = {'x':slpos_grid[mask],
                         'y':y_grid[mask],
                         'c':color_grid[mask, step]}
        return centroid_mask
    def sji_raster(self, steps):
        '''
        SJI time that would make most sense with all rasters (av).
        '''
        if self.steps != 1: sji_raster_pos = int(self.steps/2)
        if self.steps == 1: sji_raster_pos = 0
        return sji_raster_pos
    def animate(self, clr_dic, labels, gamma=0.2, cutoff_percentile=99.9, figsize=(15,15), transparency=.5, marker_size=5, x_range=None, y_range=None, save_path=None):
        '''
        Setup the first step in the animation.
        Input - clr_dic: dictionary of colrs associated to each centroid
              - labels: labels from k-means indicating which centroid each spectra is best represented by.
                Lables could also be a vector of feature values.
              - gamma: exponent for gamma correction that adjusts the plot color scale
              -  cutoff_percentile: "often the maximum pixels shine out everything else, even after gamma correction. In order to reduce
                 this effect, the percentile at which to cut the intensity off can be specified with cutoff_percentile
                 in a range between 0 and 100".
              - transparency: adjust the transparency of the centroid mask (0=transparent, 1=opaque)
              - marker_size: adjust the size of each marker denoting the spectra position on the SJI
              - save_path: specify a path to save the data
        Returns - centroid or feature masked SJI animation of the specified observation.
        '''
        step=0
        fig = plt.figure( figsize=figsize )
        sji_raster_pos = self.sji_raster(self.steps)
        times = [ to_epoch( from_Tformat( self.hdrs[sji_raster_pos][i]['DATE_OBS'] ) ) for i in range(self.n_rasters) ]
        sjiind = np.argmin( np.abs( np.array( self.sji.get_timestamps() )-times[step] ) )
        image = self.sji.get_image_step( sjiind ).clip(min=0.01)**gamma
        vmax = np.percentile( image, cutoff_percentile ) # constant
        im = plt.imshow( image, cmap="Greys", vmax=vmax, origin='lower' )
        centroid_mask = self.get_raster_centroid_mask(step, clr_dic, labels)
        # plot slits
        slits = np.vstack( (np.asarray(centroid_mask['x']), np.asarray(centroid_mask['y']))  )
        slit_mask = plt.scatter(centroid_mask['x'],
                                centroid_mask['y'],
                                c = 'white',
                                s=.1)
        # plot centroid mask
        clean_centroid_mask = self.clean_mask(step, centroid_mask['x'], centroid_mask['y'], centroid_mask['c'])
        if self.mode == 'cluster':
            clrs =clean_centroid_mask['c']
        if self.mode == 'features':
            cmap = plt.cm.get_cmap(self.hmap)
            clrs = cmap(clean_centroid_mask['c'])
        scat = plt.scatter(clean_centroid_mask['x'],
                           clean_centroid_mask['y'],
                           c = clrs,
                           s=marker_size,
                           alpha=transparency)
        date_obs = self.sji.headers[sjiind]['DATE_OBS']
        im.axes.set_title( "Frame {}: {}".format( step, date_obs ), fontsize=18, alpha=.8)
        plt.xlim(x_range[0],x_range[1])
        plt.ylim(y_range[0],y_range[1])
        plt.close( fig )
        # do nothing in the initialization function
        def init():
            return im,
        # animation function
        def update(i):
            '''
            Update the data for each successive raster.
            '''
            sjiind = np.argmin( np.abs( np.array( self.sji.get_timestamps() )-times[i] ) )
            date_obs = self.sji.headers[sjiind]['DATE_OBS']
            im.axes.set_title( "Frame {}: {}".format( i, date_obs ), fontsize=18, alpha=.8)
            im.set_data( self.sji.get_image_step( sjiind ).clip(min=0.01)**gamma )
            centroid_mask = self.get_raster_centroid_mask(0, clr_dic, labels) # make 0->i but doesnt seem to work
            slits = np.vstack( (np.asarray(centroid_mask['x']), np.asarray(centroid_mask['y']))  )
            slit_mask.set_offsets( slits.T )

            clean_centroid_mask = self.clean_mask(i, centroid_mask['x'], centroid_mask['y'], centroid_mask['c'])
            dd = np.vstack( (np.asarray(clean_centroid_mask['x']), np.asarray(clean_centroid_mask['y']))  )
            scat.set_offsets( dd.T )
            if self.mode == 'cluster':
                scat.set_color(clean_centroid_mask['c'])
            if self.mode == 'features':
                cmap = plt.cm.get_cmap(self.hmap)
                scat.set_color(cmap(clean_centroid_mask['c']))
            return im,
        # Call the animator.  blit=True means only re-draw the parts that have changed.
        anim = animation.FuncAnimation( fig, lambda i: update(i), init_func=init, frames=self.n_rasters,
            interval=200, blit=True )
        # Save animation if requested
        if save_path is not None:
            anim.save( save_path )
        # Show animation in notebook
        return HTML(anim.to_html5_video())



def centroid_summary( centroids, colors, rows=5, cols=4 ):
    '''
    plots a summary of the centroids found by the k-means run
    '''
    clr_dic = {i:colors.get(i, 'grey') for i in range(len(centroids))}
    k_line_core = 2796.34
    h_line_core = 2803.52
    in_text = 8
    nprofxax = np.linspace( 2794.14, 2805.72, 240 )
    kline = np.where( abs(nprofxax - k_line_core) == min(abs(nprofxax - k_line_core)))
    hline  = np.where( abs(nprofxax - h_line_core) == min(abs(nprofxax - h_line_core)))

    # full in some data to make clean plots when no centroids
    empty = ( rows*cols - len(centroids) )

    fig, axs = plt.subplots(rows, cols, figsize = (15, 15) )
    ax=axs.ravel()
    for k in range(len(centroids)):
        ax[k].plot(centroids[k], color='black', linewidth=.5, linestyle='--')
        ax[k].plot(centroids[k], color=clr_dic.get(k,'black'), linewidth=2, alpha=.5)
        ax[k].axvline(x=kline,color='black',linewidth = .5)
        ax[k].axvline(x=hline,color='black',linewidth = .5)
        ax[k].text( .04, .86, str(k), transform=ax[k].transAxes, size=in_text)
        ax[k].axes.get_xaxis().set_ticks([])
        ax[k].axes.get_yaxis().set_ticks([])

    for nothing in range( len(centroids), rows*cols ):
        ax[nothing].plot(0,0)
        ax[nothing].set_axis_off()
    plt.show()
    return None


def merge(centroids, items, mode='merge'):
    if mode == 'merge':
        less_centroids = np.delete(centroids,items, axis=0)
        new_item = np.mean(centroids[items], axis=0)
        new_centroids = np.concatenate((less_centroids,new_item.reshape(1,centroids.shape[1])),axis=0)
    if mode == 'del':
        new_centroids = np.delete(centroids, items, axis=0)
    return new_centroids


def sav(centroids, color_dictionary, desc=''):
    save_path = '/data1/userspace/bpanos/multiline/centroids/' + name + '/'
    with open(save_path + flares[ff].split('_')[-1] + '_' + desc + ".pickle", "wb") as f:
        pickle.dump( (centroids, color_dictionary), f )
    return None



def load(path):
    with open(path, "rb") as f:
        centroids, color_dictionary = pickle.load(f)
    return centroids, color_dictionary
