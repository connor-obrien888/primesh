import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as ks
import tensorflow.keras.backend as K
import tensorflow.keras.callbacks as ksc
import scipy.interpolate as spi
import joblib

class prime(ks.Model):
    def __init__(self, model = None, in_scaler = None, tar_scaler = None, loc_scaler = None, in_keys = None, tar_keys = None, out_keys = None, hps = [55, 16, 0.05]):
        '''
        Class to wrap a keras model to be used with the SH dataset.

        Parameters:
            model (keras model): Keras model to be used for prediction
            in_scaler (sklearn scaler): Scaler to be used for input data
            tar_scaler (sklearn scaler): Scaler to be used for target data
            loc_scaler (sklearn scaler): Scaler to be used for location data
        '''
        super(prime, self).__init__()
        if model is None:
            self.model = self.build_model()
            self.model.load_weights('primesh.h5')
            self.model = model
        else:
            self.model = model
        if in_scaler is None:
            self.in_scaler = joblib.load('in_scaler.pkl')
        else:
            self.in_scaler = in_scaler
        if tar_scaler is None:
            self.tar_scaler = joblib.load('tar_scaler.pkl')
        else:
            self.tar_scaler = tar_scaler
        if loc_scaler is None:
            self.loc_scaler = joblib.load('loc_scaler.pkl')
        else:
            self.loc_scaler = loc_scaler
        if in_keys is None:
            self.in_keys = ['B_xgsm', 'B_ygsm', 'B_zgsm', 'Vi_xgse', 'Vi_ygse', 'Vi_zgse', 'Ni', 'Vth', 'R_xgse', 'R_ygse', 'R_zgse', 'target_R_xgse', 'target_R_ygse', 'target_R_zgse'] #Wind data keys to include in input dataset
        else:
            self.in_keys = in_keys
        if tar_keys is None:
            self.tar_keys = ['B_xgsm', 'B_ygsm', 'B_zgsm', 'Vi_xgse', 'Vi_ygse', 'Vi_zgse', 'Ni', 'Tipar', 'Tiperp'] #Targets from MMS dataset to match with input data
        else:
            self.tar_keys = tar_keys
        if out_keys is None:
            self.out_keys = ['B_xgsm', 'B_xgsm_sig', 'B_ygsm', 'B_ygsm_sig', 'B_zgsm', 'B_zgsm_sig', 'Vi_xgse', 'Vi_xgse_sig', 'Vi_ygse', 'Vi_ygse_sig', 'Vi_zgse', 'Vi_zgse_sig', 'Ni', 'Ni_sig', 'Tipar', 'Tipar_sig', 'Tiperp', 'Tiperp_sig']
        else:
            self.out_keys = out_keys
        self.window = hps[0]
        self.stride = hps[1]
        self.fraction = hps[2]
    def predict(self, input):
        '''
        High-level wrapper function to generate prime predictions from input dataframes.
        
        Parameters:
            input (dataframe, ndarray): Input data to be scaled and predicted
        Returns:
            output (dataframe): Scaled output data
        '''
        if isinstance(input, pd.DataFrame): #If input is a dataframe
            input_arr = input[self.in_keys].to_numpy() #Convert input dataframe to array
        if isinstance(input, np.ndarray): #If input is an array
            input_arr = input #Set input array to input
        output_arr = self.predict_raw(input_arr) #Predict with the keras model
        output = pd.DataFrame(output_arr, columns = self.out_keys) #Convert output array to dataframe
        output_epoch = input['Epoch'].to_numpy()[(self.window-1):] #Stage an epoch column to be added to the output dataframe
        output_epoch += pd.Timedelta(seconds = 100*self.stride) #Add lead time to the epoch column
        output['Epoch'] = output_epoch #Add the epoch column to the output dataframe
        return output
    def predict_raw(self, input):
        '''
        Wrapper function to predict with a keras model.
        '''
        input_scaled = self.in_scaler.transform(input)
        input_arr = np.zeros((len(input_scaled)-(self.window-1), self.window, len(self.in_keys))) #Reshape input data to be 3D
        for i in np.arange(len(input_scaled)-(self.window-1)):
            input_arr[i,:,:] = input_scaled[i:(i+self.window)] #Move the 55 unit window through the input data
        output_unscaled = self.model.predict(input_arr)
        output = np.zeros((len(output_unscaled),len(self.out_keys))) #Stage output data to be 2x target dimensions
        output[:, ::2] = self.tar_scaler.inverse_transform(output_unscaled[:, ::2]) #Mean values
        output[:, 1::2] = np.abs(self.tar_scaler.inverse_transform(output_unscaled[:, ::2] + output_unscaled[:, 1::2]) - self.tar_scaler.inverse_transform(output_unscaled[:, ::2])) #Standard deviations
        return output
    def predict_grid(self, gridsize, x_extent, framenum, bx, by, bz, vx, vy, vz, ni, vt, rx, ry, rz, y_extent = None, z_extent = None, loc_mask = False, subtract_ecliptic = False):
        '''
        Generate predictions from prime model on a grid of points.

        Parameters:
            gridsize (float): Spacing of grid points
            x_extent (list): Range of x values to calculate on
            framenum (int): Number of frames to calculate
            bx (float, array-like): IMF Bx value. If array like, must be of length framenum.
            by (float, array-like): IMF By value. If array like, must be of length framenum.
            bz (float, array-like): IMF Bz value. If array like, must be of length framenum.
            vx (float, array-like): Solar wind Vx value. If array like, must be of length framenum.
            vy (float, array-like): Solar wind Vy value. If array like, must be of length framenum.
            vz (float, array-like): Solar wind Vz value. If array like, must be of length framenum.
            ni (float, array-like): Solar wind ion density value. If array like, must be of length framenum.
            vt (float, array-like): Solar wind ion thermal speed value. If array like, must be of length framenum.
            rx (float, array-like): Wind spacecraft position x value. If array like, must be of length framenum.
            ry (float, array-like): Wind spacecraft position y value. If array like, must be of length framenum.
            rz (float, array-like): Wind spacecraft position z value. If array like, must be of length framenum.
            y_extent (list): Range of y values to calculate on. If None, z_extent must be specified.
            z_extent (list): Range of z values to calculate on. If None, y_extent must be specified.
            loc_mask (bool): Whether or not to mask the output to just select the region between the bow shock and magnetopause (Shue et al 1998 and Jelinek et al 2012).
            subtract_ecliptic (bool): Whether or not to subtract the Earth's motion in the ecliptic from Vy
        Returns:
            output_grid (ndarray): Array of predicted values on the grid. Shape (framenum, x_extent/gridsize, y_extent/gridsize, 18)
        '''
        x_arr = np.arange(x_extent[0], x_extent[1], gridsize) #Create a grid to calculate the magnetosheath conditions on
        if y_extent is None and z_extent is None:
            raise ValueError('Must specify either y_extent or z_extent')
        if y_extent is not None and z_extent is not None:
            raise ValueError('Cannot specify both y_extent and z_extent')
        if y_extent is not None:
            y_arr = np.arange(y_extent[0], y_extent[1], gridsize) #Create a grid to calculate the magnetosheath conditions on
        if z_extent is not None:
            y_arr = np.arange(z_extent[0], z_extent[1], gridsize) #Create a grid to calculate the magnetosheath conditions on (called y but contains z because im lazy)
        x_grid, y_grid = np.meshgrid(x_arr, y_arr) #Create a grid to calculate the magnetosheath conditions on
        input_seed = np.zeros((len(x_grid.flatten())*framenum, len(self.in_keys))) #Initialize array to hold the input data before unfolding it
        for idx, element in enumerate([bx, by, bz, vx, vy, vz, ni, vt, rx, ry, rz]): #Loop through the input data and repeat it
            try:
                iter(element) #Check if the element is iterable
                input_seed[:, idx] = np.repeat(element, len(x_grid.flatten())) #If it is, repeat it for each grid point
            except TypeError: #This error throws if iter(element) fails (i.e. element is not iterable)
                input_seed[:, idx] = np.repeat(element, framenum*len(x_grid.flatten())) #If it isn't, repeat it for each grid point *and frame*
        loc_arr = np.zeros((len(x_grid.flatten())*framenum, 3)) #Initialize array to hold the location data
        loc_arr[:, 0] = np.tile(x_grid.flatten(), framenum)
        if y_extent is not None:
            loc_arr[:, 1] = np.tile(y_grid.flatten(), framenum)
            loc_arr[:, 2] = 0 #Set target z to 0
        if z_extent is not None:
            loc_arr[:, 2] = np.tile(y_grid.flatten(), framenum)
            loc_arr[:, 1] = 0 #Set target y to 0
        input_seed_scaled = self.in_scaler.transform(input_seed) #Scale the input data
        input_seed_scaled[:,11:14] = self.loc_scaler.transform(loc_arr) #Scale the location data
        input_seed_scaled = np.repeat(input_seed_scaled, self.window, axis = 0) #Repeat the input data 55 times to make static timeseries
        input_arr = input_seed_scaled.reshape(len(x_grid.flatten())*framenum, self.window, len(self.in_keys)) #Reshape the input data into the correct shape
        output_arr = self.model.predict(input_arr) #Predict the output data
        output = np.zeros((len(output_arr),len(self.out_keys))) #Stage output data to be 2x target dimensions
        output[:, ::2] = self.tar_scaler.inverse_transform(output_arr[:, ::2]) #Mean values
        output[:, 1::2] = np.abs(self.tar_scaler.inverse_transform(output_arr[:, ::2] + output_arr[:, 1::2]) - self.tar_scaler.inverse_transform(output_arr[:, ::2])) #Standard deviations
        output_grid = output.reshape(framenum, len(y_arr), len(x_arr), len(self.out_keys)) #Reshape the output data into the correct shape
        output_grid = np.swapaxes(output_grid, 1, 2) #Move the y axis to the second axis
        if loc_mask:
            bs_array = np.zeros((framenum, len(y_arr))) #Initialize array to hold the bow shock locations
            mp_array = np.zeros((framenum, len(y_arr))) #Initialize array to hold the magnetopause locations
            #Make a mask for all points outside the bow shock or inside the magnetopause
            output_mask = np.zeros(output_grid.shape, dtype=bool) #Initialize array to hold the frame mask
            for i in np.arange(framenum):
                if framenum == 1: #If there is only one frame turn pdyn and bz into iterables
                    pdyn_iter = [ni * vx**2 * 1.673e-6] #Dynamic pressure (nPa)
                    bz_iter = [bz]
                else:
                    pdyn_iter = ni * vx**2 * 1.673e-6
                    bz_iter = bz
                bs_array[i,:] = jelinek_bs(y_arr, pdyn_iter[i])
                mp_array[i,:] = shue_mp_interp(y_arr, pdyn_iter[i], bz_iter[i])
                for idy, y in enumerate(y_arr):
                    for idx, x in enumerate(x_arr):
                        if ((x > bs_array[i,idy])|(x < mp_array[i,idy])):
                            output_mask[i,idx,idy,:] = True
            #Make a masked version of the output grid
            output_grid = np.ma.masked_array(output_grid, mask=output_mask)
        if subtract_ecliptic: #If subtract_ecliptic is true, subtract the Earth's motion in the ecliptic from Vy
            output_grid[:,:,:,8] -= 29.8
        return output_grid
    def load_weights(self, modelpath, scalerpath):
        '''
        Wrapper function to load saved keras model and scalers
        
        Parameters:
            modelpath (str): Path to saved keras model
            scalerpath (str): Path to saved scalers
        '''
        self.model.load_weights(modelpath)
        self.in_scaler = joblib.load(scalerpath + 'in_scaler.pkl')
        self.tar_scaler = joblib.load(scalerpath + 'tar_scaler.pkl')
    def save_weights(self, modelpath, scalerpath):
        '''
        Wrapper function to save keras model and scalers

        Parameters:
            modelpath (str): Path to save keras model
            scalerpath (str): Path to save scalers
        '''
        self.model.save_weights(modelpath)
        joblib.dump(self.in_scaler, scalerpath + 'in_scaler.pkl')
        joblib.dump(self.tar_scaler, scalerpath + 'tar_scaler.pkl')
    def build_model(self, units = [544, 224, 64, 80], activation = 'elu', dropout = 0.35, lr = 1e-4):
        '''
        Function to build keras model

        Parameters:
            units (list): Number of units in each layer of the model
            activation (str): Activation function to use in hidden layers
            dropout (float): Dropout rate to use in hidden layers
            lr (float): Learning rate to use in optimizer
        
        Returns:
            model (keras model): Keras model to be used for prediction (weights not initialized)
        '''
        model = ks.Sequential([ks.layers.GRU(units=units[0]),
                               ks.layers.Dense(units=units[1], activation=activation),
                               ks.layers.Dense(units=units[2], activation=activation),
                               ks.layers.Dense(units=units[3], activation=activation),
                               ks.layers.LayerNormalization(),
                               ks.layers.Dropout(dropout),
                               ks.layers.Dense(len(self.tar_keys),activation='linear')
                               ])
        model.compile(optimizer=tf.optimizers.Adamax(learning_rate=lr), loss=ksc.crps_loss)
        model.build(input_shape = (1, self.window, len(self.in_keys)))
        return model

#Plotting functions
def streamline(axes, frames, x_grid, y_grid, frame_index = None, param_index = None, u_index = None, v_index = None, data = None, cmap = 'viridis', x_extent = [0, 20], y_extent = [-35, 35], density = 2, vmin = -50, vmax = 50, draw_streamline = True, linecolor = 'k'):
    '''
    Draws heatmap from frames of the magnetosheath parameter specified by param_index at the frame specified by frame_index.
    Draws streamlines of the magnetosheath velocity/B field from frames at the frame specified by frame_index.

    Parameters:
        axes (matplotlib axes): Axes to draw the heatmap on
        frames (ndarray): Array of frames to draw from
        x_grid (ndarray): X grid to draw on
        y_grid (ndarray): Y grid to draw on
        frame_index (int): Index of frame to draw from
        param_index (int): Index of parameter to draw
        u_index (int): Index of x component of velocity/B field
        v_index (int): Index of y component of velocity/B field
        cmap (str): Colormap to use for heatmap
        x_extent (list): Range of x values to draw on
        y_extent (list): Range of y values to draw on
        density (float): Density of streamlines to draw
        vmin (float): Minimum value of heatmap
        vmax (float): Maximum value of heatmap
        draw_streamline (bool): Whether or not to draw streamlines
    Returns:
        im (matplotlib image): Image of heatmap
        stream (matplotlib streamplot): Streamplot of velocity/B field
        ax2 (matplotlib axes): Invisible axes overtop of axes containing streamplot
    '''
    if data is not None:
        im = axes.imshow(data, origin='lower', extent=[y_extent[0], y_extent[1], x_extent[0], x_extent[1]], aspect='equal', cmap=cmap, vmin=vmin, vmax=vmax)
    else:
        im = axes.imshow(frames[frame_index, :, :, param_index], origin='lower', extent=[y_extent[0], y_extent[1], x_extent[0], x_extent[1]], aspect='equal', cmap=cmap, vmin=vmin, vmax=vmax)
    axes.set_ylim(x_extent[0], x_extent[1])
    axes.set_xlim(y_extent[1], y_extent[0])
    axes.set_ylabel(r'X GSE ($R_{E}$)', fontsize = 16)
    axes.set_xlabel(r'Y GSE ($R_{E}$)', fontsize = 16)
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    axes.spines['bottom'].set_visible(False)
    axes.spines['left'].set_visible(False)

    #Create new invisible axes overtop of axes
    ax2 = axes.twinx()
    ax2.set_ylim(x_extent[0], x_extent[1])
    ax2.set_xlim(y_extent[1], y_extent[0])
    ax2.set_aspect('equal')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.set_yticks([])
    if draw_streamline:
        #Draw streamlines  in ax2
        stream = ax2.streamplot(y_grid, x_grid, frames[frame_index, :, :, v_index], frames[frame_index, :, :, u_index], color=linecolor, density=density, linewidth=1,)
    else:
        stream = None
    return im, stream, ax2

def colorbar_maker(fig, im, param_index = None, label = None, box = [0.92, 0.55, 0.015, 0.33]):
    '''
    Make a colorbar next to fig from data in im.

    Parameters:
        fig (matplotlib figure): Figure to draw colorbar on
        im (matplotlib image): Image to draw colorbar from
        param_index (int): Index of parameter displayed in imt
        label (str): Label to put on colorbar ticks
        box (list): Box to draw colorbar in
        
    Returns:
        cbar (matplotlib colorbar): Colorbar drawn on fig
    '''
    cbar_ax = fig.add_axes(box)
    cbar = fig.colorbar(im, cax=cbar_ax)
    if param_index is not None:
        labels = [r'$B_{X}$ GSE (nT)', r'$B_{X}$ GSE $\sigma$ (nT)',
                r'$B_{Y}$ GSE (nT)', r'$B_{Y}$ GSE $\sigma$ (nT)',
                r'$B_{Z}$ GSE (nT)', r'$B_{Z}$ GSE $\sigma$ (nT)',
                r'$V_{X}$ GSE (km/s)', r'$V_{X}$ GSE $\sigma$ (km/s)',
                r'$V_{Y}$ GSE (km/s)', r'$V_{Y}$ GSE $\sigma$ (km/s)',
                r'$V_{Z}$ GSE (km/s)', r'$V_{Z}$ GSE $\sigma$ (km/s)',
                r'$n_{i}$ ($cm^{-3}$)', r'$n_{i}$ $\sigma$ ($cm^{-3}$)',
                r'$T_{i\perp}$ (eV)', r'$T_{i\perp}$ $\sigma$ (eV)',
                r'$T_{i\parallel}$ (eV)', r'$T_{i\parallel}$ $\sigma$ (eV)']
        cbar.set_label(labels[param_index], fontsize = 14)
    elif label is not None:
        cbar.set_label(label, fontsize = 14)
    else:
        raise ValueError('Must specify either param_index or label')
    cbar.ax.tick_params(labelsize=14)
    return cbar

# Custom Keras functions for loss function
def crps_loss(y_true, y_pred):
    """
    Continuous rank probability score function.
    Assumes tensorflow backend.
    
    Parameters
    ----------
    y_true : tf.Tensor
        Ground truth values of predicted variable.
    y_pred : tf.Tensor
        mu and sigma^2 values of predicted distribution.
        
    Returns
    -------
    crps : tf.Tensor
        Continuous rank probability score.
    """
    # Separate the parameters into means and squared standard deviations
    mu0, sg0, mu1, sg1, mu2, sg2, mu3, sg3, mu4, sg4, mu5, sg5, mu6, sg6, mu7, sg7, mu8, sg8, y_true0, y_true1, y_true2, y_true3, y_true4, y_true5, y_true6, y_true7, y_true8 = unstack_helper(y_true, y_pred)
    
    # CRPS (assuming gaussian distribution)
    crps0 = tf.math.reduce_mean(crps_f(ep_f(y_true0, mu0), sg0))
    crps1 = tf.math.reduce_mean(crps_f(ep_f(y_true1, mu1), sg1))
    crps2 = tf.math.reduce_mean(crps_f(ep_f(y_true2, mu2), sg2))
    crps3 = tf.math.reduce_mean(crps_f(ep_f(y_true3, mu3), sg3))
    crps4 = tf.math.reduce_mean(crps_f(ep_f(y_true4, mu4), sg4))
    crps5 = tf.math.reduce_mean(crps_f(ep_f(y_true5, mu5), sg5))
    crps6 = tf.math.reduce_mean(crps_f(ep_f(y_true6, mu6), sg6))
    crps7 = tf.math.reduce_mean(crps_f(ep_f(y_true7, mu7), sg7))
    crps8 = tf.math.reduce_mean(crps_f(ep_f(y_true8, mu8), sg8))
    
    # Average the continuous rank probability scores
    crps = (crps0 + crps1 + crps2 + crps3 + crps4 + crps5 + crps6 + crps7 + crps8) / 9.0
    
    return crps

def crps_f(ep, sg):
    '''
    Helper function that calculates continuous rank probability scores
    '''
    crps = sg * ((ep/sg) * tf.math.erf((ep/(np.sqrt(2)*sg))) + tf.math.sqrt(2/np.pi) * tf.math.exp(-ep**2 / (2*sg**2)) - 1/tf.math.sqrt(np.pi))
    return crps

def ep_f(y, mu):
    '''
    Helper function that calculates epsilon for reliability score
    '''
    ep = tf.math.abs(y - mu)
    return ep

def unstack_helper(y_true, y_pred):
    '''
    Helper function that unstacks the outputs and targets
    '''
    # Separate the parameters into means and squared standard deviations
    mu0, sg0, mu1, sg1, mu2, sg2, mu3, sg3, mu4, sg4, mu5, sg5, mu6, sg6, mu7, sg7, mu8, sg8 = tf.unstack(y_pred, axis=-1)
    #Oh my god mu2, I can't believe it
    
    # Separate the ground truth into each parameter
    y_true0, y_true1, y_true2, y_true3, y_true4, y_true5, y_true6, y_true7, y_true8 = tf.unstack(y_true, axis=-1)
    
    # Add one dimension to make the right shape
    mu0 = tf.expand_dims(mu0, -1)
    sg0 = tf.expand_dims(sg0, -1)
    mu1 = tf.expand_dims(mu1, -1)
    sg1 = tf.expand_dims(sg1, -1)
    mu2 = tf.expand_dims(mu2, -1)
    sg2 = tf.expand_dims(sg2, -1)
    mu3 = tf.expand_dims(mu3, -1)
    sg3 = tf.expand_dims(sg3, -1)
    mu4 = tf.expand_dims(mu4, -1)
    sg4 = tf.expand_dims(sg4, -1)
    mu5 = tf.expand_dims(mu5, -1)
    sg5 = tf.expand_dims(sg5, -1)
    mu6 = tf.expand_dims(mu6, -1)
    sg6 = tf.expand_dims(sg6, -1)
    mu7 = tf.expand_dims(mu7, -1)
    sg7 = tf.expand_dims(sg7, -1)
    mu8 = tf.expand_dims(mu8, -1)
    sg8 = tf.expand_dims(sg8, -1)
    y_true0 = tf.expand_dims(y_true0, -1)
    y_true1 = tf.expand_dims(y_true1, -1)
    y_true2 = tf.expand_dims(y_true2, -1)
    y_true3 = tf.expand_dims(y_true3, -1)
    y_true4 = tf.expand_dims(y_true4, -1)
    y_true5 = tf.expand_dims(y_true5, -1)
    y_true6 = tf.expand_dims(y_true6, -1)
    y_true7 = tf.expand_dims(y_true7, -1)
    y_true8 = tf.expand_dims(y_true8, -1)
    return mu0, sg0, mu1, sg1, mu2, sg2, mu3, sg3, mu4, sg4, mu5, sg5, mu6, sg6, mu7, sg7, mu8, sg8, y_true0, y_true1, y_true2, y_true3, y_true4, y_true5, y_true6, y_true7, y_true8

#Analytic surface functions for bow shock and magnetopause
def jelinek_bs(y, pdyn, r0 = 15.02, l=1.17, e=6.55):
    '''
    Bow shock model from Jelinek et al 2012. Assumes GSE Z=0.

    Parameters:
        y (float): GSE Y coordinate
        pdyn (float): Solar wind dynamic pressure (nPa)
        r0 (float): Bow shock average standoff distance tuning parameter (RE)
        l (float): Lambda tuning parameter
        e (float): Epsilon tuning parameter
    '''
    bs_x = r0*(pdyn**(-1/e)) - (y**2)*(l**2)/(4*r0*(pdyn**(-1/e)))
    return bs_x

def shue_mp(theta, pdyn, bz):
    '''
    Magnetopause model from Shue et al 1998. Assumes GSE Z=0.

    Parameters:
        theta (float): Polar angle position of desired MP location (radians)
        pdyn (float): Solar wind dynamic pressure (nPa)
        bz (float): IMF Bz (nT)
    '''
    r0 = (10.22 + 1.29*np.tanh(0.184*(bz + 8.14)))*(pdyn**(-1/6.6))
    a1 = (0.58 - 0.007*bz) * (1 + 0.024*np.log(pdyn))
    rmp = r0*(2/(1 + np.cos(theta)))**a1
    return rmp

def shue_mp_interp(y, pdyn, bz, theta_extent = [-np.pi/2, np.pi/2], gridsize = 0.01):
    '''
    Magnetopause model from Shue et al 1998, interpolated so GSE X can be specified from GSE Y. Assumes GSE Z=0.

    Parameters:
        y (float): GSE Y coordinate
        pdyn (float): Solar wind dynamic pressure (nPa)
        bz (float): IMF Bz (nT)
        theta_extent (list): Polar angle extent of the grid (radians)
    '''
    theta = np.arange(theta_extent[0], theta_extent[1], 0.01)
    r0 = (10.22 + 1.29*np.tanh(0.184*(bz + 8.14)))*(pdyn**(-1/6.6))
    a1 = (0.58 - 0.007*bz) * (1 + 0.024*np.log(pdyn))
    rmp = r0*(2/(1 + np.cos(theta)))**a1
    x_mp = rmp*np.cos(theta)
    y_mp = rmp*np.sin(theta)
    f = spi.interp1d(y_mp, x_mp, fill_value=np.nan, bounds_error=False)
    x = f(y)
    return x