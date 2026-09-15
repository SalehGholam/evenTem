"""
copyright University of Antwerp 2025

author: Arno Annys
"""
import os
base_dir = os.path.dirname(__file__)
import sys
sys.path.append(os.path.join(base_dir,'binaries'))
import eventem
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", message="The value of the smallest subnormal for <class 'numpy.float64'> type is zero.")

def safelog(x):
    return np.log(1+1000*x/np.max(x))

class Pacbed(eventem.Pacbed):
    """
    Position avg CBED processor

    """
    def __init__(self, nx,ny,repetitions,filename):
        """
        Instantiate a Position averaged convergend beam electron diffraction (PACBED) processor

        Parameters
        ----------
        nx : int
            number of pixels in x direction
        ny : int
            number of pixels in y direction
        repetitions : int
            number of complete scans in the dataset
        filename : str
            path to the dataset file

        Returns
        -------
        None.

        """
        super().__init__(repetitions)
        self.b_cumulative = True
        self.nx = nx
        self.ny = ny    
        self.set_file(filename)

    # @property
    # def nx(self):
    #     """
    #     int : number of pixels in x direction
    #     """
    #     return super().nx
    
    # @nx.setter
    # def nx(self, value):
    #     self.nx = value

    # @property
    # def ny(self):
    #     """
    #     int : number of pixels in y direction
    #     """
    #     return super().ny
    
    # @ny.setter
    # def ny(self, value):
    #     self.ny = value
        
    
    @property
    def DwellTime(self):
        """
        int : dwell time
        """
        return super().dt
    
    @DwellTime.setter
    def DwellTime(self, value):
        self.dt = value

    @property
    def DetectorSize(self):
        """
        int : size of the detector
        """
        return super().detector_size
    
    @DetectorSize.setter
    def DetectorSize(self, value):
        self.detector_size = value


    def Run(self):
        """
        Run the virtual STEM reconstruction
        """
        super().run()

    @property
    def image(self):
        """
        2D numpy array : reconstructed image [nx,ny]
        """
        return np.array(super().Pacbed_image).reshape(self.DetectorSize,self.DetectorSize)
    

    def PlotImage(self,log=False):
        """
        Plot the reconstructed image

        Parameters
        ----------
        log : bool
            plot the image in log scale
        """
        if log:
            fig, ax = plt.subplots(1,1,figsize=(10,10))
            ax.imshow(safelog(self.image),cmap='gnuplot')
        else:
            fig, ax = plt.subplots(1,1,figsize=(10,10))
            ax.imshow(self.image,cmap='gnuplot')
        ax.axis('off')

    def GetCOM(self):
        """
        Get the center of mass of the PACBED image

        Returns
        -------
        tuple : x,y coordinates of the center of mass

        """

        X,Y = np.meshgrid(np.arange(self.DetectorSize), np.arange(self.DetectorSize))
        X = X.flatten()
        Y = Y.flatten()
        I = self.image.astype(float).flatten()
        COM = np.array([np.sum(X*I)/np.sum(I), np.sum(Y*I)/np.sum(I)])
        return COM

    @property
    def Decluster(self):
        """
        bool : charge-weighted, declustered PACBED (CHEETAH/.tpx3 only). Off by
        default -- when False, behavior is unchanged from before this feature
        existed.
        """
        return super().decluster

    @Decluster.setter
    def Decluster(self, value):
        self.decluster = value

    @property
    def Dtime(self):
        """
        int : declustering time window (same meaning/default as Roi.Dtime)
        """
        return super().dtime

    @Dtime.setter
    def Dtime(self, value):
        self.dtime = value

    @property
    def Dspace(self):
        """
        int : declustering space window (same meaning/default as Roi.Dspace)
        """
        return super().dspace

    @Dspace.setter
    def Dspace(self, value):
        self.dspace = value

    @property
    def ClusterRange(self):
        """
        int : declustering search range (same meaning/default as Roi.ClusterRange)
        """
        return super().cluster_range

    @ClusterRange.setter
    def ClusterRange(self, value):
        self.cluster_range = value

    @property
    def TotPerElectron(self):
        """
        float : calibration constant (typical single-electron cluster ToT-sum) used
        to resolve each cluster's electron count when Decluster=True. No safe default
        -- must be set from your own cluster ToT-sum histogram (e.g. Roi's, if
        acquired under the same beam/detector conditions).
        """
        return super().tot_per_electron

    @TotPerElectron.setter
    def TotPerElectron(self, value):
        self.tot_per_electron = value

    @property
    def ElectronCountLut(self):
        """
        2D nested list/array of int : alternative to TotPerElectron -- resolves each
        cluster's electron count from a (cluster size, ToT) map instead of a single
        ToT/TotPerElectron ratio. Row = cluster size, column = ToT, matching
        ClustersizeTotHistogram's own layout. Empty by default (TotPerElectron stays
        the default resolver); set directly, or set ElectronCountLutFile to load a
        saved template instead. Both options remain available side by side.
        """
        return np.array(super().electron_count_lut)

    @ElectronCountLut.setter
    def ElectronCountLut(self, value):
        self.electron_count_lut = value

    @property
    def ElectronCountLutFile(self):
        """
        str : path to a plain-text electron-count template file (see
        ElectronCountLut) -- loaded into ElectronCountLut at the start of Run() if
        non-empty. A reusable calibration artifact: valid for reuse as long as the
        beam energy / detector setup it was built from hasn't changed. Empty by
        default (feature off).
        """
        return super().electron_count_lut_file

    @ElectronCountLutFile.setter
    def ElectronCountLutFile(self, value):
        self.electron_count_lut_file = value

    @property
    def ClustersizeHistogram(self):
        """
        1D numpy array : histogram of resolved cluster sizes, from the most recent
        declustered Run().
        """
        return np.array(super().clustersize_histogram)

    @property
    def EnergyHistogram(self):
        """
        1D numpy array : histogram of resolved clusters' total summed ToT, from the
        most recent declustered Run().
        """
        return np.array(super().energy_histogram)

    @property
    def ClustersizeTotHistogram(self):
        """
        2D numpy array, shape (50, 4096) : joint histogram of resolved cluster size
        (rows) vs. total summed ToT (columns), from the most recent declustered Run().
        """
        return np.array(super().clustersize_tot_histogram)

    def SetScanMask(self, mask):
        """
        Restrict PACBED accumulation to scan positions selected by a mask (e.g. a
        segmented particle/vacuum shape), instead of the whole scan. Works with or
        without Decluster=True -- when both are set, a cluster's SEED hit position
        decides whether it counts.

        Parameters
        ----------
        mask : 2D array-like of int, shape (ny, nx), or a flat nx*ny 1D array
            Nonzero = included. Row-major, index = ry*nx+rx (no y-flip) -- same
            scan-grid convention as everything else in this file (see set_offsets,
            image reshaping elsewhere).
        """
        flat = np.asarray(mask, dtype=np.int32).reshape(-1)
        super().set_scan_mask(flat)


class vSTEM(eventem.vSTEM):
    """
    virtual STEM processor

    """
    def __init__(self, nx,ny,repetitions,filename):
        """
        Instantiate a virtual STEM processor

        Parameters
        ----------
        nx : int
            number of pixels in x direction
        ny : int
            number of pixels in y direction
        repetitions : int
            number of complete scans in the dataset
        filename : str
            path to the dataset file

        Returns
        -------
        None.

        """
        super().__init__(repetitions)
        self.b_cumulative = True
        self.nx = nx
        self.ny = ny    
        self.set_file(filename)

    # @property
    # def nx(self):
    #     """
    #     int : number of pixels in x direction
    #     """
    #     return super().nx
    
    # @nx.setter
    # def nx(self, value):
    #     self.nx = value

    # @property
    # def ny(self):
    #     """
    #     int : number of pixels in y direction
    #     """
    #     return super().ny
    
    # @ny.setter
    # def ny(self, value):
    #     self.ny = value
        
    @property
    def DwellTime(self):
        """
        int : dwell time
        """
        return super().dt
    
    @DwellTime.setter
    def DwellTime(self, value):
        self.dt = value

    @property
    def DetectorSize(self):
        """
        int : size of the detector
        """
        return super().detector_size
    
    @DetectorSize.setter
    def DetectorSize(self, value):
        self.detector_size = value

    @property
    def InnerRadia(self):
        """
        list of int : inner radii of the annular detectors
        """
        return super().inner_radia
    
    @InnerRadia.setter
    def InnerRadia(self, value):
        self.inner_radia = value

    @property
    def OuterRadia(self):
        """
        list of int : outer radii of the annular detectors
        """
        return self.outer_radia
    
    @OuterRadia.setter
    def OuterRadia(self, value):
        self.outer_radia = value

    @property
    def Centers(self):
        """
        list of tuples : centers of the annular detectors
        """
        return self.offsets
    
    @Centers.setter
    def Centers(self, value):
        """
        list of tuples : centers of the annular detectors
        """
        self.set_offsets(value)

    def GetDetector(self):
        """
        Get the detector image

        Returns
        -------
        2D numpy array
           
        """
        return np.array(self.get_detector()).reshape(self.DetectorSize,self.DetectorSize)
    
    def PlotDetector(self,pacbed=None,log=False):
        """
        Plot the detector image
        """
        detector = self.GetDetector()
        if pacbed is None:
            fig, ax = plt.subplots(1,1,figsize=(10,10))
            ax.imshow(detector)
        elif isinstance(pacbed, Pacbed):
            if pacbed.image.shape == (self.DetectorSize,self.DetectorSize):
                self.incircle=plt.Circle(self.offsets[0],self.inner_radia[0], fill=False, color='red', linewidth=2)
                self.outcircle=plt.Circle(self.offsets[0],self.outer_radia[0], fill=False, color='red', linewidth=2)

                fig, ax = plt.subplots(1,1,figsize=(10,10))
                if log:
                    ax.imshow(safelog(pacbed.image),cmap='gray')
                else:
                    ax.imshow(pacbed.image,cmap='gray')
                ax.imshow(detector,alpha=0.2,cmap='Reds')
                ax.add_patch(self.incircle)
                ax.add_patch(self.outcircle)
                ax.axis('off')
        else:
            raise ValueError("pacbed should be a Pacbed object with the same detector size as the vSTEM object, or None")
        
    def MakeDPCDetector(self,inner_radius,outer_radius,N_segments,segment_weights,center,rotation):
        rotation_rad = rotation*np.pi/180
        _mask = np.zeros((self.DetectorSize, self.DetectorSize))
        for i in range(self.DetectorSize):
            for j in range(self.DetectorSize):
                if (i-center[0])**2 + (j-center[1])**2 < outer_radius**2:
                    if (i-center[0])**2 + (j-center[1])**2 > inner_radius**2:
                        angle = np.arctan2(i-center[0], j-center[1])
                        angle += rotation_rad
                        if angle < 0:
                            angle += 2*np.pi
                        rotation_rad = rotation*np.pi/180
                        segment = int(N_segments*angle/(2*np.pi))
                        _mask[i, j] = segment_weights[segment]
        
        _mask = _mask.flatten().astype(int)
        self.set_detector_mask(_mask)

    def Run(self):
        """
        Run the virtual STEM reconstruction
        """
        super().run()

    @property
    def image(self):
        """
        2D numpy array : reconstructed image [nx,ny]
        """
        return np.array(super().vSTEM_image).reshape(super().ny,super().nx)
    
    @property
    def image_stack(self):
        """
        3D numpy array : reconstructed image stack [nx,ny,repetitions]
        """
        return np.array(super().vSTEM_stack).reshape(super().repetitions+1,super().ny,super().nx)[:-1,:,:]

    @property
    def Decluster(self):
        """
        bool : charge-weighted, declustered dose counting (CHEETAH/.tpx3 only,
        single-annulus detector only). Off by default -- when False, behavior is
        unchanged from before this feature existed.
        """
        return super().decluster

    @Decluster.setter
    def Decluster(self, value):
        self.decluster = value

    @property
    def Dtime(self):
        """
        int : declustering time window (same meaning/default as Roi.Dtime)
        """
        return super().dtime

    @Dtime.setter
    def Dtime(self, value):
        self.dtime = value

    @property
    def Dspace(self):
        """
        int : declustering space window (same meaning/default as Roi.Dspace)
        """
        return super().dspace

    @Dspace.setter
    def Dspace(self, value):
        self.dspace = value

    @property
    def ClusterRange(self):
        """
        int : declustering search range (same meaning/default as Roi.ClusterRange)
        """
        return super().cluster_range

    @ClusterRange.setter
    def ClusterRange(self, value):
        self.cluster_range = value

    @property
    def TotPerElectron(self):
        """
        float : calibration constant (typical single-electron cluster ToT-sum) used
        to resolve each cluster's electron count when Decluster=True. No safe default
        -- must be set from your own cluster ToT-sum histogram (e.g. Roi's, if
        acquired under the same beam/detector conditions).
        """
        return super().tot_per_electron

    @TotPerElectron.setter
    def TotPerElectron(self, value):
        self.tot_per_electron = value

    @property
    def ElectronCountLut(self):
        """
        2D nested list/array of int : alternative to TotPerElectron -- resolves each
        cluster's electron count from a (cluster size, ToT) map instead of a single
        ToT/TotPerElectron ratio. Row = cluster size, column = ToT, matching
        ClustersizeTotHistogram's own layout. Empty by default (TotPerElectron stays
        the default resolver); set directly, or set ElectronCountLutFile to load a
        saved template instead. Both options remain available side by side.
        """
        return np.array(super().electron_count_lut)

    @ElectronCountLut.setter
    def ElectronCountLut(self, value):
        self.electron_count_lut = value

    @property
    def ElectronCountLutFile(self):
        """
        str : path to a plain-text electron-count template file (see
        ElectronCountLut) -- loaded into ElectronCountLut at the start of Run() if
        non-empty. A reusable calibration artifact: valid for reuse as long as the
        beam energy / detector setup it was built from hasn't changed. Empty by
        default (feature off).
        """
        return super().electron_count_lut_file

    @ElectronCountLutFile.setter
    def ElectronCountLutFile(self, value):
        self.electron_count_lut_file = value

    @property
    def ClustersizeHistogram(self):
        """
        1D numpy array : histogram of resolved cluster sizes, from the most recent
        declustered Run().
        """
        return np.array(super().clustersize_histogram)

    @property
    def EnergyHistogram(self):
        """
        1D numpy array : histogram of resolved clusters' total summed ToT, from the
        most recent declustered Run().
        """
        return np.array(super().energy_histogram)

    @property
    def ClustersizeTotHistogram(self):
        """
        2D numpy array, shape (50, 4096) : joint histogram of resolved cluster size
        (rows) vs. total summed ToT (columns), from the most recent declustered Run().
        """
        return np.array(super().clustersize_tot_histogram)

    def PlotImage(self):
        """
        Plot the reconstructed image
        """
        fig, ax = plt.subplots(1,1,figsize=(10,10))
        ax.imshow(self.image,cmap='gray')
        ax.axis('off')


class Var(eventem.Var):
    """
    Variance processor

    """
    def __init__(self, nx,ny,repetitions,filename):
        """
        Instantiate a variance processor

        Parameters
        ----------
        nx : int
            number of pixels in x direction
        ny : int
            number of pixels in y direction
        repetitions : int
            number of complete scans in the dataset
        filename : str
            path to the dataset file

        Returns
        -------
        None.

        """
        super().__init__(repetitions)
        self.b_cumulative = True
        self.nx = nx
        self.ny = ny    
        self.set_file(filename)

    # @property
    # def nx(self):
    #     """
    #     int : number of pixels in x direction
    #     """
    #     return super().nx
    
    # @nx.setter
    # def nx(self, value):
    #     self.nx = value

    # @property
    # def ny(self):
    #     """
    #     int : number of pixels in y direction
    #     """
    #     return super().ny
    
    # @ny.setter
    # def ny(self, value):
    #     self.ny = value
        
    @property
    def DwellTime(self):
        """
        int : dwell time
        """
        return super().dt
    
    @DwellTime.setter
    def DwellTime(self, value):
        self.dt = value

    @property
    def DetectorSize(self):
        """
        int : size of the detector
        """
        return super().detector_size
    
    @DetectorSize.setter
    def DetectorSize(self, value):
        self.detector_size = value


    @property
    def Center(self):
        """
        tuple : 
        """
        return self.offset
    
    @Center.setter
    def Center(self, value):
        """
        tuple: centers 
        """
        super().set_offset(value)

    def Run(self):
        """
        Run the virtual STEM reconstruction
        """
        super().run()

    @property
    def image(self):
        """
        2D numpy array : reconstructed image [nx,ny]
        """
        return np.array(super().Var_image).reshape(super().ny,super().nx)
    
    @property
    def InnerRadius(self):
        """
        int : inner radius of the annular detector
        """
        return super().inner_radius
    
    @InnerRadius.setter
    def InnerRadius(self, value):
        """
        int : inner radius of the annular detector
        """
        self.inner_radius = value

    @property
    def OuterRadius(self):
        """
        int : outer radius of the annular detector
        """
        return super().outer_radius
    
    @OuterRadius.setter
    def OuterRadius(self, value):
        """
        int : outer radius of the annular detector
        """
        self.outer_radius = value

    @property
    def Decluster(self):
        """
        bool : charge-weighted, declustered variance imaging (CHEETAH/.tpx3 only).
        Off by default -- when False, behavior is unchanged from before this feature
        existed.
        """
        return super().decluster

    @Decluster.setter
    def Decluster(self, value):
        self.decluster = value

    @property
    def Dtime(self):
        """
        int : declustering time window (same meaning/default as Roi.Dtime)
        """
        return super().dtime

    @Dtime.setter
    def Dtime(self, value):
        self.dtime = value

    @property
    def Dspace(self):
        """
        int : declustering space window (same meaning/default as Roi.Dspace)
        """
        return super().dspace

    @Dspace.setter
    def Dspace(self, value):
        self.dspace = value

    @property
    def ClusterRange(self):
        """
        int : declustering search range (same meaning/default as Roi.ClusterRange)
        """
        return super().cluster_range

    @ClusterRange.setter
    def ClusterRange(self, value):
        self.cluster_range = value

    @property
    def TotPerElectron(self):
        """
        float : calibration constant (typical single-electron cluster ToT-sum) used
        to resolve each cluster's electron count when Decluster=True. No safe default
        -- must be set from your own cluster ToT-sum histogram (e.g. Roi's, if
        acquired under the same beam/detector conditions).
        """
        return super().tot_per_electron

    @TotPerElectron.setter
    def TotPerElectron(self, value):
        self.tot_per_electron = value

    @property
    def ElectronCountLut(self):
        """
        2D nested list/array of int : alternative to TotPerElectron -- resolves each
        cluster's electron count from a (cluster size, ToT) map instead of a single
        ToT/TotPerElectron ratio. Row = cluster size, column = ToT, matching
        ClustersizeTotHistogram's own layout. Empty by default (TotPerElectron stays
        the default resolver); set directly, or set ElectronCountLutFile to load a
        saved template instead. Both options remain available side by side.
        """
        return np.array(super().electron_count_lut)

    @ElectronCountLut.setter
    def ElectronCountLut(self, value):
        self.electron_count_lut = value

    @property
    def ElectronCountLutFile(self):
        """
        str : path to a plain-text electron-count template file (see
        ElectronCountLut) -- loaded into ElectronCountLut at the start of Run() if
        non-empty. A reusable calibration artifact: valid for reuse as long as the
        beam energy / detector setup it was built from hasn't changed. Empty by
        default (feature off).
        """
        return super().electron_count_lut_file

    @ElectronCountLutFile.setter
    def ElectronCountLutFile(self, value):
        self.electron_count_lut_file = value

    @property
    def ClustersizeHistogram(self):
        """
        1D numpy array : histogram of resolved cluster sizes, from the most recent
        declustered Run().
        """
        return np.array(super().clustersize_histogram)

    @property
    def EnergyHistogram(self):
        """
        1D numpy array : histogram of resolved clusters' total summed ToT, from the
        most recent declustered Run().
        """
        return np.array(super().energy_histogram)

    @property
    def ClustersizeTotHistogram(self):
        """
        2D numpy array, shape (50, 4096) : joint histogram of resolved cluster size
        (rows) vs. total summed ToT (columns), from the most recent declustered Run().
        """
        return np.array(super().clustersize_tot_histogram)

    def PlotImage(self):
        """
        Plot the reconstructed image
        """
        fig, ax = plt.subplots(1,1,figsize=(10,10))
        ax.imshow(self.image,cmap='gray')
        ax.axis('off')


class Ricom(eventem.Ricom):
    """
    Ricom processor
    """
    def __init__(self, nx,ny,repetitions,filename):
        """
        Instantiate a real-time integrated center-of-mass processor

        Parameters
        ----------
        nx : int
            number of pixels in x direction
        ny : int
            number of pixels in y direction
        repetitions : int
            number of complete scans in the dataset
        filename : str
            path to the dataset file

        Returns
        -------
        None.

        """
        super().__init__(repetitions)
        self.b_cumulative = True
        self.nx = nx
        self.ny = ny    
        self.set_file(filename)
        self.n_threads = 8

    # @property
    # def nx(self):
    #     """
    #     int : number of pixels in x direction
    #     """
    #     return super().nx

    # @nx.setter
    # def nx(self, value):
    #     self.nx = value

    # @property
    # def ny(self):
    #     """
    #     int : number of pixels in y direction
    #     """
    #     return super().ny

    # @ny.setter
    # def ny(self, value):
    #     self.ny = value
        

    @property
    def DwellTime(self):
        """
        int : dwell time
        """
        return super().dt
    
    @DwellTime.setter
    def DwellTime(self, value):
        self.dt = value

    @property
    def DetectorSize(self):
        """
        int : size of the detector
        """
        return super().detector_size

    @DetectorSize.setter
    def DetectorSize(self, value):
        self.detector_size = value


    def Run(self):
        """
        Run the Ricom reconstruction
        """
        self.run()

    def SetKernel(self, kernelsize,rotation=0):
        """
        Set the kernel for the Ricom reconstruction

        Parameters
        ----------
        kernelsize : int 
            size of the kernel
        rotation : int
            rotation of the real and momentum space

        Returns
        -------
        None.

        """
        self.KS = kernelsize
        super().set_kernel(kernelsize,rotation)

    @property
    def CoMx_image(self):
        """
        2D numpy arrays : center of mass image X
        """
        return np.array(self.comx_image).reshape(self.ny,self.nx)
    
    @property
    def CoMy_image(self):
        """
        2D numpy arrays : center of mass image Y
        """
        return np.array(self.comy_image).reshape(self.ny,self.nx)
    
    @property
    def CoM(self):
        """
        tuple : center of mass
        """
        return self.offset
    
    @CoM.setter
    def CoM(self,value):
        self.set_offset(value)

    @property
    def image(self):
        """
        2D numpy array : reconstructed image [nx,ny]
        """
        return np.array(self.ricom_image).reshape(self.ny,self.nx)

    @property
    def image_stack(self):
        """
        3D numpy array : reconstructed image stack [nx,ny,repetitions]
        """
        return np.array(self.ricom_stack).reshape(self.repetitions+1,self.ny,self.nx)[:-1,:,:]

    def PlotImage(self,crop_kernel=True):
        """
        Plot the reconstructed image

        Parameters
        ----------
        crop_kernel : bool
            crop a border from the image that is the size of the kernel
        """
        if crop_kernel:
            border = self.KS
        else:
            border = 0
        fig, ax = plt.subplots(1,1,figsize=(10,10))
        ax.imshow(self.image[border:-border,border:-border])
        ax.axis('off')


class Electron(eventem.Electron):
    """
    Convert to electron file format

    """
    def __init__(self,nx,ny,repetitions,filename):
        """
        Instantiate 

        Parameters
        ----------
        nx : int
            number of pixels in x direction
        ny : int
            number of pixels in y direction
        repetitions : int
            number of complete scans in the dataset
        filename : str
            path to the dataset file

        Returns
        -------
        None.

        """
        super().__init__(repetitions)
        self.b_cumulative = True
        self.nx = nx
        self.ny = ny    
        self.set_file(filename)

    # @property
    # def nx(self):
    #     """
    #     int : number of pixels in x direction
    #     """
    #     return super().nx
    
    # @nx.setter
    # def nx(self, value):
    #     self.nx = value

    # @property
    # def ny(self):
    #     """
    #     int : number of pixels in y direction
    #     """
    #     return super().ny
    
    # @ny.setter
    # def ny(self, value):
    #     self.ny = value

    @property
    def DetectorSize(self):
        """
        int : size of the detector
        """
        return super().detector_size
    
    @DetectorSize.setter
    def DetectorSize(self, value):
        self.detector_size = value
        
    @property
    def DwellTime(self):
        """
        int : dwell time
        """
        return super().dt
    
    @DwellTime.setter
    def DwellTime(self, value):
        self.dt = value


    @property
    def xCrop(self):
        """
        int : x crop
        """
        return super().x_crop
    
    @xCrop.setter
    def xCrop(self, value):
        self.x_crop = value

    @property
    def yCrop(self):
        """
        int : y crop
        """
        return super().y_crop
    
    @yCrop.setter
    def yCrop(self, value):
        self.y_crop = value

    @property
    def ScanBin(self):
        """
        int : scan bin
        """
        return super().scan_bin
    
    @ScanBin.setter
    def ScanBin(self, value):
        self.scan_bin = value

    @property
    def DetectorBin(self):
        """
        int : detector bin
        """
        return super().detector_bin
    
    @DetectorBin.setter
    def DetectorBin(self, value):
        self.detector_bin = value

    @property
    def DropRate(self):
        """
        int : drop rate
        """
        return super().drop_rate
    
    @DropRate.setter
    def DropRate(self, value):
        self.drop_rate = value

    @property
    def ClusterRange(self):
        """
        int : cluster range
        """
        return super().cluster_range
    
    @ClusterRange.setter
    def ClusterRange(self, value):
        self.cluster_range = value

    @property
    def Dspace(self):
        """
        int : dspace
        """
        return super().dspace
    
    @Dspace.setter
    def Dspace(self, value):
        self.dspace = value

    @property
    def Dtime(self):
        """
        int : dtime
        """
        return super().dtime
    
    @Dtime.setter
    def Dtime(self, value):
        self.dtime = value

    @property
    def Decluster(self):
        """
        bool: decluster
        """
        return super().decluster
    
    @Decluster.setter
    def Decluster(self, value):
        self.decluster = value

    @property
    def Nthreads(self):
        """
        int : number of threads
        """
        return super().n_threads
    
    @Nthreads.setter
    def Nthreads(self, value):
        self.n_threads = value


    def Run(self):
        """
        Run
        """
        super().run()


class Roi(eventem.Roi):
    """
    Roi processor
    """
    def __init__(self, nx,ny,repetitions,filename,extract_4D=False):
        """
        Instantiate a ROI processor

        Parameters
        ----------
        nx : int
            number of pixels in x direction
        ny : int
            number of pixels in y direction
        repetitions : int
            number of complete scans in the dataset
        filename : str
            path to the dataset file

        Returns
        -------
        None.

        """
        super().__init__(repetitions,extract_4D)
        self.extract_4D = extract_4D
        self.b_cumulative = True
        self.nx = nx
        self.ny = ny    
        self.width = nx
        self.height = ny
        self.set_file(filename)
        self.set_roi(x=0,y=0,width=nx,height=ny) #default ROI is full image

    # @property
    # def nx(self):
    #     """
    #     int : number of pixels in x direction
    #     """
    #     return super().nx
    
    # @nx.setter
    # def nx(self, value):
    #     self.nx = value

    # @property
    # def ny(self):
    #     """
    #     int : number of pixels in y direction
    #     """
    #     return super().ny
    
    # @ny.setter
    # def ny(self, value):
    #     self.ny = value
        
    @property
    def DwellTime(self):
        """
        int : dwell time
        """
        return super().dt
    
    @DwellTime.setter
    def DwellTime(self, value):
        self.dt = value

    @property
    def ROI(self):
        """
        tuple : ROI coordinates as [x_origin,y_origin,width,height]
        """
        return np.array(super().get_roi())
    
    @ROI.setter
    def ROI(self, value):
        self.width = value[2]
        self.height = value[3]
        self.set_roi(x=value[0],y=value[1],width=value[2],height=value[3])

    def SetROI(self,x_origin,y_origin,width,height):
        """
        Set the ROI

        Parameters
        ----------
        x_origin : int
            x origin
        y_origin : int
            y origin
        width : int
            width of the ROI
        height : int
            height of the ROI

        Returns
        -------
        None.

        """
        self.width = width
        self.height = height
        self.set_roi(x=x_origin,y=y_origin,width=width,height=height)

    def SetROIMask(self,masks):
        """
        Set the ROI

        Parameters
        ----------
        x_origin : int
            x origin
        y_origin : int
            y origin
        width : int
            width of the ROI
        height : int
            height of the ROI

        Returns
        -------
        None.

        """
        self.width = self.ny #carefull with the convention here
        self.height = self.nx  #carefull with up the convention here
        mask_list = [masks[i].flatten().astype(int) for i in range(len(masks))]
        self.set_roi_mask(mask_list)

    @property
    def ScanImage(self):
        """
        2D numpy array : scan image
        """
        return np.array(super().Roi_scan_image).reshape(self.width,self.height)
    
    @property
    def DiffractionPattern(self):
        """
        2D numpy array : diffraction pattern
        """
        return np.array(super().Roi_diffraction_pattern).reshape(self.DetectorSize,self.DetectorSize)
    
    @property
    def ScanImageStack(self):
        """
        3D numpy array : scan image stack
        """
        return np.array(super().Roi_scan_image_stack).reshape(self.repetitions+1,self.width,self.height)[:-1,:,:]
    
    @property
    def DiffractionPatternStack(self):
        """
        3D numpy array : diffraction pattern stack
        """
        return np.array(super().Roi_diffraction_pattern_stack).reshape(self.repetitions+1,self.DetectorSize,self.DetectorSize)[:-1,:,:]
    
    @property
    def Roi4D(self):
        """
        4D numpy array : 4D dataset
        """
        if self.extract_4D:
            return np.array(super().get_4D(),dtype=np.uint8)
        else:
            print("4D ROI is only extracted when extract_4D is set to True")
            return None
    
    @property
    def DetectorSize(self):
        """
        int : size of the detector
        """
        return super().detector_size
    
    @DetectorSize.setter
    def DetectorSize(self, value):
        self.detector_size = value

    @property
    def DetectorBin(self):
        """
        int : detector bin
        """
        return super().det_bin

    @DetectorBin.setter
    def DetectorBin(self, value):
        self.det_bin = value

    @property
    def Decluster(self):
        """
        bool : if True, group raw pixel activations into physical-electron clusters
        (same windowed dspace/dtime/cluster_range grouping as the Electron processor)
        before counting into the ROI outputs, crediting each cluster's ToT-weighted
        centroid with the number of electrons resolved from its total charge (see
        TotPerElectron). Currently only supported for .tpx3 (CHEETAH) files. Off by
        default, in which case every raw pixel activation is counted as one electron,
        same as before this option existed.
        """
        return super().decluster

    @Decluster.setter
    def Decluster(self, value):
        self.decluster = value

    @property
    def Dtime(self):
        """
        int : maximum time (in ToA clock ticks) between two pixel activations for them
        to be merged into the same cluster. Only used when Decluster is True.
        """
        return super().dtime

    @Dtime.setter
    def Dtime(self, value):
        self.dtime = value

    @property
    def Dspace(self):
        """
        int : maximum distance (in pixels, x and y) between two pixel activations for
        them to be merged into the same cluster. Only used when Decluster is True.
        """
        return super().dspace

    @Dspace.setter
    def Dspace(self, value):
        self.dspace = value

    @property
    def ClusterRange(self):
        """
        int : how many subsequent raw hits are checked as merge candidates for a given
        cluster seed. Only used when Decluster is True.
        """
        return super().cluster_range

    @ClusterRange.setter
    def ClusterRange(self, value):
        self.cluster_range = value

    @property
    def TotPerElectron(self):
        """
        float : calibration constant -- the typical total summed ToT (charge) of a
        single electron's cluster, at your current beam energy / detector threshold
        setting. Read this off your own cluster (ToT-sum vs. hit-count) histogram --
        e.g. the ToT-sum value at the center of the "1 electron" band. A cluster's
        resolved electron count is round(cluster_tot_sum / TotPerElectron); clusters
        well below this (noise/X-rays) resolve to 0, clusters near 2x/3x/... resolve
        to genuine multi-electron pile-up. Must be set to a positive value before
        Decluster can be used -- there is no safe default.
        """
        return super().tot_per_electron

    @TotPerElectron.setter
    def TotPerElectron(self, value):
        self.tot_per_electron = value

    @property
    def ElectronCountLut(self):
        """
        2D nested list/array of int : alternative to TotPerElectron -- resolves each
        cluster's electron count from a (cluster size, ToT) map instead of a single
        ToT/TotPerElectron ratio. Row = cluster size, column = ToT, matching
        ClustersizeTotHistogram's own layout. Empty by default (TotPerElectron stays
        the default resolver); set directly, or set ElectronCountLutFile to load a
        saved template instead. Both options remain available side by side.
        """
        return np.array(super().electron_count_lut)

    @ElectronCountLut.setter
    def ElectronCountLut(self, value):
        self.electron_count_lut = value

    @property
    def ElectronCountLutFile(self):
        """
        str : path to a plain-text electron-count template file (see
        ElectronCountLut) -- loaded into ElectronCountLut at the start of Run() if
        non-empty. A reusable calibration artifact: valid for reuse as long as the
        beam energy / detector setup it was built from hasn't changed. Empty by
        default (feature off).
        """
        return super().electron_count_lut_file

    @ElectronCountLutFile.setter
    def ElectronCountLutFile(self, value):
        self.electron_count_lut_file = value

    @property
    def ClustersizeHistogram(self):
        """
        1D numpy array : histogram of resolved cluster sizes (number of raw pixel hits
        merged per cluster), from the most recent declustered Run(). Useful for
        sanity-checking against your own calibration plot.
        """
        return np.array(super().clustersize_histogram)

    @property
    def EnergyHistogram(self):
        """
        1D numpy array : histogram of resolved clusters' total summed ToT, from the
        most recent declustered Run(). Useful for sanity-checking TotPerElectron
        against your own calibration plot.
        """
        return np.array(super().energy_histogram)

    @property
    def ClustersizeTotHistogram(self):
        """
        2D numpy array, shape (50, 4096) : joint histogram of resolved cluster size
        (rows) vs. total summed ToT (columns), from the most recent declustered Run().
        This is the "hits vs. summed ToT" 2D calibration plot -- reproduces the kind of
        plot used to visually pick TotPerElectron by looking for single- vs.
        multi-electron pile-up bands.
        """
        return np.array(super().clustersize_tot_histogram)

    def Run(self):
        """
        Run the ROI reconstruction
        """
        if self.extract_4D:
            print(f"extracting 4D sub-dataset that requires {self.width*self.height*(self.DetectorSize/self.DetectorBin)**2/1e9:.2f} GB of RAM")
        super().run()
    


def make_hyperspy_compatible(src_path, dst_path=None, dataset_key="4D",
                              nav_axis_names=("y", "x"), sig_axis_names=("ky", "kx")):
    """
    Converts a FourD-written HDF5 file (run with `SaveMetadata = False`, so it
    contains only a top-level `dataset_key` 4D array and nothing else) into a file
    that loads directly via `hyperspy.api.load(path, lazy=True)` -- no extra
    kwargs, no reader ambiguity.

    Two things are required for that, neither achieved by dataset-pruning alone:
    1. HyperSpy's own minimal HDF5 schema: two root attributes, an
       "Experiments/<name>" group, one "axis-i" group per array dimension (a
       handful of scalar attributes each: name/navigate/size/scale/offset/units/
       type), the array itself relocated to ".../data", and three small
       placeholder groups HyperSpy's reader unconditionally looks for (metadata,
       original_metadata, learning_results). Confirmed by round-tripping a real
       signal through HyperSpy's own writer and inspecting the result -- this
       function reproduces exactly that, moving the existing dataset in place with
       `h5py.Group.move()` (no data copy).
    2. A ".hspy" file extension. `hs.load()`'s reader dispatch is purely
       extension-based -- ".hdf5"/".h5" are claimed by multiple plugins (Delmic,
       HSPY, USID) regardless of whether the file content is actually valid for
       any of them, and `hs.load()` refuses to guess between them. ".hspy" is
       HyperSpy's own, unambiguous extension.

    Parameters
    ----------
    src_path : str
        Path to the existing FourD-written HDF5 file (its own extension, e.g.
        ".hdf5", is fine -- it's read here, not modified).
    dst_path : str, optional
        Output path. Defaults to `src_path` with its extension replaced by
        ".hspy". Must differ from `src_path` (a copy is made; `src_path` is left
        untouched).
    dataset_key : str
        Name of the array inside `src_path` (matches FourD's own dataset name;
        "4D" is FourD's default and virtually never needs changing).
    nav_axis_names, sig_axis_names : tuple of str
        Names for the two navigation (scan) axes and two signal (detector) axes,
        in that order -- matches FourD's own axis order (scan_y, scan_x, det_y,
        det_x).

    Returns
    -------
    str
        `dst_path` actually written.
    """
    import shutil
    import h5py

    if dst_path is None:
        dst_path = os.path.splitext(src_path)[0] + ".hspy"
    if os.path.abspath(dst_path) == os.path.abspath(src_path):
        raise ValueError("dst_path must differ from src_path -- this function copies, it doesn't convert in place.")
    shutil.copyfile(src_path, dst_path)

    with h5py.File(dst_path, "a") as f:
        if dataset_key not in f:
            raise KeyError(f"'{dataset_key}' dataset not found in {src_path}")
        shape = f[dataset_key].shape
        if len(shape) != 4:
            raise ValueError(f"expected a 4D dataset, got shape {shape}")

        f.attrs["file_format"] = "HyperSpy"
        f.attrs["file_format_version"] = "3.3"

        exps = f.require_group("Experiments")
        expg = exps.require_group(dataset_key)
        f.move(dataset_key, f"{expg.name}/data")

        axis_names = list(nav_axis_names) + list(sig_axis_names)
        navigate = [True, True, False, False]
        for i, (name, size, nav) in enumerate(zip(axis_names, shape, navigate)):
            ag = expg.require_group(f"axis-{i}")
            ag.attrs["_type"] = "UniformDataAxis"
            ag.attrs["name"] = name
            ag.attrs["navigate"] = bool(nav)
            ag.attrs["size"] = int(size)
            ag.attrs["scale"] = 1.0
            ag.attrs["offset"] = 0.0
            ag.attrs["units"] = "_None_"
            ag.attrs["is_binned"] = False

        expg.require_group("metadata")
        expg.require_group("original_metadata")
        expg.require_group("learning_results")
        att = expg.require_group("attributes")
        att.attrs["_lazy"] = False
        att.attrs["ragged"] = False

    return dst_path


class FourD():
    """
    makes a 4D dataset from events
    """

    def __init__(self,nx,ny,repetitions,filename,output_filename,bitdepth,compression_factor=7):

        if repetitions != 1:
            raise ValueError("repetitions should be 1 for FourD processor")
        
        if bitdepth not in [8,16,32]:
            raise ValueError("bitdepth should be 8, 16 or 32")
        if bitdepth == 8:
            self.super = eventem.FourD8(output_filename,repetitions,bitdepth=bitdepth,compression_factor=compression_factor)
        elif bitdepth == 16:
            self.super = eventem.FourD16(output_filename,repetitions,bitdepth=bitdepth,compression_factor=compression_factor)
        elif bitdepth == 32:
            self.super = eventem.FourD32(output_filename,repetitions,bitdepth=bitdepth,compression_factor=compression_factor)

        self.super.b_cumulative = True
        self.super.nx = nx
        self.super.ny = ny    
        self.super.set_file(filename)
        self.super.det_bin = 1
        self.super.scan_bin = 1
        self.super.chunksize = 2
        self.super.chunksize_x = 2

    def set_pattern_file(self, filename):
        """
        Switch to pixel-triggered (smart-scan / custom pattern) acquisition: reads
        `filename` as one linearized scan-position index per line and uses it to
        resolve each trigger's real (row, col), instead of computing position from
        dwell time. Call this AFTER __init__ (which already called set_file) --
        set_pattern_file re-decides the camera type from scratch, so it must be the
        last call to win. Requires ScanBin=1.
        """
        self.super.set_pattern_file(filename)

    @property
    def DetectorSize(self):
        """
        int : size of the detector
        """
        return self.super.detector_size
    
    @DetectorSize.setter
    def DetectorSize(self, value):
        self.super.detector_size = value

    @property
    def DetectorBin(self):
        """
        int : detector bin
        """
        return self.super.det_bin
    
    @DetectorBin.setter
    def DetectorBin(self, value):
        self.super.det_bin = value

    @property
    def ScanBin(self):
        """
        int : scan bin
        """
        return self.super.scan_bin
    
    @ScanBin.setter
    def ScanBin(self, value):
        self.super.scan_bin = value

    @property
    def ChunkSize(self):
        """
        int : chunk size
        """
        return self.super.chunksize
    
    @ChunkSize.setter
    def ChunkSize(self, value):
        self.super.chunksize = value

    @property
    def ChunkSizeX(self):
        """
        int : on-disk chunk width along the scan-x axis (independent of ChunkSize,
        which is the scan-y chunk height). Small values here (relative to nx) are
        what makes a small-ROI-across-several-lines read fast -- the old behavior
        (before this field existed) was always the full scan row width.
        """
        return self.super.chunksize_x

    @ChunkSizeX.setter
    def ChunkSizeX(self, value):
        self.super.chunksize_x = value

    @property
    def Format(self):
        """
        str : output container format, "hdf5" (default) or "zarr". Both write the
        same three logical arrays ("4D", "dose_image", "shape") with identical
        values and chunk shape -- only the on-disk container differs. Zarr output
        is a standard, spec-compliant Zarr v2 store (readable by any
        zarr-python/dask.array.from_zarr client, no custom reader needed), stored
        as "<output_filename>.zarr" alongside the (also always created, currently
        unused when Format="zarr") "<output_filename>.hdf5" file.
        """
        return self.super.format

    @Format.setter
    def Format(self, value):
        self.super.format = value

    @property
    def SaveMetadata(self):
        """
        bool : when True (default), the output file/store also gets the
        "dose_image" and "shape" auxiliary arrays alongside "4D". Set False to
        write only "4D" -- Dose_image is still computed and available from this
        object's .Dose_image property either way, just not persisted to disk.
        Generic 4D-STEM readers expect exactly one array per file/store; in
        particular hyperspy's hs.load(fn, lazy=True) assumes this and gets
        confused by "dose_image"/"shape" as extra siblings -- set SaveMetadata to
        False before calling .run() if you intend to load the output that way.
        """
        return self.super.save_metadata

    @SaveMetadata.setter
    def SaveMetadata(self, value):
        self.super.save_metadata = value

    @property
    def Decluster(self):
        """
        bool : charge-weighted, declustered 4D conversion (CHEETAH/.tpx3 only,
        bitdepth=32 only, requires ScanBin=1). Off by default -- when False,
        behavior is unchanged from before this feature existed.
        """
        return self.super.decluster

    @Decluster.setter
    def Decluster(self, value):
        self.super.decluster = value

    @property
    def Dtime(self):
        """
        int : declustering time window (same meaning/default as Roi.Dtime)
        """
        return self.super.dtime

    @Dtime.setter
    def Dtime(self, value):
        self.super.dtime = value

    @property
    def Dspace(self):
        """
        int : declustering space window (same meaning/default as Roi.Dspace)
        """
        return self.super.dspace

    @Dspace.setter
    def Dspace(self, value):
        self.super.dspace = value

    @property
    def ClusterRange(self):
        """
        int : declustering search range (same meaning/default as Roi.ClusterRange)
        """
        return self.super.cluster_range

    @ClusterRange.setter
    def ClusterRange(self, value):
        self.super.cluster_range = value

    @property
    def TotPerElectron(self):
        """
        float : calibration constant (typical single-electron cluster ToT-sum) used
        to resolve each cluster's electron count when Decluster=True. No safe
        default -- must be set from your own cluster ToT-sum histogram (e.g. Roi's,
        if acquired under the same beam/detector conditions).
        """
        return self.super.tot_per_electron

    @TotPerElectron.setter
    def TotPerElectron(self, value):
        self.super.tot_per_electron = value

    @property
    def ElectronCountLut(self):
        """
        2D nested list/array of int : alternative to TotPerElectron -- resolves each
        cluster's electron count from a (cluster size, ToT) map instead of a single
        ToT/TotPerElectron ratio. Row = cluster size, column = ToT, matching
        ClustersizeTotHistogram's own layout. Empty by default (TotPerElectron stays
        the default resolver); set directly, or set ElectronCountLutFile to load a
        saved template instead. Both options remain available side by side.
        """
        return np.array(self.super.electron_count_lut)

    @ElectronCountLut.setter
    def ElectronCountLut(self, value):
        self.super.electron_count_lut = value

    @property
    def ElectronCountLutFile(self):
        """
        str : path to a plain-text electron-count template file (see
        ElectronCountLut) -- loaded into ElectronCountLut at the start of Run() if
        non-empty. A reusable calibration artifact: valid for reuse as long as the
        beam energy / detector setup it was built from hasn't changed. Empty by
        default (feature off).
        """
        return self.super.electron_count_lut_file

    @ElectronCountLutFile.setter
    def ElectronCountLutFile(self, value):
        self.super.electron_count_lut_file = value

    @property
    def ClustersizeHistogram(self):
        """
        1D numpy array : histogram of resolved cluster sizes, from the most recent
        declustered Run().
        """
        return np.array(self.super.clustersize_histogram)

    @property
    def EnergyHistogram(self):
        """
        1D numpy array : histogram of resolved clusters' total summed ToT, from the
        most recent declustered Run().
        """
        return np.array(self.super.energy_histogram)

    @property
    def ClustersizeTotHistogram(self):
        """
        2D numpy array, shape (50, 4096) : joint histogram of resolved cluster size
        (rows) vs. total summed ToT (columns), from the most recent declustered
        Run().
        """
        return np.array(self.super.clustersize_tot_histogram)

    @property
    def DwellTime(self):
        """
        int : dwell time
        """
        return self.super.dt
    
    @DwellTime.setter
    def DwellTime(self, value):
        self.super.dt = value

    @property
    def CountImage(self):
        """
        2D numpy array : image of the total number of counts in each probe position
        """
        return np.array(self.super.Dose_image).reshape(self.super.nx,self.super.ny)


    def Run(self):
        """
        Run the 4D dataset creation
        """

        self.super.allocate_chunk() # allocate the memory for the 4D array
        self.super.init_4D_file() # initialize the hdf5 file writing
        self.super.run()

