# *****************************************************************************
# Copyright (C) 2021 INAF
# This software was provided as IKC to the Cherenkov Telescope Array Observatory
# This software is distributed under the terms of the BSD-3-Clause license
#
# Authors:
#
#    Ambra Di Piano <ambra.dipiano@inaf.it>
#    Nicolò Parmiggiani <nicolo.parmiggiani@inaf.it>
#    Andrea Bulgarelli <andrea.bulgarelli@inaf.it>
#    Valentina Fioretti <valentina.fioretti@inaf.it>
#    Leonardo Baroncelli <leonardo.baroncelli@inaf.it>
#    Antonio Addis <antonio.addis@inaf.it>
#    Giovanni De Cesare <giovanni.decesare@inaf.it>
#    Gabriele Panebianco <gabriele.panebianco3@unibo.it>
# *****************************************************************************

import os
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.coordinates import SkyCoord, Angle
from astropy.table import Table
from astropy.time import Time
from astropy.visualization.wcsaxes import SphericalCircle
from matplotlib import use
from matplotlib.colors import LogNorm, PowerNorm
from matplotlib.pyplot import subplots
from regions import CircleSkyRegion, Regions
from warnings import filterwarnings
from gammapy.data import EventList, GTI, Observation
from gammapy.datasets import MapDataset
from gammapy.estimators import ExcessMapEstimator
from gammapy.estimators.utils import find_peaks
from gammapy.irf import load_irf_dict_from_file
from gammapy.makers import  MapDatasetMaker, SafeMaskMaker, ReflectedRegionsFinder, ReflectedRegionsBackgroundMaker
from gammapy.maps import Map, WcsGeom, MapAxis
from astroai.tools.utils import convert_tt_to_mjd, get_irf_file

# Ignore some warnings
from warnings import filterwarnings
filterwarnings("ignore")

class GAnalysis():
    def __init__(self) -> None:
        pass

    def set_logger(self, logger):
        self.log = logger
        return self

    def set_conf(self, conf):
        self.conf = conf
        return self
    
    def set_eventfilename(self, eventfilename):
        self.eventfilename = eventfilename
        return self
    
    def set_reducedirfs(self, directory, seed):
        self.reducedirfs = os.path.join(directory, f"reduced_{seed}.fits")
        assert os.path.isfile(self.reducedirfs)
        return self
    
    def define_energy_axis_reco(self):
        # Read Energy Bounds and spectral resolution from the Jobconf
        emin_reco, emax_reco = (self.conf['selection']['emin'], self.conf['selection']['emax']) * u.TeV 
        axis_energy_reco = MapAxis.from_energy_bounds(emin_reco, emax_reco, self.conf['selection']['energy_bins_per_decade'], per_decade=True, name='energy' )        
        return axis_energy_reco
        
    def define_energy_axis_true(self):
        # Read Energy Bounds and spectral resolution from the Jobconf
        emin_reco, emax_reco = (self.conf['selection']['emin'], self.conf['selection']['emax']) * u.TeV
        # Define the True Energy Axis as Larger and better sampled than Reco Axis.
        #TODO: How much?
        axis_energy_true = MapAxis.from_energy_bounds(0.95*emin_reco,  1.05*emax_reco, int(1.5*self.conf['selection']['energy_bins_per_decade']), per_decade=True, name='energy_true' )
        return axis_energy_true
        
    def define_geometry(self):
        # 1 - Define the Spatial Sky Geometry for the Maps as the ROI
        sky_direction = SkyCoord(self.conf['simulation']['point_ra'], self.conf['simulation']['point_dec'], unit=u.Unit(self.conf['simulation']['skyframeunitref']), frame=self.conf['simulation']['skyframeref'])
        sky_mapwidth = 2*float(self.conf['selection']['roi_ringrad']) * u.Unit(self.conf['simulation']['skyframeunitref'])
        sky_pixsize = self.conf['execute']['pixel_size'] # Must be in deg.
        # 2 - Define the Spectral Grid: Energy Axes.
        axis_energy_reco = self.define_energy_axis_reco()
        # 3 - Define the Geometry object
        geom = WcsGeom.create(skydir=sky_direction, binsz=sky_pixsize, width=(sky_mapwidth, sky_mapwidth), frame=self.conf['simulation']['skyframeref'], proj="CAR", axes=[axis_energy_reco])
        return geom
        
    def define_observation(self):
        # 1 - Load IRFs
        irfs = load_irf_dict_from_file(self.conf['simulation']['irf_file'])
        # 2 - Pointing. From OBS.xml
        pointing = SkyCoord(self.conf['simulation']['point_ra'], self.conf['simulation']['point_dec'], frame=self.conf['simulation']['skyframeref'], unit=self.conf['simulation']['skyframeunitref'])
        # 3 - Set a standard livetime through tstart and tstop.
        # It will be updated during the science analysis with the actual data livetime.
        # Typical batch size = 100 s.
        tstart, tstop =  (0.0, 100.0) * u.s
        # Define the gammapy Observation
        observation = Observation.create(pointing = pointing, obs_id = f"{self.conf['simulation']['id']}", irfs=irfs, tstart=tstart, tstop=tstop)        
        return observation
    
    def make_empty_dataset(self):
        # 1 - Define the Geometry Object i.e., analysis 3D grid: spatial + spectral (reconstructed energy)
        geom = self.define_geometry()
        # 2 - Define the True Energy Axis
        axis_energy_true = self.define_energy_axis_true()
        # 3 - Define the empty dataset.
        dataset = MapDataset.create(geom = geom, energy_axis_true=axis_energy_true, name = f"MapDataset_seed{self.conf['simulation']['id']}")
        # Set Mask_safe to True, otherwise it won't find the counts
        dataset.mask_safe.data=True        
        return dataset
    
    def make_reduced_irfs(self):
        # 1 - Define the Observation Object
        observation = self.define_observation()
        # 2 - Define the Geometry Object i.e., analysis 3D grid: spatial + spectral (reconstructed energy)
        geom = self.define_geometry()
        # 3 - Define the True Energy Axis
        axis_energy_true = self.define_energy_axis_true()
        # 4 - Perform DL3->DL4 Reduction of IRFs in the chosen Geometry.
        dataset_empty = MapDataset.create(geom = geom, energy_axis_true=axis_energy_true, name=f"MapDataset_obsid_{self.conf['simulation']['id']}")
        maker = MapDatasetMaker(selection = ['exposure', 'background', 'psf', 'edisp'])
        # Offset-max  = mask IRF data outside a maximum offset from Geometry central Sky Direction
        # Aeff-default= mask IRF data outside the energy ranged specified in the DL3 data files, if available.
        sky_mapwidth = 2*float(self.conf['selection']['roi_ringrad']) * u.Unit(self.conf['simulation']['skyframeunitref'])
        safe_mask_maker = SafeMaskMaker(methods=["aeff-default","offset-max"], offset_max=sky_mapwidth)
        dataset = maker.run(dataset_empty, observation)
        dataset = safe_mask_maker.run(dataset, observation)        
        # 5 - Add exclusion Mask given a DS9 region file with known sources
        if self.conf['execute']['exclusion'] != None:
            exclude_regions = os.path.join(self.conf['execute']['outdir'], self.conf['execute']['exclusion'])
            exclusion_mask = dataset.geoms['geom'].region_mask(Regions.read(exclude_regions,format="ds9"), inside=False)
            dataset.mask_safe &= exclusion_mask
        return dataset

    def write_dataset(self, dataset):
        # Set Output Folder
        reduced_irfs_directory = self.conf['execute']['reducedirfdir']
        os.makedirs(reduced_irfs_directory, exist_ok=True)
        # Write the Dataset with the Reduced IRFs that is going to be used by every pipeline call
        dataset.write(self.reducedirfs, overwrite=True)
        # Plot Reduced IRFs Maps
        if self.conf['execute']['plotirfs']:
            self.plot_Wcs2DMap(dataset.background, "IRF Bkgd Counts 100s", run_ID=False, stretch="sqrt", output_directory=reduced_irfs_directory)
            self.plot_Wcs2DMap(dataset.exposure  , "IRF Exposure 100s"   , run_ID=False, stretch="log" , output_directory=reduced_irfs_directory)
            self.plot_Wcs2DMap(dataset.mask_safe_image, "Exclusion Mask", run_ID=False, cmap="gray", extracolor="blue", output_directory=reduced_irfs_directory)


    def plot_Wcs2DMap(self, sky_cube, tag, run_ID = True, gti = None, figsize = (15, 15), cmap = 'cividis', stretch = 'linear', output_directory = None, hotspots = None, extracolor = "white", mask = None, FORMAT="png", backend="Agg"):
        # Sum map values over all non-spatial axes.
        if sky_cube.data.ndim > 2: 
            sky_map = sky_cube.sum_over_axes(keepdims=False)
        elif sky_cube.data.ndim==2:
            sky_map = sky_cube
        else:
            raise NotImplementedError("Define a strategy to deal with 1D maps.")

        data = sky_map.data
        unit = sky_map.unit
        projection = sky_map.geom.wcs

        # Mask
        if mask is not None:
            data *= mask.data
        
        # Define Color Normalization
        if stretch == "log":
            norm = LogNorm()
        elif stretch =='linear':
            norm = PowerNorm(gamma=1)
        elif stretch == 'sqrt':
            norm = PowerNorm(gamma=0.5)
        elif isinstance(stretch, float):
            norm = PowerNorm(gamma=stretch)
        else:
            raise NotImplementedError(f"stretch={stretch} not allowed. Only float or string in[\'linear\', \'sqrt\', \'log\'] are allowed.")          
        
        # Plot
        use(backend)
        fig, ax = subplots(1, subplot_kw={"projection":projection}, figsize = figsize)
        im = ax.imshow(data, cmap=cmap, norm=norm)
        cb = fig.colorbar(im, ax = ax, location = 'right', shrink = 0.8)
        
        # Pot FoV Centre if it is inside axes limits
        pointing = SkyCoord(self.conf['simulation']['point_ra'], self.conf['simulation']['point_dec'], unit=self.conf['simulation']['skyframeunitref'], frame=self.conf['simulation']['skyframeref'])
        if sky_map.geom.contains(pointing):
            ax.plot_coord(pointing, marker='X', color=extracolor)
        
        # Plot one or more regions as circles
        if hotspots is not None:
            for hotspot in hotspots:
                hotspot_color = hotspot.meta['color'] if 'color' in hotspot.meta else extracolor
                s = SphericalCircle(hotspot.center, hotspot.radius, edgecolor=hotspot_color, facecolor='none')
                ax.add_patch(s)
        
        # Customisation: Titles and Labels
        title_plot   = f"Map {tag}. OB {self.conf['simulation']['id']}."
        title_output = f"seed{self.conf['simulation']['id']}_"
        if run_ID is True:
            title_plot  += f" Job {self.conf['simulation']['id']}."
        if gti is not None:
            time_ref   =  gti.time_ref.to_value(format='iso')
            time_start = (gti.time_start- gti.time_ref).to('s')[0]
            time_stop  = (gti.time_stop - gti.time_ref).to('s')[0]
            title_plot+= f" {gti.time_ref.scale.upper()} {time_ref} + [{time_start}, {time_stop}]."
        if output_directory is None:
            output_directory = self.conf['execute']['outdir']
        title_output+= f"{tag.replace(' ', '_').lower()}.{FORMAT.lower()}"
        
        Titles = {"Plot"    : title_plot,
                  "X_label" : "Right Ascension",
                  "Y_label" : "Declination",
                  "Colorbar": f"{tag} [{unit}]. Stretch: {stretch}",
                  "Output"  : os.path.join(output_directory, title_output)
                  } 
        if self.conf['simulation']['skyframeref'] == 'galactic':
            Titles['X_label'] = "Galactic Longitude"
            Titles['Y_label'] = "Galactic Latitude"    
        
        
        ax.grid(color=extracolor, ls='dotted')
        ax.set_facecolor(im.cmap(0))
        cb.set_label(Titles['Colorbar'], fontsize='large', rotation=90)
        ax.set_title(Titles['Plot'    ], fontsize='large')
        ax.set_xlabel(Titles['X_label'], fontsize='large')
        ax.set_ylabel(Titles['Y_label'], fontsize='large')
        
        # RA, DEC in degree.decimal, not hh:mm:ss
        if self.conf['simulation']['skyframeref'] != 'galactic':
            ax.coords[0].set_major_formatter('d.d')
            ax.coords[1].set_major_formatter('d.d') 
            ax.invert_xaxis()       

        # Save Image. Create job directory if it does not exist
        os.makedirs(self.conf['execute']['outdir'], exist_ok=True)
        fig.savefig(Titles['Output'])
        plt.close()
    
    def read_dataset(self):
        # Read the dataset
        dataset = MapDataset.read(self.reducedirfs, name=f"seed{self.conf['simulation']['id']}_")
        return dataset
    
    def run_gammapy_analysis_pipeline(self, dataset, name, target_dict):
        # SECTION 1 - Setup.
        # Get the ID of the Observation Block and of the current data batch (Job ID)
        Id_OB = self.conf['simulation']['id']
        # Read and set all the data, IRFs, GTIs and make appropriate corrections.
        dataset, event_list, gti = self.read_events(dataset)
        
        # SECTION 2 - COUNTS MAP
        if self.conf['execute']['savefits']:
            # Write 3D Counts Cube as FITS
            output_name = os.path.join(self.conf['execute']['outdir'], f"seed{Id_OB}_counts_cube.fits")
            dataset.counts.write(output_name, overwrite=True)
        # Save Plot of 2D Counts Map
        # When AP is inactive, jobconf.makemap will control this functionality.
        # When AP is active, jobconf.makemap will create the zoomed map, only plotfullfov can print this. 
        if self.conf['execute']['plotfullfov'] or (self.conf['execute']['makemap'] and not self.conf['execute']['computeph']):
            self.plot_Wcs2DMap(dataset.counts, "Counts", stretch='sqrt', gti=gti)
        # Save Plots for Predicted Background Counts and Exposure
        if self.conf['execute']['plotirfs']:
            self.plot_Wcs2DMap(dataset.background, "IRF Bkgd Counts", stretch="sqrt"  , gti=dataset.gti)
            self.plot_Wcs2DMap(dataset.exposure  , "IRF Exposure"   , stretch="linear", gti=dataset.gti)
        
        # SECTION 3 - BLIND SEARCH
        if self.conf['execute']['blindsearch']:            
            # Perform Blindsearch
            try:
                target_ra, target_dec = self.run_blind_search(dataset, blind_search_method = 'first')
            except:
                target_ra, target_dec = np.nan, np.nan
            # Update target dict
            target_dict = {'ra': target_ra, 'dec': target_dec, 'rad': self.conf['photometry']['onoff_radius']}
            if name=='None':
                name='Hotspot'        

        # SECTION 4 - APERTURE PHOTOMETRY ON THE TARGET (1D Analysis)
        if self.conf['execute']['computeph'] and (target_ra, target_dec) != (np.nan, np.nan):
            spectrum_dataset_OnOff, stats = self.run_aperture_photometry(dataset, target_dict, name, event_list, gti, method=self.conf['photometry']['onoff_method'])
        
            # Propagate statistical errors on Excess and Li&Ma Significance
            excess_err= np.sqrt(np.power(np.sqrt(stats['counts']),2) + np.power(np.sqrt(stats['counts_off']),2))
            sigma_err = 0
            stats['excess_error'] = excess_err       
            stats['sigma_error'] = sigma_err
        else:
            stats={'counts'       :0.0,
                   'counts_off'   :0.0,
                   'excess'       :0.0,
                   'alpha'        :0.0,
                   'sigma'        :0.0,
                   'livetime'     :0.0,
                   'excess_error' :0.0,
                   'sigma_error'  :0.0,
                   'aeff_mean'    :0.0
                   }
        return stats, target_dict

    def read_events(self, dataset):
        from gammapy.data import EventList, GTI
        from astropy.time import Time
        import astropy.units as u
        from gammapy.maps import Map

        event_list = EventList.read(self.eventfilename, hdu="EVENTS")
        event_table = event_list.table

        # TIME 现在是 astropy.time.Time，需把 tmin/tmax 转成绝对时间再比较
        time_min = event_list.time_ref + self.conf["selection"]["tmin"] * u.s
        time_max = event_list.time_ref + self.conf["selection"]["tmax"] * u.s

        # ENERGY 最好补单位，避免 quantity 比较问题
        emin = self.conf["selection"]["emin"] * event_table["ENERGY"].unit
        emax = self.conf["selection"]["emax"] * event_table["ENERGY"].unit

        mask = (
            (event_table["TIME"] > time_min) &
            (event_table["TIME"] < time_max) &
            (event_table["ENERGY"] > emin) &
            (event_table["ENERGY"] < emax)
        )

        event_table = event_table[mask]
        event_table.meta["RA_PNT"] = self.conf["simulation"]["point_ra"]
        event_table.meta["DEC_PNT"] = self.conf["simulation"]["point_dec"]
        event_list.table = event_table

        if self.conf["simulation"]["timesys"] == "tt":
            time_ref = convert_tt_to_mjd(self.conf["simulation"]["timeref"])
        else:
            time_ref = self.conf["simulation"]["timeref"]

        tmin = self.conf["selection"]["tmin"] * u.Unit(self.conf["simulation"]["timeunit"])
        tmax = self.conf["selection"]["tmax"] * u.Unit(self.conf["simulation"]["timeunit"])
        gti = GTI.create(tmin, tmax, reference_time=Time(time_ref, format="mjd"))

        if self.conf["execute"]["reducedirfdir"] is not None:
            duration_rescale_factor = (gti.time_sum / dataset.gti.time_sum).to("").value
            dataset.background = dataset.background * duration_rescale_factor
            dataset.exposure = dataset.exposure * duration_rescale_factor
            dataset.exposure.meta["livetime"] *= duration_rescale_factor
        else:
            dataset.exposure.meta["livetime"] = gti.time_sum

        dataset.gti = gti

        counts_cube = Map.from_geom(dataset.geoms["geom"])
        counts_cube.fill_events(event_list)
        dataset.counts = counts_cube
        return dataset, event_list, gti
    
    def run_blind_search(self, dataset, blind_search_method='first'):
        OnOffRegionRadius = self.conf['photometry']['onoff_radius'] * u.Unit("deg")

        # Blind Search performed with ExcessMapEstimator
        excess_map_estimator = ExcessMapEstimator(correlation_radius = self.conf['blindsearch']['corr_rad'] * u.deg)
        result = excess_map_estimator.run(dataset)
        sqrt_TS_map = result['sqrt_ts']

        # Find the Peaks
        hotspots_table = find_peaks(sqrt_TS_map, threshold=self.conf['blindsearch']['sigmathresh'], min_distance=OnOffRegionRadius)
        try:
            # Select hotspots within a maximum offset from pointing direction.
            Pointing = SkyCoord(self.conf['simulation']['point_ra'], self.conf['simulation']['point_dec'], frame=self.conf['simulation']['skyframeref'], unit=self.conf['simulation']['skyframeunitref'])
            Max_offset = self.conf['blindsearch']['maxoffset'] * u.Unit(self.conf['simulation']['skyframeunitref'])
            positions = SkyCoord(hotspots_table['ra'], hotspots_table['dec'], unit=u.deg, frame=self.conf['simulation']['skyframeref'])
            offset_mask = [position.separation(Pointing) > Max_offset for position in positions]
            hotspots_table.remove_rows(offset_mask)
            # Select a maximum number of hotspots
            hotspots_table = hotspots_table[:self.conf['blindsearch']['maxsrc']]
            # Save Hotspots as a list of CircleSkyRegion
            hotspots = SkyCoord(hotspots_table["ra"], hotspots_table["dec"])
            hotspots = [CircleSkyRegion(hotspot, OnOffRegionRadius) for hotspot in hotspots]
            
        except KeyError: 
            raise KeyError('No candidate found.')
        
        # Save SqrtTS Map and Zoom on the Hotspots.        
        if self.conf['execute']['plotts']:
            # Plot the sqrt_ts map with all the regions            
            self.plot_Wcs2DMap(sqrt_TS_map, f"SqrtTS", gti=dataset.gti, hotspots=hotspots)

        # Plot the zoom over each of the hotspots (Counts Map)
        if self.conf['blindsearch']['plotzoom']:
            for i, hotspot in enumerate(hotspots):
                cutout = dataset.cutout(hotspot.center, width=4.0*OnOffRegionRadius)
                self.plot_Wcs2DMap(cutout.counts, f"Hotspot{i+1} Counts", stretch='sqrt', gti=dataset.gti, hotspots=[hotspot])
        # Plot the zoom over each of the hotspots (Sqrt TS Map)
        if self.conf['blindsearch']['plotzoom'] and self.conf['execute']['plotts']:
            for i, hotspot in enumerate(hotspots):
                cutout = sqrt_TS_map.cutout(hotspot.center, width=4.0*OnOffRegionRadius)
                self.plot_Wcs2DMap(cutout, f"Hotspot{i+1} SqrtTS", gti=dataset.gti, hotspots=[hotspot])

        # Select sources according to the requested method
        if blind_search_method == 'first':
            target_ra, target_dec = (hotspots[0].center.ra.deg, hotspots[0].center.dec.deg)
        else:
            raise NotImplementedError('Currently only the "first" method is available.')
        
        # Write selected source as a DS9 region file. Create job directory if it does not exist
        os.makedirs(self.conf['execute']['outdir'], exist_ok=True)
        regions = CircleSkyRegion(SkyCoord(target_ra, target_dec, unit=u.deg, frame = self.conf['simulation']['skyframeref']), OnOffRegionRadius)
        regions.write(os.path.join(self.conf['execute']['outdir'], f"{self.conf['simulation']['id']}_candidates.ds9"), overwrite=True)        
        return target_ra, target_dec

    def run_aperture_photometry(self, dataset, target_dict, target_name, event_list, gti, method="reflection"):
        obs = Observation(events=event_list, gti=gti)

        # 1 - Define the ON Region, Compute and store ON Counts info.
        target_position = SkyCoord(target_dict['ra'], target_dict['dec'], frame=self.conf['simulation']['skyframeref'], unit=self.conf['simulation']['skyframeunitref'])
        on_region_radius = Angle(target_dict['rad'], unit=u.deg)
        on_region = CircleSkyRegion(target_position, on_region_radius)
        
        # 2 - Define the Pointing and compute the Target-Pointing distance
        pointing = SkyCoord(self.conf['simulation']['point_ra'], self.conf['simulation']['point_dec'], frame=self.conf['simulation']['skyframeref'], unit=self.conf['simulation']['skyframeunitref'])
        distance_pointing_target = target_position.separation(pointing)

        # 3 - Extract Counts, Exposure, Edisp, IRF background in the ON region.
        # Correct for PSF only if IRFs are available.
        containment_correction = self.conf['execute']['reducedirfdir'] is not None
        spectrum_dataset = dataset.to_spectrum_dataset(on_region, containment_correction=containment_correction, name=target_name)
        
        # 4 - Define Algorithm to compute OFF Regions  
        if method == "reflection":
            # We want to exclude regions too close to ON regions due to source contamination.
            # Let's compute the plain angle spanned by the On region wrt Pointing direction
            radii_ratio = (on_region_radius / distance_pointing_target).to('')
            region_aperture_angle = np.arccos(np.sqrt(1-np.power(radii_ratio,2)))
            off_regions_finder = ReflectedRegionsFinder(angle_increment=2.0*region_aperture_angle, min_distance_input=2.0*region_aperture_angle, binsz=self.conf['execute']['pixel_size'] * u.deg)
        else:
            raise ValueError("Methods to find OFF regions must be \'reflection\'.")
        refl_bkg_maker = ReflectedRegionsBackgroundMaker(region_finder=off_regions_finder) 

        # 5 - Write Off regions
        off_regions, off_wcs = refl_bkg_maker.region_finder.run(center=obs.pointing.fixed_icrs, region=on_region)
        if off_regions == []:
            print('Cannot compute off regions.')
            spectrum_dataset_OnOff = None
            stats = {'counts'    : np.nan,
                    'counts_off' : np.nan,
                    'excess'     : np.nan,
                    'alpha'      : np.inf,
                    'sigma'      : np.nan,
                    'livetime'   : np.nan,
                    'aeff_mean'  : np.nan,  
                    'offset'     : np.nan,
                    }
            return spectrum_dataset_OnOff, stats
        else:   
            Regions(off_regions).write(os.path.join(self.conf['execute']['outdir'], 'hotspots.reg'), overwrite=True)           

        # 6 - Compute the OFF Regions: Counts are taken from the Event List
        spectrum_dataset_OnOff = refl_bkg_maker.run(spectrum_dataset, obs)
        
        # Extract Info from SpectrumDatasetOnOff
        infodict = spectrum_dataset_OnOff.info_dict()
        effective_area_mean = 0.0 # TODO: COMPUTE MEAN EFFECTIVE AREA
        stats = {'counts'    : infodict['counts'],
                'counts_off' : infodict['counts_off'],
                'excess'     : infodict['excess'],
                'alpha'      : infodict['alpha'],
                'sigma'      : infodict['sqrt_ts'],
                'livetime'   : infodict['livetime'].to('s').value,
                'aeff_mean'  : effective_area_mean,
                'offset'     : distance_pointing_target.deg,
                }

        # 7 - Plot Counts Map and Mask
        if self.conf['execute']['makemap']:
            # Plot On and Off Regions in the Counts Map if requested 
            if self.conf['execute']['mapreg']:
                # Pointing-Target circle
                hotspots = [CircleSkyRegion(pointing, distance_pointing_target, meta={'color':'white'})]
                # Off Regions
                for off_region in off_regions:
                    off_region.meta = {'color':'white'}
                hotspots+=off_regions
                # On Region
                hotspots.append(CircleSkyRegion(on_region.center, on_region.radius, meta={'color':'green'}))
            else:
                hotspots=None
            
            # Zoom of the Aperture Photometry Region
            cutout = dataset.cutout(pointing, width=2.0*self.conf['execute']['maproi'] * u.Unit(self.conf['simulation']['skyframeunitref']))
            # Plot
            self.plot_Wcs2DMap(cutout.counts, f"sky1", gti=cutout.gti, stretch='sqrt', hotspots=hotspots, mask=cutout.mask_safe_image)
        return spectrum_dataset_OnOff, stats

    def execute_dl3_dl4_reduction(self):
        # Set (Externally stored) IRFs file name. Need OBS.xml, JOB.xml
        self.conf['simulation']['irf_file'] = get_irf_file(self.conf['simulation']['caldb'], self.conf['simulation']['irf'], self.conf['simulation']['caldb_path'])
        # Start Gammapy Analysis.
        # Run the DL3 to DL4 IRFs reduction.
        dataset = self.make_reduced_irfs()
        self.write_dataset(dataset)
        self.set_reducedirfs(self.conf['execute']['reducedirfdir'], seed=self.conf['simulation']['id'])