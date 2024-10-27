import glob
import os
import warnings

import numpy as np
import spectral as spec
from speclearn.tools.constants import LAT_DEG, LONG_DEG, M3_DATA_DIR
from speclearn.tools.map_projections import stereographic_projection, to_radians


class M3_image:
    """
    Class containing a single M3 image.
    """

    def __init__(self, file='', level=None):
        self.level = level

        # Image members
        self.img_hdr = file
        self.img = None
        self.loc_hdr = None
        self.loc_img = None
        self.sup_hdr = None
        self.sup_img = None
        self.obs_hdr = None
        self.obs_img = None

        # observation
        self.sun_azimuth = None
        self.sun_zenith = None
        self.sensor_azimuth = None
        self.sensor_zenith = None
        self.obs_phase_angle = None
        self.to_sun_path_length = None
        self.to_sensor_path_length = None
        self.surface_slope_from_DEM = None
        self.surface_aspect_from_DEM = None
        self.local_cosine_i = None

        # date
        self.date = None
        self.period = None
        self.data_file = None
        self.date_interval = None

        # Data members
        self.data = None
        # Location
        self.latitude = None
        self.longitude = None
        self.radius = None
        self.r = None
        self.theta = None
        # Supplementary
        self.reflectance = None
        self.temperature = None
        self.radiance = None

    def __repr__(self):
        output_str = 'Data source ' + self.img_hdr + '\n'
        if self.img:
            output_str += \
                f'Image \n' \
                f'# Rows:           {self.img.shape[0]} \n' \
                f'# Samples:        {self.img.shape[1]} \n' \
                f'# Bands:          {self.img.shape[2]} \n'
        if self.loc_img:
            output_str += \
                f'Location (longitude, latitude, radius) \n' \
                f'# Rows:           {self.loc_img.shape[0]} \n' \
                f'# Samples:        {self.loc_img.shape[1]} \n' \
                f'# Location :      {self.loc_img.shape[2]} \n'
        if self.sup_img:
            output_str += \
                f'Supplementary (reflectance, temperature(degK), radiance) \n' \
                f'# Rows:           {self.sup_img.shape[0]} \n' \
                f'# Samples:        {self.sup_img.shape[1]} \n' \
                f'# Supplementary : {self.sup_img.shape[2]} '
        return output_str

    def __getitem__(self, select):
        img = M3_image(file=self.img_hdr)
        img.level = self.level

        # Image members
        img.img_hdr = self.img_hdr
        img.img = self.img
        img.loc_hdr = self.loc_hdr
        img.loc_img = self.loc_img
        img.sup_hdr = self.sup_hdr
        img.sup_img = self.sup_img
        img.obs_hdr = self.obs_hdr
        img.obs_img = self.obs_img

        # date
        img.date = self.date
        img.period = self.period
        img.data_file = self.data_file
        img.date_interval = self.date_interval

        # Data members
        if self.data is not None:
            img.data = self.data  # samples, 304, bands (85)
        # Location
        img.latitude = self.latitude  # samples, 304
        img.longitude = self.longitude  # samples, 304
        img.radius = self.radius  # samples, 304
        img.r = self.r  # samples, 304
        img.theta = self.theta  # samples, 304
        # Supplementary
        img.reflectance = self.reflectance
        img.temperature = self.temperature
        img.radiance = self.radiance
        # Observation
        img.sun_azimuth = self.sun_azimuth
        img.sun_zenith = self.sun_zenith
        img.sensor_azimuth = self.sensor_azimuth
        img.sensor_zenith = self.sensor_zenith
        img.obs_phase_angle = self.obs_phase_angle
        img.to_sun_path_length = self.to_sun_path_length
        img.to_sensor_path_length = self.to_sensor_path_length
        img.surface_slope_from_DEM = self.surface_slope_from_DEM
        img.surface_aspect_from_DEM = self.surface_aspect_from_DEM
        img.local_cosine_i = self.local_cosine_i

        # Data members
        if self.data is not None:
            img.data[np.squeeze(~select), :] = float(
                'NaN')  # samples, 304, bands (85)
        # Location
        img.latitude[~select] = float('NaN')  # samples, 304
        img.longitude[~select] = float('NaN')  # samples, 304
        img.radius[~select] = float('NaN')  # samples, 304
        if img.r is not None:
            img.r[~select] = float('NaN')  # samples, 304
            img.theta[~select] = float('NaN')  # samples, 304
        # Supplementary
        img.reflectance[~select] = float('NaN')
        img.temperature[~select] = float('NaN')
        img.radiance[~select] = float('NaN')
        # Observation
        img.sun_azimuth[~select] = float('NaN')
        img.sun_zenith[~select] = float('NaN')
        img.sensor_azimuth[~select] = float('NaN')
        img.sensor_zenith[~select] = float('NaN')
        img.obs_phase_angle[~select] = float('NaN')
        img.to_sun_path_length[~select] = float('NaN')
        img.to_sensor_path_length[~select] = float('NaN')
        img.surface_slope_from_DEM[~select] = float('NaN')
        img.surface_aspect_from_DEM[~select] = float('NaN')
        img.local_cosine_i[~select] = float('NaN')

        return img

    def get_info(self):
        self.fill_dates()
        return self

    def fill(self):
        self.get_info()
        self.img = spec.io.envi.open(self.img_hdr, image=self.get_image_file())
        # self.fill_sup()
        if self.level:
            self.fill_location(level=self.level)
        return self

    def load_data(self):
        self.data = np.array(self.img.load())
        self.data = format_flagged_pixels(self.data)
        return

    def get_image_file(self):
        file_split = self.img_hdr.split('/')[:]
        data_dir = '/'.join(file_split[:-9] + ['*'] + file_split[-8:-6])
        file_path = data_dir + '/' + \
            '/'.join(['CH1M3_0004', 'DATA',
                     self.date_interval, self.date[0:6], 'L2'])
        files = glob.glob(os.path.join(file_path, self.data_file + '*RFL.IMG'))
        img_file = files[0]
        return img_file

    def get_loc_img_file(self):
        file_split = self.img_hdr.split('/')[:]
        data_dir = '/'.join(file_split[:-9] + ['*'] + file_split[-8:-6])
        file_path = data_dir + '/' + \
            '/'.join(['CH1M3_0003', 'DATA', self.date_interval,
                     self.date[0:6], 'L1B'])
        files = glob.glob(os.path.join(file_path, self.data_file + '*LOC.IMG'))
        img_file = files[0]
        return img_file

    def get_loc_hdr_file(self):
        file_split = self.img_hdr.split('/')[:]
        data_dir = '/'.join(file_split[:-9] + ['*'] + file_split[-8:-6])
        file_path = data_dir + '/' + \
            '/'.join(['CH1M3_0003', 'DATA',
                     self.date_interval, self.date[0:6], 'L1B'])
        files = glob.glob(os.path.join(file_path, self.data_file + '*LOC.HDR'))
        img_file = files[0]
        return img_file

    def fill_sup(self):
        self.sup_hdr = self.find_sup_header()
        if self.sup_hdr is None:
            return self
        self.sup_img = spec.open_image(self.sup_hdr)
        try:
            self.reflectance = self.sup_img[:, :, 0]
            self.temperature = self.sup_img[:, :, 1]
            self.radiance = self.sup_img[:, :, 2]
            self.reflectance = format_flagged_pixels(self.reflectance)

        except:
            pass
        finally:
            return self
        return self

    def find_sup_header(self):
        file_split = self.img_hdr.split('/')[:]
        file_path = '/'.join(file_split[:-1])
        headers = glob.glob(os.path.join(
            file_path, self.data_file + '*SUP.HDR'))
        if len(headers) == 0:
            warnings.warn('No LOC header found.')
            sup_hdr = None
        else:
            sup_hdr = headers[0]
        if len(headers) > 1:
            warnings.warn(
                'Found more than one SUP header, something is wrong.')
        return sup_hdr

    def fill_observation(self):
        if self.date is None:
            self.get_info()
        self.obs_hdr = self.find_obs_header()
        if os.path.exists(self.obs_hdr):
            self.obs_img = spec.open_image(self.obs_hdr)
            # decimal degrees, clockwise from local north
            self.sun_azimuth = self.obs_img[:, :, 0]
            # incidence angle in decimal degrees, zero at zenith
            self.sun_zenith = self.obs_img[:, :, 1]
            # decimal degrees, clockwise from local north
            self.sensor_azimuth = self.obs_img[:, :, 2]
            # emission angle in decimal degrees, zero at zenith
            self.sensor_zenith = self.obs_img[:, :, 3]
            # decimal degrees, in plane of to-sun and to-sensor rays
            self.obs_phase_angle = self.obs_img[:, :, 4]
            # decimal au with scene mean subtracted and noted in PDS label
            self.to_sun_path_length = self.obs_img[:, :, 5]
            # decimal meters
            self.to_sensor_path_length = self.obs_img[:, :, 6]
            # decimal degrees, zero at horizontal
            self.surface_slope_from_DEM = self.obs_img[:, :, 7]
            # decimal degrees, clockwise from local north
            self.surface_aspect_from_DEM = self.obs_img[:, :, 8]
            # unitless, cosine of angle between to-sun and local DEM facet normal vectors
            self.local_cosine_i = self.obs_img[:, :, 9]
        else:
            warnings.warn(f'OBS file {self.obs_hdr} does not exist.')
        return self

    def fill_location(self, level, proj=''):
        self.get_info()
        if level == 2:
            self.loc_hdr = self.find_loc_header()
            if self.loc_hdr is None:
                return self
            self.loc_img = spec.io.envi.open(
                self.loc_hdr, image=self.find_loc_img())
            try:
                self.longitude = self.loc_img[:, :, 0]
                if proj == 'polar':
                    self._fill_polar()
                # recenter to [-LAT_DEG, LAT_DEG]
                self.longitude = self.longitude - LONG_DEG
                self.longitude[self.longitude < -LAT_DEG] += LONG_DEG

                self.latitude = self.loc_img[:, :, 1]
                self.radius = self.loc_img[:, :, 2]
            except:
                print('M3_Image: Could not fill location.')
                print('M3_Image:', self.img_hdr)
                print('M3 LOC HDR:', self.loc_hdr)
                print('M3 LOC IMG:', self.find_loc_img())
                pass
            finally:
                return self
        return self

    def _fill_polar(self):
        long = to_radians(self.longitude)
        lat = to_radians(self.latitude)
        self.r, self.theta = stereographic_projection(long, lat)
        return self

    def find_loc_img(self):
        file_split = self.img_hdr.split('/')[:]
        data_dir = '/'.join(file_split[:-9] + ['*'] + file_split[-8:-6])
        file_path = data_dir + '/' + \
            '/'.join(['CH1M3_0003', 'DATA', self.date_interval,
                     self.date[0:6], 'L1B'])
        headers = glob.glob(os.path.join(
            file_path, self.data_file + '*LOC.IMG'))
        if len(headers) == 0:
            warnings.warn('No LOC header found.')
            loc_img = None
        else:
            loc_img = headers[0]
        if len(headers) > 1:
            warnings.warn(
                'Found more than one LOC header, something is wrong.')
        return loc_img

    def find_loc_header(self):
        file_split = self.img_hdr.split('/')[:]
        data_dir = '/'.join(file_split[:-9] + ['*'] + file_split[-8:-6])
        file_path = data_dir + '/' + \
            '/'.join(['CH1M3_0003', 'DATA', self.date_interval,
                     self.date[0:6], 'L1B'])
        headers = glob.glob(os.path.join(
            file_path, self.data_file + '*LOC.HDR'))
        if len(headers) == 0:
            warnings.warn('No LOC header found.')
            loc_hdr = None
        else:
            loc_hdr = headers[0]
        if len(headers) > 1:
            warnings.warn(
                'Found more than one LOC header, something is wrong.')
        return loc_hdr

    def find_obs_header(self):
        file_split = self.img_hdr.split('/')[:]
        data_dir = '/'.join(file_split[:-6])
        file_path = data_dir + '/' + \
            '/'.join(['CH1M3_0003', 'DATA', self.date_interval,
                     self.date[0:6], 'L1B'])

        headers = glob.glob('/media/freya/**/*' +
                            self.data_file+'_V03_OBS.HDR', recursive=True)
        if len(headers) == 0:
            warnings.warn('No OBS header found.')
            obs_hdr = None
        else:
            obs_hdr = headers[0]
        if len(headers) > 1:
            warnings.warn(
                'Found more than one OBS header, something is wrong.')
        return obs_hdr

    def fill_dates(self):
        file_split = self.img_hdr.split('/')[:]
        file = file_split[-1]
        self.data_file = file.split('_')[0]
        self.date = self.img_hdr.split('.')[-2][-23:-15]  # file_split[-3]

        self.date_interval = file_split[-4]
        return self


def find(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)


def format_flagged_pixels(image):
    """
    Pixels with the value -999 is a flagged value as unusable data. This functions
    fills the values with NaNs instead.

    :param image:
    :return:
    """
    image[image == -999] = float('NaN')
    return image
