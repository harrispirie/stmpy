'''
Class to represent the RHK technology Inc. SM4 file and all attributes and methods to read the binary data

# Using this module:
import rhk_stmpy.rhk_sm4 as rhk  # Importing the rhk_sm4 module

# initializing
f = rhk.load_sm4("test_files/example.sm4")  # Loading .sm4 file
pg = f[6]  # assigning 7th page of .sm4 file

# metadata
pg_count = f.page_count  # number of pages as int
sum = f.info()  # summary of sm4 file pages as pandas data frame
attrs = pg.attrs  # page/data attributes as a dictionary

# data
coords = pg.coords
ramp = coords[1][1]  # x-axis ramping values as numpy array
data = pg.data  # data as numpy array
'''

import numpy as np
from enum import Enum


# Object type
class object_type(Enum):
    OBJECT_UNDEFINED = 0
    OBJECT_PAGE_INDEX_HEADER = 1
    OBJECT_PAGE_INDEX_ARRAY = 2
    OBJECT_PAGE_HEADER = 3
    OBJECT_PAGE_DATA = 4
    OBJECT_IMAGE_DRIFT_HEADER = 5
    OBJECT_IMAGE_DRIFT = 6
    OBJECT_SPEC_DRIFT_HEADER = 7
    OBJECT_SPEC_DRIFT_DATA = 8
    OBJECT_COLOR_= 9
    OBJECT_STRING_DATA = 10
    OBJECT_TIP_TRACK_HEADER = 11
    OBJECT_TIP_TRACK_DATA = 12
    OBJECT_PRM = 13
    OBJECT_THUMBNAIL = 14
    OBJECT_PRM_HEADER = 15
    OBJECT_THUMBNAIL_HEADER = 16
    OBJECT_API_INFO = 17
    OBJECT_HISTORY_INFO = 18
    OBJECT_PIEZO_SENSITIVITY = 19
    OBJECT_FREQUENCY_SWEEP_DATA = 20
    OBJECT_SCAN_PROCESSOR_INFO = 21
    OBJECT_PLL_INFO = 22
    OBJECT_CH1_DRIVE_INFO = 23
    OBJECT_CH2_DRIVE_INFO = 24
    OBJECT_LOCKIN0_INFO = 25
    OBJECT_LOCKIN1_INFO = 26
    OBJECT_ZPI_INFO = 27
    OBJECT_KPI_INFO = 28
    OBJECT_AUX_PI_INFO = 29
    OBJECT_LOWPASS_FILTER0_INFO = 30
    OBJECT_LOWPASS_FILTER1_INFO = 31


# Page Data type
class page_data_type(Enum):
    DATA_IMAGE = 0
    DATA_LINE = 1
    DATA_XY_DATA = 2
    DATA_ANNOTATED_LINE = 3
    DATA_TEXT = 4
    DATA_ANNOTATED_TEXT = 5
    DATA_SEQUENTIAL = 6
    DATA_MOVIE = 7


# Page Source type
class page_source_type(Enum):
    SOURCE_RAW = 0
    SOURCE_PROCESSED = 1
    SOURCE_CALCULATED = 2
    SOURCE_IMPORTED = 3


# Page type
class page_type(Enum):
    PAGE_UNDEFINED = 0
    PAGE_TOPOGRAPHIC = 1
    PAGE_CURRENT = 2
    PAGE_AUX = 3
    PAGE_FORCE = 4
    PAGE_SIGNAL = 5
    PAGE_FFT_TRANSFORM = 6
    PAGE_NOISE_POWER_SPECTRUM = 7
    PAGE_LINE_TEST = 8
    PAGE_OSCILLOSCOPE = 9
    PAGE_IV_SPECTRA = 10
    PAGE_IV_4x4 = 11
    PAGE_IV_8x8 = 12
    PAGE_IV_16x16 = 13
    PAGE_IV_32x32 = 14
    PAGE_IV_CENTER = 15
    PAGE_INTERACTIVE_SPECTRA = 16
    PAGE_AUTOCORRELATION = 17
    PAGE_IZ_SPECTRA = 18
    PAGE_4_GAIN_TOPOGRAPHY = 19
    PAGE_8_GAIN_TOPOGRAPHY = 20
    PAGE_4_GAIN_CURRENT = 21
    PAGE_8_GAIN_CURRENT = 22
    PAGE_IV_64x64 = 23
    PAGE_AUTOCORRELATION_SPECTRUM = 24
    PAGE_COUNTER = 25
    PAGE_MULTICHANNEL_ANALYSER = 26
    PAGE_AFM_100 = 27
    PAGE_CITS = 28
    PAGE_GPIB = 29
    PAGE_VIDEO_CHANNEL = 30
    PAGE_IMAGE_OUT_SPECTRA = 31
    PAGE_I_DATALOG = 32
    PAGE_I_ECSET = 33
    PAGE_I_ECDATA = 34
    PAGE_I_DSP_AD = 35
    PAGE_DISCRETE_SPECTROSCOPY_PP = 36
    PAGE_IMAGE_DISCRETE_SPECTROSCOPY = 37
    PAGE_RAMP_SPECTROSCOPY_RP = 38
    PAGE_DISCRETE_SPECTROSCOPY_RP = 39


# Line type
class line_type(Enum):
    LINE_NOT_A_LINE = 0
    LINE_HISTOGRAM = 1
    LINE_CROSS_SECTION = 2
    LINE_LINE_TEST = 3
    LINE_OSCILLOSCOPE = 4
    LINE_RESERVED = 5
    LINE_NOISE_POWER_SPECTRUM = 6
    LINE_IV_SPECTRUM = 7
    LINE_IZ_SPECTRUM = 8
    LINE_IMAGE_X_AVERAGE = 9
    LINE_IMAGE_Y_AVERAGE = 10
    LINE_NOISE_AUTOCORRELATION_SPECTRUM = 11
    LINE_MULTICHANNEL_ANALYSER_DATA = 12
    LINE_RENORMALIZED_IV = 13
    LINE_IMAGE_HISTOGRAM_SPECTRA = 14
    LINE_IMAGE_CROSS_SECTION = 15
    LINE_IMAGE_AVERAGE = 16
    LINE_IMAGE_CROSS_SECTION_G = 17
    LINE_IMAGE_OUT_SPECTRA = 18
    LINE_DATALOG_SPECTRUM = 19
    LINE_GXY = 20
    LINE_ELECTROCHEMISTRY = 21
    LINE_DISCRETE_SPECTROSCOPY = 22
    LINE_DATA_LOGGER = 23
    LINE_TIME_SPECTROSCOPY = 24
    LINE_ZOOM_FFT = 25
    LINE_FREQUENCY_SWEEP = 26
    LINE_PHASE_ROTATE = 27
    LINE_FIBER_SWEEP = 28


# Image type
class image_type(Enum):
    IMAGE_NORMAL = 0
    IMAGE_AUTOCORRELATED = 1


# Scan direction type
class scan_type(Enum):
    SCAN_RIGHT = 0
    SCAN_LEFT = 1
    SCAN_UP = 2
    SCAN_DOWN = 3


# Drift option type
class drift_option_type(Enum):
    DRIFT_DISABLED = 0
    DRIFT_EACH_SPECTRA = 1
    DRIFT_EACH_LOCATION = 2


# SM4 class definition
class RHKsm4:
    '''
    Main class that represents a RHK SM4 file
    Args: filename (name of the .sm4 file to be opened)
    '''

    def __init__(self, filename):
        # Open the file
        self._file = open(filename, 'rb')
        # Read the File Header
        self._header = RHKFileHeader(self)
        # Read Object list of File Header
        self._header._read_object_list(self)
        # Read Page Index Header
        self._page_index_header = RHKPageIndexHeader(self)
        # Read Object list of Page Index Header
        self._page_index_header._read_object_list(self)
        # Seek to the start position of the Page Index Array
        offset = self._page_index_header._get_offset('OBJECT_PAGE_INDEX_ARRAY')
        self._seek(offset, 0)
        # Read Page Index Array
        self._pages = []
        self.page_count = self._page_index_header.page_count
        for i in range(self._page_index_header.page_count):
            page = RHKPage(self)
            # Read Page Index
            self._pages.append(page)
            # Read Object list of Page Index
            page._read_object_list(self)
        # Read Pages content
        for page in self:
            page._read()
        # Close the file
        self._file.close()
        return

    def __getitem__(self, index):
        return self._pages[index]

    def _readb(self, dtype, count):
        # Read single line of file byte-wise
        return np.fromfile(self._file, dtype=dtype, count=count)[0]

    def _reads(self, count):
        # Read byte-wise *count* lines of the file and join as string
        string = ''.join([chr(i) for i in np.fromfile(self._file, dtype=np.uint16, count=count)])
        return string.rstrip('\x00')

    def _readstr(self):
        '''
        Read RHK string object
        Each string is written to file by first writing the string length (2 bytes),
        then the string. So when we read, first read a short value, which gives the
        string length, then read that much bytes which represents the string.
        '''
        length = self._readb(np.uint16, 1)  # first 2 bytes is the string length
        string = ''.join([chr(i) for i in np.fromfile(self._file, dtype=np.uint16, count=length)])
        return string.rstrip('\x00')

    def _readtime(self):
        '''
        Read RHK filetime object
        It is expressed in Windows epoch, a 64-bit value representing
        the number of 100-nanosecond intervals since January 1, 1601 (UTC).
        '''
        return np.fromfile(self._file, dtype=np.uint64, count=1)[0]

    def _seek(self, offset, whence):
        # Seek the file to the given position
        self._file.seek(offset, whence)

    def print_info(self):
        # Provide summary of .sm4 file attributes (pandas required)
        try:
            # importing pandas and setting max rows/columns to display
            import pandas as pd
            pd.set_option('display.max_rows', 15)
            pd.set_option('display.max_columns', 10)
        except:
            print("Error: pandas package not found.")
            return


        info = []
        for i in range(self.page_count):
            info.append([self[i].attrs['Label'],self[i].attrs['PageDataTypeName'], self[i].attrs['PageSourceTypeName'],
                         self[i].attrs['PageTypeName'], self[i].attrs['LineTypeName'],
                         self[i].attrs['ImageTypeName'], self[i].attrs['Xsize'],
                         self[i].attrs['Ysize'], self[i].attrs['Bias'],
                         self[i].attrs['Current']])
        table = pd.DataFrame(info, columns=["Label", "PageDataType", "PageSourceType", "PageType", "LineType", "ImageType", "XSize",                                                  "YSize", "Bias", "Current"])
        #print(table)  # remove to stop autoprint
        return table



class RHKObject:
    '''
    Define an RHK object, containing
    Object ID: (4 bytes) Type of data stored
    Offset: (4 bytes) Data offset
    Size: (4 bytes) size of the data
    Using the data offset and size, we can read the corresponding object data.
    '''

    def __init__(self, sm4):
        # Read the object properties.
        self.id = sm4._readb(np.uint32, 1)
        try:
            self.name = object_type(self.id).name
        except ValueError:
            self.name = 'OBJECT_UNKNOWN'
        self.offset = sm4._readb(np.uint32, 1)
        self.size = sm4._readb(np.uint32, 1)

        '''
        Seek to the end position of the current Object
        (for compatibility with future file versions
        in case Object Field Size is no longer 12 bytes)
        '''
        # sm4._seek(sm4._header.object_field_size - 12, 1)


class RHKObjectContainer:
    # Represents a class containing RHK Objects
    def _read_object_list(self, sm4):
        # Populate Object list

        self._object_list = []
        for i in range(self._object_list_count):
            self._object_list.append(RHKObject(sm4))

    def _get_offset(self, object_name):
        # Get offset of the given object
        for obj in self._object_list:
            if obj.name == object_name:
                return obj.offset


class RHKFileHeader(RHKObjectContainer):
    '''
    Class representing the File Header.
    The File Header contains the general information about the SM4 file
    and the file offset to other details like index header, PRM data etc.
    '''

    def __init__(self, sm4):
        '''
        Read the File Header.
        File header size: (2 bytes) the size for the actual File Header (in current version =56 bytes)
        File header content:
        Signature: (18x2 bytes) "STiMage 005.006 1". Mayor version.Minor version Unicode=1
        Total page count: (4 bytes) the basic structure is a page, where data is saved
        Object list count: (4 bytes) the count of Objects stored just after the file header (currently =3).
        Object field size: (4 bytes) the size of the Object structure (currently =12 bytes per Object)
        Reserved: (4x2 bytes) 2 fields reserved for future use.
        '''

        # File Header Size
        self.header_size = sm4._readb(np.uint16, 1)

        # File Header
        self.signature = sm4._reads(18)
        self.total_page_count = sm4._readb(np.uint32, 1)
        self._object_list_count = sm4._readb(np.uint32, 1)
        self.object_field_size = sm4._readb(np.uint32, 1)
        self.reserved = sm4._readb(np.uint32, 2)

        ''' 
        Seek to the end position of the File Header
        (for compatibility with future file versions
        in case File Header Size is no longer 56 bytes)
        '''
        sm4._seek(self.header_size + 2, 0)


class RHKPageIndexHeader(RHKObjectContainer):
    # Class representing the Page Index Header.

    def __init__(self, sm4):
        '''
        Read the Page Index Header.
        Page Index Header content:
        Page count: (4 bytes) Stores the number of pages in the Page Index Array
        Object List Count: Stores the count of Objects stored after Page Index Header (currently =1)
        Reserved: (4x2 bytes) 2 fields reserved for future use.
        Object List: Stores the Objects in the Page Index Header. Currently is stored one Object: Page Index Array
        '''

        # Seek to the position of the Page Index Header
        self.offset = sm4._header._get_offset('OBJECT_PAGE_INDEX_HEADER')
        sm4._seek(self.offset, 0)

        self.page_count = sm4._readb(np.uint32, 1)  # the number of pages in the page index array
        self._object_list_count = sm4._readb(np.uint32, 1)
        self.reserved = sm4._readb(np.uint32, 2)


class RHKPageHeader(RHKObjectContainer):
    # Class representing the Page Header

    def __init__(self, page, sm4):
        '''
        Read the Page Header
        The page header stores the header details of each page.
        It is followed by its Objects in the number given by 'object-list_count'.
        '''

        self.sm4 = sm4

        # Seek for the position of the Page Header
        self.offset = page._get_offset('OBJECT_PAGE_HEADER')
        self.sm4._seek(self.offset, 0)

        if (page._page_data_type == 6):  # "Sequential" Page Data type
            self.read_sequential_type(page)
        else:
            self.read_default_type(page)  # all other Page Data types

    def read_sequential_type(self, page):

        page.attrs['DataType'] = self.sm4._readb(np.uint32, 1)
        page.attrs['DataLength'] = self.sm4._readb(np.uint32, 1)
        page.attrs['ParamCount'] = self.sm4._readb(np.uint32, 1)

        self._object_list_count = self.sm4._readb(np.uint32, 1)

        page.attrs['DataInfoSize'] = self.sm4._readb(np.uint32, 1)
        page.attrs['DataInfoStringCount'] = self.sm4._readb(np.uint32, 1)

        # Adding manually these attributes for consistency with subsequent code
        page._page_type = 0
        page._line_type = 0
        page.attrs['PageType'] = page._page_type
        page.attrs['PageTypeName'] = page_type(page._page_type).name
        page.attrs['LineType'] = page._line_type
        page.attrs['LineTypeName'] = line_type(page._line_type).name
        page._page_data_size = page.attrs['ParamCount'] * (page.attrs['DataLength'] + 1)

    def read_default_type(self, page):

        _ = self.sm4._readb(np.uint16, 1)  # FieldSize
        page._string_count = self.sm4._readb(np.uint16, 1)

        page._page_type = self.sm4._readb(np.uint32, 1)
        page.attrs['PageType'] = page._page_type
        try:
            page.attrs['PageTypeName'] = page_type(page._page_type).name
        except ValueError:
            page.attrs['PageTypeName'] = 'PAGE_UNKNOWN'

        page.attrs['DataSubSource'] = self.sm4._readb(np.uint32, 1)

        page._line_type = self.sm4._readb(np.uint32, 1)
        page.attrs['LineType'] = page._line_type
        try:
            page.attrs['LineTypeName'] = line_type(page._line_type).name
        except ValueError:
            page.attrs['LineTypeName'] = 'LINE_UNKNOWN'

        page.attrs['Xcorner'] = self.sm4._readb(np.uint32, 1)
        page.attrs['Ycorner'] = self.sm4._readb(np.uint32, 1)

        # Xsize is number of pixels in X direction for an image page or the number of spectra stored in page.
        page.attrs['Xsize'] = self.sm4._readb(np.uint32, 1)

        # Ysize is number of pixels in Y direction for an image page or the number of spectra stored in page.
        page.attrs['Ysize'] = self.sm4._readb(np.uint32, 1)

        page._image_type = self.sm4._readb(np.uint32, 1)
        page.attrs['ImageType'] = page._image_type
        try:
            page.attrs['ImageTypeName'] = image_type(page._image_type).name
        except ValueError:
            page.attrs['ImageTypeName'] = 'IMAGE_UNKNOWN'

        page._scan_type = self.sm4._readb(np.uint32, 1)
        page.attrs['ScanType'] = page._scan_type
        try:
            page.attrs['ScanTypeName'] = scan_type(page._scan_type).name
        except ValueError:
            page.attrs['ScanTypeName'] = 'SCAN_UNKNOWN'

        page.attrs['GroupId'] = self.sm4._readb(np.uint32, 1)

        page._page_data_size = self.sm4._readb(np.uint32, 1)

        page.attrs['MinZvalue'] = self.sm4._readb(np.uint32, 1)
        page.attrs['MaxZvalue'] = self.sm4._readb(np.uint32, 1)
        page.attrs['Xscale'] = self.sm4._readb(np.float32, 1)
        page.attrs['Yscale'] = self.sm4._readb(np.float32, 1)
        page.attrs['Zscale'] = self.sm4._readb(np.float32, 1)
        page.attrs['XYscale'] = self.sm4._readb(np.float32, 1)
        page.attrs['Xoffset'] = self.sm4._readb(np.float32, 1)
        page.attrs['Yoffset'] = self.sm4._readb(np.float32, 1)
        page.attrs['Zoffset'] = self.sm4._readb(np.float32, 1)
        page.attrs['Period'] = self.sm4._readb(np.float32, 1)
        page.attrs['Bias'] = self.sm4._readb(np.float32, 1)
        page.attrs['Current'] = self.sm4._readb(np.float32, 1)
        page.attrs['Angle'] = self.sm4._readb(np.float32, 1)

        page._color_info_count = self.sm4._readb(np.uint32, 1)
        page.attrs['GridXsize'] = self.sm4._readb(np.uint32, 1)
        page.attrs['GridYsize'] = self.sm4._readb(np.uint32, 1)

        self._object_list_count = self.sm4._readb(np.uint32, 1)
        page._32bit_data_flag = self.sm4._readb(np.uint8, 1)
        page._reserved_flags = self.sm4._readb(np.uint8, 3)  # 3 bytes
        page._reserved = self.sm4._readb(np.uint8, 60)  # 60 bytes

    def read_objects(self, page):

        # Read Page Header objects
        self._read_object_list(self.sm4)

        # Add Data Info if "Sequential" Page Data type
        if (page._page_data_type == 6):

            ## Initialize metadata
            page.attrs['Sequential_ParamGain'] = []
            page.attrs['Sequential_ParamLabel'] = []
            page.attrs['Sequential_ParamUnit'] = []

            for i in range(page.attrs['ParamCount']):
                ## Parameter gain
                page.attrs['Sequential_ParamGain'].append(self.sm4._readb(np.float32, 1))
                ## Name of the parameter
                page.attrs['Sequential_ParamLabel'].append(self.sm4._readstr())
                ## Unit of the parameter
                page.attrs['Sequential_ParamUnit'].append(self.sm4._readstr())

        # Read each object and add to Page metadata
        for obj in self._object_list:
            self.read_object(page, obj)

    def read_object(self, page, obj):

        # Check if object position is valid then read it
        if obj.offset * obj.size != 0:
            if obj.id == 5:
                page._read_ImageDriftHeader()
            elif obj.id == 6:
                page._read_ImageDrift()
            elif obj.id == 7:
                page._read_SpecDriftHeader()
            elif obj.id == 8:
                page._read_SpecDriftData()
            elif obj.id == 9:
                # Color Info is skipped
                # page._read_ColorInfo()
                pass
            elif obj.id == 10:
                page._read_StringData()
            elif obj.id == 11:
                page._read_TipTrackHeader()
            elif obj.id == 12:
                page._read_TipTrackData()
            elif obj.id == 13:
                page._read_PRMData()
            elif obj.id == 15:
                page._read_PRMHeader()
            elif obj.id == 17:
                page._read_APIInfo()
            elif obj.id == 18:
                page._read_HistoryInfo()
            elif obj.id == 19:
                page._read_PiezoSensitivity()
            elif obj.id == 20:
                page._read_FrequencySweepData()
            elif obj.id == 21:
                page._read_ScanProcessorInfo()
            elif obj.id == 22:
                page._read_PLLInfo()
            elif obj.id == 23:
                page._read_ChannelDriveInfo('OBJECT_CH1_DRIVE_INFO', 'CH1Drive')
            elif obj.id == 24:
                page._read_ChannelDriveInfo('OBJECT_CH2_DRIVE_INFO', 'CH2Drive')
            elif obj.id == 25:
                page._read_LockinInfo('OBJECT_LOCKIN0_INFO', 'Lockin0')
            elif obj.id == 26:
                page._read_LockinInfo('OBJECT_LOCKIN1_INFO', 'Lockin1')
            elif obj.id == 27:
                page._read_PIControllerInfo('OBJECT_ZPI_INFO', 'ZPI')
            elif obj.id == 28:
                page._read_PIControllerInfo('OBJECT_KPI_INFO', 'KPI')
            elif obj.id == 29:
                page._read_PIControllerInfo('OBJECT_AUX_PI_INFO', 'AuxPI')
            elif obj.id == 30:
                page._read_LowPassFilterInfo('OBJECT_LOWPASS_FILTER0_INFO', 'LowPassFilter0')
            elif obj.id == 31:
                page._read_LowPassFilterInfo('OBJECT_LOWPASS_FILTER1_INFO', 'LowPassFilter1')


class RHKPage(RHKObjectContainer):
    # Class representing Page

    def __init__(self, sm4):
        '''
        Read the Page Index
        
        Content:
            Page ID: Unique GUID for each Page
            Page Data Type: The type of data stored with the page.
            Page Source Type: Identifies the page source type.
            Object List Count: Stores the count of Objects stored after each Page Index
            Minor Version: (4 bytes) stores the minor version of the file (2 in QP,
                4 in XPMPro, 6 in Rev9)
            Object List: Stores the Objects in the Page Index. Currently we are storing:
                1. Page Header
                2. Page Data
                3. Thumbnail
                4. Thumbnail header
        '''

        self._sm4 = sm4

        # Initialize Page Index and Page meta dictionaries
        self.attrs = {}

        self.attrs['PageID'] = sm4._readb(np.uint16, 8)
        self.name = "ID" + str(self.attrs['PageID'])

        self._page_data_type = sm4._readb(np.uint32, 1)
        self.attrs['PageDataType'] = self._page_data_type
        try:
            self.attrs['PageDataTypeName'] = page_data_type(self._page_data_type).name
        except ValueError:
            self.attrs['PageDataTypeName'] = 'DATA_UNKNOWN'

        self._page_source_type = sm4._readb(np.uint32, 1)
        self.attrs['PageSourceType'] = self._page_source_type
        try:
            self.attrs['PageSourceTypeName'] = page_source_type(self._page_source_type).name
        except ValueError:
            self.attrs['PageSourceTypeName'] = 'SOURCE_UNKNOWN'

        self._object_list_count = sm4._readb(np.uint32, 1)
        self.attrs['MinorVer'] = sm4._readb(np.uint32, 1)

        # Add signature from File Header
        self.attrs['Signature'] = sm4._header.signature


    def _read(self):
        # Read the Page Header and Page Data. Thumbnail and Thumbnail Header are discarded

        # Read Page Header
        self._header = RHKPageHeader(self, self._sm4)
        self._header.read_objects(self)

        # Read Page Data
        self._read_data()

    def _read_data(self):
        # Read Page Data

        # Seek for the position of the Page Data
        offset = self._get_offset('OBJECT_PAGE_DATA')
        self._sm4._seek(offset, 0)

        # Load data, selecting float or long integer type
        data_size = int(self._page_data_size / 4)

        if (self._line_type in [1, 6, 9, 10, 11, 13, 18, 19, 21, 22] or self._page_data_type == 6):
            raw_data = np.fromfile(self._sm4._file, dtype=np.float32, count=data_size)
            '''
            For Sequential_data page, the page data contains an array of size ‘n’ with ‘m’ elements is stored. 
            Where m is the Param count and n is the Data length (array size) stored in the page header. 
            The first float data in each element represents the output values.
            '''
        else: ###
            raw_data = np.fromfile(self._sm4._file, dtype=np.int32, count=data_size) ###
            raw_data = np.float32(raw_data) * self.attrs['Zscale'] + self.attrs['Zoffset']

        # Reshape and store data
        self.data, self.coords = self._reshape_data(raw_data)

    def _reshape_data(self, raw_data):
        # Reshape data of the page and create its coordinates
        xsize = self.attrs['Xsize']
        ysize = self.attrs['Ysize']
        xscale = self.attrs['Xscale']
        yscale = self.attrs['Yscale']

        # Reshape data
        if self._page_data_type == 0:  # Image type

            data = raw_data.reshape(xsize, ysize)

            # Check inversion of piezo sensitivity and adjust accordingly data orientation
            xsens = self.attrs['PiezoSensitivity_TubeX']
            ysens = self.attrs['PiezoSensitivity_TubeY']
            if xsens < 0:
                xscale *= -1
                data = np.flip(data, axis=1)
            if ysens < 0:
                yscale *= -1
                data = np.flip(data, axis=0)

            coords = [('y', yscale * np.arange(ysize, dtype=np.float64)),
                      ('x', xscale * np.arange(xsize, dtype=np.float64))]

            # Check if coordinates are positive, if not change sign and flip
            if xscale < 0:
                coords[1] = ('x', -np.flip(coords[1][1]))
            if yscale < 0:
                coords[0] = ('y', -np.flip(coords[0][1]))

        elif self._page_data_type == 1:  # Line type
            data = raw_data.reshape(ysize, xsize)  # raw_data.reshape(xsize, ysize).transpose()
            xoffset = self.attrs['Xoffset']
            coords = [('y', int(yscale) * np.arange(ysize, dtype=np.uint32)),
                      ('x', xscale * np.arange(xsize, dtype=np.float64) + xoffset)]

            if self._line_type == 22:  # Discrete spectroscopy has shape xsize * (ysize + 1)
                tmp = raw_data.reshape(xsize, ysize + 1).transpose()
                coords[1] = ('x', tmp[0])
                data = tmp[1:]

        else:
            data = raw_data
            coords = [('x', np.arange(xsize * ysize, dtype=np.uint32))]

        return data, coords

    def _read_StringData(self):
        # Read String Data for the current page. _string_count gives the number of strings in the current page.
        offset = self._header._get_offset('OBJECT_STRING_DATA')
        self._sm4._seek(offset, 0)

        # Create string labels list, adding any additional (at date unknown) label
        strList = ["Label",
                   "SystemText",
                   "SessionText",
                   "UserText",
                   "_path",
                   "Date",
                   "Time",
                   "Xunits",
                   "Yunits",
                   "Zunits",
                   "Xlabel",
                   "Ylabel",
                   "StatusChannelText",
                   "CompletedLineCount",
                   "OverSamplingCount",
                   "SlicedVoltage",
                   "PLLProStatus",
                   "SetpointUnit",
                   "CHlist"]
        for i in range(self._string_count - 19):
            strList.append('Unknown' + "{:0>3d}".format(i))

        # Actual read of the strings
        for k in range(self._string_count):
            if k == 4:  # file path
                self._path = self._sm4._readstr()
            elif k in [13, 14]:  # conversion to integer
                self.attrs[strList[k]] = int(self._sm4._readstr())
            elif k == 18:  # parse CHDriveValues string
                CHlist = self._sm4._readstr().split("\n")
                for i, CH in enumerate(CHlist):
                    self.attrs["CH" + str(i + 1) + "DriveValue"] = float(CH.split(" ")[3])
                    self.attrs["CH" + str(i + 1) + "DriveValueUnits"] = CH.split(" ")[4]
            else:
                self.attrs[strList[k]] = self._sm4._readstr()

        # Create ISO8601 datetime stamp
        mm, dd, yy = self.attrs['Date'].split('/')
        datetime = '20' + yy + '-' + mm + '-' + dd + 'T' + self.attrs['Time'] + '.000'
        self.attrs['DateTime'] = datetime

        # Add line type units based on line_type enum class
        line_type_xunits = {'LINE_NOT_A_LINE': '',
                            'LINE_HISTOGRAM': '',
                            'LINE_CROSS_SECTION': '',
                            'LINE_LINE_TEST': '',
                            'LINE_OSCILLOSCOPE': '',
                            'LINE_RESERVED': '',
                            'LINE_NOISE_POWER_SPECTRUM': '',
                            'LINE_IV_SPECTRUM': 'Bias',
                            'LINE_IZ_SPECTRUM': 'Z',
                            'LINE_IMAGE_X_AVERAGE': '',
                            'LINE_IMAGE_Y_AVERAGE': '',
                            'LINE_NOISE_AUTOCORRELATION_SPECTRUM': '',
                            'LINE_MULTICHANNEL_ANALYSER_DATA': '',
                            'LINE_RENORMALIZED_IV': '',
                            'LINE_IMAGE_HISTOGRAM_SPECTRA': '',
                            'LINE_IMAGE_CROSS_SECTION': '',
                            'LINE_IMAGE_AVERAGE': '',
                            'LINE_IMAGE_CROSS_SECTION_G': '',
                            'LINE_IMAGE_OUT_SPECTRA': '',
                            'LINE_DATALOG_SPECTRUM': '',
                            'LINE_GXY': '',
                            'LINE_ELECTROCHEMISTRY': '',
                            'LINE_DISCRETE_SPECTROSCOPY': '',
                            'LINE_DATA_LOGGER': '',
                            'LINE_TIME_SPECTROSCOPY': 'Time',
                            'LINE_ZOOM_FFT': '',
                            'LINE_FREQUENCY_SWEEP': '',
                            'LINE_PHASE_ROTATE': '',
                            'LINE_FIBER_SWEEP': ''}

        if self.attrs['Xlabel'] == '':
            self.attrs['Xlabel'] = line_type_xunits[self.attrs['LineTypeName']]

    def _read_SpecDriftHeader(self):
        # Read Spec Drift Header for the current page.

        offset = self._header._get_offset('OBJECT_SPEC_DRIFT_HEADER')
        self._sm4._seek(offset, 0)

        self.attrs['SpecDrift_Filetime'] = self._sm4._readtime()
        self.attrs['SpecDrift_DriftOptionType'] = self._sm4._readb(np.uint32, 1)
        self.attrs['SpecDrift_DriftOptionTypeName'] = drift_option_type(self.attrs['SpecDrift_DriftOptionType']).name
        _ = self._sm4._readb(np.uint32, 1)  # SpecDrift StringCount
        self.attrs['SpecDrift_Channel'] = self._sm4._readstr()

    def _read_SpecDriftData(self):
        # Read Spec Drift Data for the current page.

        offset = self._header._get_offset('OBJECT_SPEC_DRIFT_DATA')
        self._sm4._seek(offset, 0)

        self.attrs['SpecDrift_Time'] = []
        self.attrs['SpecDrift_Xcoord'] = []
        self.attrs['SpecDrift_Ycoord'] = []
        self.attrs['SpecDrift_dX'] = []
        self.attrs['SpecDrift_dY'] = []
        self.attrs['SpecDrift_CumulativeX'] = []
        self.attrs['SpecDrift_CumulativeY'] = []

        for k in range(self.attrs['Ysize']):
            self.attrs['SpecDrift_Time'].append(self._sm4._readb(np.float32, 1))
            self.attrs['SpecDrift_Xcoord'].append(self._sm4._readb(np.float32, 1))
            self.attrs['SpecDrift_Ycoord'].append(self._sm4._readb(np.float32, 1))
            self.attrs['SpecDrift_dX'].append(self._sm4._readb(np.float32, 1))
            self.attrs['SpecDrift_dY'].append(self._sm4._readb(np.float32, 1))
            self.attrs['SpecDrift_CumulativeX'].append(self._sm4._readb(np.float32, 1))
            self.attrs['SpecDrift_CumulativeY'].append(self._sm4._readb(np.float32, 1))

    def _read_ImageDriftHeader(self):
        # Read Image Drift Header for the current page.

        offset = self._header._get_offset('OBJECT_IMAGE_DRIFT_HEADER')
        self._sm4._seek(offset, 0)

        self.attrs['ImageDrift_Filetime'] = self._sm4._readtime()
        self.attrs['ImageDrift_DriftOptionType'] = self._sm4._readb(np.uint32, 1)
        self.attrs['ImageDrift_DriftOptionTypeName'] = drift_option_type(self.attrs['ImageDrift_DriftOptionType']).name

    def _read_ImageDrift(self):
        # Read Image Drift for the current page.

        offset = self._header._get_offset('OBJECT_IMAGE_DRIFT')
        self._sm4._seek(offset, 0)

        self.attrs['ImageDrift_Time'] = self._sm4._readb(np.float32, 1)
        self.attrs['ImageDrift_dX'] = self._sm4._readb(np.float32, 1)
        self.attrs['ImageDrift_dY'] = self._sm4._readb(np.float32, 1)
        self.attrs['ImageDrift_CumulativeX'] = self._sm4._readb(np.float32, 1)
        self.attrs['ImageDrift_CumulativeY'] = self._sm4._readb(np.float32, 1)
        self.attrs['ImageDrift_VectorX'] = self._sm4._readb(np.float32, 1)
        self.attrs['ImageDrift_VectorY'] = self._sm4._readb(np.float32, 1)

    def _read_ColorInfo(self):
        # Read Color Info for the current page. Color Info is only for use into RHK DAW software.

        offset = self._header._get_offset('OBJECT_COLOR_INFO')
        self._sm4._seek(offset, 0)

        # Initialize metadata
        self.attrs['Color_StructSize'] = []
        self.attrs['Color_Reserved'] = []

        # HSVColor
        self.attrs['Color_Hstart'] = []
        self.attrs['Color_Sstart'] = []
        self.attrs['Color_Vstart'] = []
        self.attrs['Color_Hstop'] = []
        self.attrs['Color_Sstop'] = []
        self.attrs['Color_Vstop'] = []

        self.attrs['Color_ClrDirection'] = []
        self.attrs['Color_NumEntries'] = []
        self.attrs['Color_StartSlidePos'] = []
        self.attrs['Color_EndSlidePos'] = []

        # Color Transform
        self.attrs['Color_Gamma'] = []
        self.attrs['Color_Alpha'] = []
        self.attrs['Color_Xstart'] = []
        self.attrs['Color_Xstop'] = []
        self.attrs['Color_Ystart'] = []
        self.attrs['Color_Ystop'] = []
        self.attrs['Color_MappingMode'] = []
        self.attrs['Color_Invert'] = []

        for k in range(self._color_info_count):
            self.attrs['Color_StructSize'].append(self._sm4._readb(np.uint16, 1))
            self.attrs['Color_Reserved'].append(self._sm4._readb(np.uint16, 1))

            # HSVColor
            self.attrs['Color_Hstart'].append(self._sm4._readb(np.float32, 1))
            self.attrs['Color_Sstart'].append(self._sm4._readb(np.float32, 1))
            self.attrs['Color_Vstart'].append(self._sm4._readb(np.float32, 1))
            self.attrs['Color_Hstop'].append(self._sm4._readb(np.float32, 1))
            self.attrs['Color_Sstop'].append(self._sm4._readb(np.float32, 1))
            self.attrs['Color_Vstop'].append(self._sm4._readb(np.float32, 1))

            self.attrs['Color_ClrDirection'].append(self._sm4._readb(np.uint32, 1))
            self.attrs['Color_NumEntries'].append(self._sm4._readb(np.uint32, 1))
            self.attrs['Color_StartSlidePos'].append(self._sm4._readb(np.float32, 1))
            self.attrs['Color_EndSlidePos'].append(self._sm4._readb(np.float32, 1))

            # Color Transform
            self.attrs['Color_Gamma'].append(self._sm4._readb(np.float32, 1))
            self.attrs['Color_Alpha'].append(self._sm4._readb(np.float32, 1))
            self.attrs['Color_Xstart'].append(self._sm4._readb(np.float32, 1))
            self.attrs['Color_Xstop'].append(self._sm4._readb(np.float32, 1))
            self.attrs['Color_Ystart'].append(self._sm4._readb(np.float32, 1))
            self.attrs['Color_Ystop'].append(self._sm4._readb(np.float32, 1))
            self.attrs['Color_MappingMode'].append(self._sm4._readb(np.uint32, 1))
            self.attrs['Color_Invert'].append(self._sm4._readb(np.uint32, 1))

    def _read_TipTrackHeader(self):
        # Read Tip track Header for the current page.

        offset = self._header._get_offset('OBJECT_TIP_TRACK_HEADER')
        self._sm4._seek(offset, 0)

        self.attrs['TipTrack_Filetime'] = self._sm4._readtime()
        self.attrs['TipTrack_FeatureHeight'] = self._sm4._readb(np.float32, 1)
        self.attrs['TipTrack_FeatureWidth'] = self._sm4._readb(np.float32, 1)
        self.attrs['TipTrack_TimeConstant'] = self._sm4._readb(np.float32, 1)
        self.attrs['TipTrack_CycleRate'] = self._sm4._readb(np.float32, 1)
        self.attrs['TipTrack_PhaseLag'] = self._sm4._readb(np.float32, 1)
        _ = self._sm4._readb(np.uint32, 1)  # TipTrack StringCount
        self.attrs['TipTrack_TipTrackInfoCount'] = self._sm4._readb(np.uint32, 1)
        self.attrs["TipTrack_Channel"] = self._sm4._readstr()

    def _read_TipTrackData(self):
        # Read Tip Track Data for the current page.

        offset = self._header._get_offset('OBJECT_TIP_TRACK_DATA')
        self._sm4._seek(offset, 0)

        self.attrs['TipTrack_CumulativeTime'] = []
        self.attrs['TipTrack_Time'] = []
        self.attrs['TipTrack_dX'] = []
        self.attrs['TipTrack_dY'] = []

        for k in range(self.attrs['TipTrack_TipTrackInfoCount']):
            self.attrs['TipTrack_CumulativeTime'].append(self._sm4._readb(np.float32, 1))
            self.attrs['TipTrack_Time'].append(self._sm4._readb(np.float32, 1))
            self.attrs['TipTrack_dX'].append(self._sm4._readb(np.float32, 1))
            self.attrs['TipTrack_dY'].append(self._sm4._readb(np.float32, 1))

    def _read_PRMData(self):
        '''
        Read PRM Data for the current page.
        Valid only for RHK XPMPro generated files.
        PRM data could be compressed with Zlib.
        '''

        offset = self._header._get_offset('OBJECT_PRM')
        self._sm4._seek(offset, 0)

        if self.attrs['PRM_CompressionFlag'] == 0:
            dataSize = self.attrs['PRM_DataSize']
        elif self.attrs['PRM_CompressionFlag'] == 1:
            dataSize = self.attrs['PRM_CompressionSize']

        self.PRMdata = np.fromfile(self._sm4._file, dtype=np.uint32, count=dataSize)

    def _read_PRMHeader(self):
        # Read PRM Header for the current page. Valid only for RHK XPMPro generated files.

        offset = self._header._get_offset('OBJECT_PRM_HEADER')
        self._sm4._seek(offset, 0)

        self.attrs['PRM_CompressionFlag'] = self._sm4._readb(np.uint32, 1)
        self.attrs['PRM_DataSize'] = self._sm4._readb(np.uint64, 1)
        self.attrs['PRM_CompressionSize'] = self._sm4._readb(np.uint64, 1)

    def _read_APIInfo(self):
        # Read API Info for the current page.

        offset = self._header._get_offset('OBJECT_API_INFO')
        self._sm4._seek(offset, 0)

        self.attrs['API_VoltageHigh'] = self._sm4._readb(np.float32, 1)
        self.attrs['API_VoltageLow'] = self._sm4._readb(np.float32, 1)
        self.attrs['API_Gain'] = self._sm4._readb(np.float32, 1)
        self.attrs['API_Offset'] = self._sm4._readb(np.float32, 1)

        self.attrs['API_RampMode'] = self._sm4._readb(np.uint32, 1)
        self.attrs['API_RampType'] = self._sm4._readb(np.uint32, 1)
        self.attrs['API_Step'] = self._sm4._readb(np.uint32, 1)
        self.attrs['API_ImageCount'] = self._sm4._readb(np.uint32, 1)
        self.attrs['API_DAC'] = self._sm4._readb(np.uint32, 1)
        self.attrs['API_MUX'] = self._sm4._readb(np.uint32, 1)
        self.attrs['API_STMBias'] = self._sm4._readb(np.uint32, 1)

        _ = self._sm4._readb(np.uint32, 1)  # API StringCount

        self.attrs['API_Units'] = self._sm4._readstr()

    def _read_HistoryInfo(self):
        # Read History Info for the current page.

        offset = self._header._get_offset('OBJECT_HISTORY_INFO')
        self._sm4._seek(offset, 0)

        _ = self._sm4._readb(np.uint32, 1)  # History StringCount
        _ = self._sm4._readstr()  # History Path
        _ = self._sm4._readstr()  # History Pixel2timeFile

    def _read_PiezoSensitivity(self):
        # Read Piezo Sensitivity for the current page.

        offset = self._header._get_offset('OBJECT_PIEZO_SENSITIVITY')
        self._sm4._seek(offset, 0)

        self.attrs['PiezoSensitivity_TubeX'] = self._sm4._readb(np.float64, 1)
        self.attrs['PiezoSensitivity_TubeY'] = self._sm4._readb(np.float64, 1)
        self.attrs['PiezoSensitivity_TubeZ'] = self._sm4._readb(np.float64, 1)
        self.attrs['PiezoSensitivity_TubeZOffset'] = self._sm4._readb(np.float64, 1)
        self.attrs['PiezoSensitivity_ScanX'] = self._sm4._readb(np.float64, 1)
        self.attrs['PiezoSensitivity_ScanY'] = self._sm4._readb(np.float64, 1)
        self.attrs['PiezoSensitivity_ScanZ'] = self._sm4._readb(np.float64, 1)
        self.attrs['PiezoSensitivity_Actuator'] = self._sm4._readb(np.float64, 1)

        _ = self._sm4._readb(np.uint32, 1)  # PiezoSensitivity StringCount

        self.attrs['PiezoSensitivity_TubeXUnit'] = self._sm4._readstr()
        self.attrs['PiezoSensitivity_TubeYUnit'] = self._sm4._readstr()
        self.attrs['PiezoSensitivity_TubeZUnit'] = self._sm4._readstr()
        self.attrs['PiezoSensitivity_TubeZOffsetUnit'] = self._sm4._readstr()
        self.attrs['PiezoSensitivity_ScanXUnit'] = self._sm4._readstr()
        self.attrs['PiezoSensitivity_ScanYUnit'] = self._sm4._readstr()
        self.attrs['PiezoSensitivity_ScanZUnit'] = self._sm4._readstr()
        self.attrs['PiezoSensitivity_ActuatorUnit'] = self._sm4._readstr()
        self.attrs['PiezoSensitivity_TubeCalibration'] = self._sm4._readstr()
        self.attrs['PiezoSensitivity_ScanCalibration'] = self._sm4._readstr()
        self.attrs['PiezoSensitivity_ActuatorCalibration'] = self._sm4._readstr()

    def _read_FrequencySweepData(self):
        # Read Frequency Sweep Data for the current page.

        offset = self._header._get_offset('OBJECT_FREQUENCY_SWEEP_DATA')
        self._sm4._seek(offset, 0)

        self.attrs['FrequencySweep_PSDTotalSignal'] = self._sm4._readb(np.float64, 1)
        self.attrs['FrequencySweep_PeakFrequency'] = self._sm4._readb(np.float64, 1)
        self.attrs['FrequencySweep_PeakAmplitude'] = self._sm4._readb(np.float64, 1)
        self.attrs['FrequencySweep_DriveAmplitude'] = self._sm4._readb(np.float64, 1)
        self.attrs['FrequencySweep_Signal2DriveRatio'] = self._sm4._readb(np.float64, 1)
        self.attrs['FrequencySweep_QFactor'] = self._sm4._readb(np.float64, 1)

        _ = self._sm4._readb(np.uint32, 1)  # FrequencySweep StringCount

        self.attrs['FrequencySweep_TotalSignalUnit'] = self._sm4._readstr()
        self.attrs['FrequencySweep_PeakFrequencyUnit'] = self._sm4._readstr()
        self.attrs['FrequencySweep_PeakAmplitudeUnit'] = self._sm4._readstr()
        self.attrs['FrequencySweep_DriveAmplitudeUnit'] = self._sm4._readstr()
        self.attrs['FrequencySweep_Signal2DriveRatioUnit'] = self._sm4._readstr()
        self.attrs['FrequencySweep_QFactorUnit'] = self._sm4._readstr()

    def _read_ScanProcessorInfo(self):
        # Read Scan Processor Info for the current page.

        offset = self._header._get_offset('OBJECT_SCAN_PROCESSOR_INFO')
        self._sm4._seek(offset, 0)

        self.attrs['ScanProcessor_XSlopeCompensation'] = self._sm4._readb(np.float64, 1)
        self.attrs['ScanProcessor_YSlopeCompensation'] = self._sm4._readb(np.float64, 1)

        _ = self._sm4._readb(np.uint32, 1)  # ScanProcessor StringCount

        self.attrs['ScanProcessor_XSlopeCompensationUnit'] = self._sm4._readstr()
        self.attrs['ScanProcessor_YSlopeCompensationUnit'] = self._sm4._readstr()

    def _read_PLLInfo(self):
        # Read PLL Info for the current page.

        offset = self._header._get_offset('OBJECT_PLL_INFO')
        self._sm4._seek(offset, 0)

        _ = self._sm4._readb(np.uint32, 1)  # PLL StringCount
        self.attrs['PLL_AmplitudeControl'] = self._sm4._readb(np.uint32, 1)
        self.attrs['PLL_DriveAmplitude'] = self._sm4._readb(np.float64, 1)
        self.attrs['PLL_DriveRefFrequency'] = self._sm4._readb(np.float64, 1)
        self.attrs['PLL_LockinFreqOffset'] = self._sm4._readb(np.float64, 1)
        self.attrs['PLL_LockinHarmonicFactor'] = self._sm4._readb(np.float64, 1)
        self.attrs['PLL_LockinPhaseOffset'] = self._sm4._readb(np.float64, 1)
        self.attrs['PLL_PIGain'] = self._sm4._readb(np.float64, 1)
        self.attrs['PLL_PIIntCutoffFreq'] = self._sm4._readb(np.float64, 1)
        self.attrs['PLL_PILowerBound'] = self._sm4._readb(np.float64, 1)
        self.attrs['PLL_PIUpperBound'] = self._sm4._readb(np.float64, 1)
        self.attrs['PLL_DissPIGain'] = self._sm4._readb(np.float64, 1)
        self.attrs['PLL_DissPIIntCutoffFreq'] = self._sm4._readb(np.float64, 1)
        self.attrs['PLL_DissPILowerBound'] = self._sm4._readb(np.float64, 1)
        self.attrs['PLL_DissPIUpperBound'] = self._sm4._readb(np.float64, 1)
        self.attrs['PLL_LockinFilterCutoffFreq'] = self._sm4._readstr()
        self.attrs['PLL_DriveAmplitudeUnit'] = self._sm4._readstr()
        self.attrs['PLL_DriveFrequencyUnit'] = self._sm4._readstr()
        self.attrs['PLL_LockinFreqOffsetUnit'] = self._sm4._readstr()
        self.attrs['PLL_LockinPhaseUnit'] = self._sm4._readstr()
        self.attrs['PLL_PIGainUnit'] = self._sm4._readstr()
        self.attrs['PLL_PIICFUnit'] = self._sm4._readstr()
        self.attrs['PLL_PIOutputUnit'] = self._sm4._readstr()
        self.attrs['PLL_DissPIGainUnit'] = self._sm4._readstr()
        self.attrs['PLL_DissPIICFUnit'] = self._sm4._readstr()
        self.attrs['PLL_DissPIOutputUnit'] = self._sm4._readstr()

    def _read_ChannelDriveInfo(self, obj, metaString):
        # Read Channel Drive Info for the current page.

        offset = self._header._get_offset(obj)
        self._sm4._seek(offset, 0)

        _ = self._sm4._readb(np.uint32, 1)  # ChannelDrive StringCount
        self.attrs[metaString + '_MasterOscillator'] = self._sm4._readb(np.uint32, 1)

        self.attrs[metaString + '_Amplitude'] = self._sm4._readb(np.float64, 1)
        self.attrs[metaString + '_Frequency'] = self._sm4._readb(np.float64, 1)
        self.attrs[metaString + '_PhaseOffset'] = self._sm4._readb(np.float64, 1)
        self.attrs[metaString + '_HarmonicFactor'] = self._sm4._readb(np.float64, 1)

        self.attrs[metaString + '_AmplitudeUnit'] = self._sm4._readstr()
        self.attrs[metaString + '_FrequencyUnit'] = self._sm4._readstr()
        self.attrs[metaString + '_PhaseOffsetUnit'] = self._sm4._readstr()
        self.attrs[metaString + '_ReservedUnit'] = self._sm4._readstr()

    def _read_LockinInfo(self, obj, metaString):
        # Read Lockin Info for the current page.

        offset = self._header._get_offset(obj)
        self._sm4._seek(offset, 0)
        _ = self._sm4._readb(np.uint32, 1)  # LockinInfo StringCount
        self.attrs[metaString + '_NonMasterOscillator'] = self._sm4._readb(np.uint32, 1)
        self.attrs[metaString + '_Frequency'] = self._sm4._readb(np.float64, 1)
        self.attrs[metaString + '_HarmonicFactor'] = self._sm4._readb(np.float64, 1)
        self.attrs[metaString + '_PhaseOffset'] = self._sm4._readb(np.float64, 1)
        self.attrs[metaString + '_FilterCutoffFrequency'] = self._sm4._readb(np.float64, 1)
        self.attrs[metaString + '_FreqUnit'] = self._sm4._readstr()
        self.attrs[metaString + '_PhaseUnit'] = self._sm4._readstr()

    def _read_PIControllerInfo(self, obj, metaString):
        # Read PI Controller Info for the current page.

        offset = self._header._get_offset(obj)
        self._sm4._seek(offset, 0)
        self.attrs[metaString + '_SetPoint'] = self._sm4._readb(np.float64, 1)
        self.attrs[metaString + '_ProportionalGain'] = self._sm4._readb(np.float64, 1)
        self.attrs[metaString + '_IntegralGain'] = self._sm4._readb(np.float64, 1)
        self.attrs[metaString + '_LowerBound'] = self._sm4._readb(np.float64, 1)
        self.attrs[metaString + '_UpperBound'] = self._sm4._readb(np.float64, 1)

        _ = self._sm4._readb(np.uint32, 1)  # PIController StringCount

        self.attrs[metaString + '_FeedbackType'] = self._sm4._readstr()
        self.attrs[metaString + '_SetPointUnit'] = self._sm4._readstr()
        self.attrs[metaString + '_ProportionalGainUnit'] = self._sm4._readstr()
        self.attrs[metaString + '_IntegralGainUnit'] = self._sm4._readstr()
        self.attrs[metaString + '_OutputUnit'] = self._sm4._readstr()

    def _read_LowPassFilterInfo(self, obj, metaString):
        # Read Low-Pass Filter Info for the current page.

        offset = self._header._get_offset(obj)
        self._sm4._seek(offset, 0)

        _ = self._sm4._readb(np.uint32, 1)  # LowPassFilter StringCount
        freq, units = self._sm4._readstr().split(" ")
        self.attrs[metaString + '_CutoffFrequency'] = float(freq)
        self.attrs[metaString + '_CutoffFrequencyUnits'] = units


# Methods: load_sm4

def load_sm4(sm4file):
    '''
    To load data and metadata from RHK .sm4 file.
    Args: sm4file (name of the .sm4 file to be loaded)
    Returns a container for the pages in the .sm4 file with their data and metadata
    '''
    return RHKsm4(sm4file)

