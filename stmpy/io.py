from __future__ import print_function  # python v2.6 and higher
'''
STMPY I/O Version 1.0

Read and write common file types.

Contents:
    load()  -   Load supported data into python
    save()  -   Save python data to disk.

Version history:
    1.0     2018-03-02  - HP : Initial release.
    1.1     2018-10-10  - HP : Python 3 compatibility

TO DO:
    - Add support for .mat files
    - Rewrite load_3ds() to buffer data and improve efficiency.
'''

__version__ = 1.1

import stmpy
from stmpy import matio
import numpy as np
import scipy.io as sio
import os
import re
import sys

from struct import pack, unpack, calcsize
from datetime import datetime, timedelta
from scipy.optimize import minimize

PY2 = sys.version_info.major == 2
PY3 = sys.version_info.major == 3


def load(filePath, biasOffset=True, niceUnits=False):
    '''
    Loads data into python. Please include the file extension in the path.

    Supported extensions:
        .spy    -   STMPY generic data format.
        .3ds    -   NANONIS grid data. Commonly used for DOS maps.
        .sxm    -   NANONIS scan data. Commonly used for topographys.
        .dat    -   NANONIS data file. Commonly used for bias spectroscopy.
        .nsp    -   NANONIS long term spectum data type.
        .nvi    -   NISTview image data, used for topography data.
        .nvl    -   NISTview layer data, used for 3D DOS maps.
        .asc    -   ASCII file type.


    For .3ds and .dat file types there is an optional flag to correct for bias offset
    that is true by default.  This does not correct for a current offset, and
    should not be used in cases where there is a significant current offset.
    Note: .mat files are supported as exports from STMView only.

    Inputs:
        filePath    - Required : Path to file including extension.
        baisOffset  - Optional : Corrects didv data for bias offset by looking
                                 for where the current is zero.
        niceUnits   - Optional : Put lock-in channel units as nS (in future
                                 will switch Z to pm, etc.)
    Returns:
        spyObject  - Custom object with attributes appropriate to the type of
                      data and containing experiment parameters in a header.

    History:
        2016-07-14  - HP : Initial commit.
        2016-07-15  - HP : Added support for NVL and NVI files.
        2016-07-29  - HP : Added support for rectangular DOS maps.
        2016-08-02  - HP : Added support for single line DOS maps.
        2016-08-09  - HP : Added bias offset for DAT files.
        2016-09-14  - HP : Added compatibility for incomplete data sets.
        2016-11-01  - HP : Added support for specific ASCII files.
        2017-01-13  - HP : Improved loading of DAT files.
        2017-03-27  - RL : Added support for NSP files.
        2017-06-08  - HP : Use sensible units when loading data.
        2017-06-16  - JG : Improve handling of multi-sweep DAT files.
        2017-08-11  - HP : Added support for non-linear bias sweep.
        2017-08-17  - JG : Added support for general ASCII files.
        2017-08-24  - HP : Better searching for Z attribute in DOS maps.
        2017-10-03  - HP : Improved reading of DAT files
        2018-03-02  - HP : VERSION  1.0 - Unified to a single SPY class.
        2018-10-10  - HP : Python 3 compatibility
        2018-11-07  - HP : Add byte support to SPY files.
        2018-11-13  - HP : Add nice_units to .dat files
        2019-01-09  - BB : Generalize file extension extraction
        2019-02-28  - HP : Loads multisweep .dat files even if missing header.


    '''
    try:
        filename, extension = os.path.splitext(filePath)
        extension = extension.replace(".","")
    except IndexError:
        raise IOError('Please include file extension in path.')
    loadFn = 'load_' + extension

    if extension in ['3ds', 'dat']:
        dataObject = eval(loadFn)(filePath)
        if biasOffset:
            dataObject = _correct_bias_offset(dataObject, extension)
        if niceUnits:
            dataObject = _nice_units(dataObject, extension)
        return dataObject

    elif extension in ['spy', 'sxm', 'nvi', 'nvl', 'nsp', 'asc']:
        return eval(loadFn)(filePath)

  #  elif filePath.endswith('.mat'):
  #      raw_mat = matio.loadmat(filePath)
  #      mappy_dict = {}
  #      for key in raw_mat:
  #          try:
  #              mappy_dict[key] = matio.Mappy()
  #              mappy_dict[key].mat2mappy(raw_mat[key])
  #              print('Created channel: {:}'.format(key))
  #          except:
  #              del mappy_dict[key]
  #              print('Could not convert: {:}'.format(key))
  #      if len(mappy_dict) == 1: return mappy_dict[mappy_dict.keys()[0]]
  #      else: return mappy_dict

    else:
        raise IOError('ERR - File type {:} not supported.'.format(extension))


def save(data, filePath, objects=[]):
    '''
    Save python data to file. Please include the file extension in the path.

    Currently supports:
        .spy    -   STMPY generic data format.

    Inputs:
        data        - Required : Any python data/object/list/...
        filePath    - Required : str. Path where the file will be saved.
        objects     - Optional : lst. Only objects with a __class__ in this
                                 list (and Spy objects) can be saved.

    Returns:
        None

    History:
        2018-03-02  - HP : Initial commit.
        2018-03-08  - HP : Added support for multi-line strings.
        2018-10-10  - HP : Python 3 compatibility
    '''
    try:
        extension = filePath.split('.')[1]
    except IndexError:
        raise IOError('Please include file extension in path.')
    saveFn = 'save_' + extension
    if extension in ['spy']:
        eval(saveFn)(data, filePath, objects)
    else:
        raise IOError('ERR - File type {:} not supported.'.format(extension))



####    ____HIDDEN METHODS____    ####
def _correct_bias_offset(data, fileType):
    try:
        if fileType == 'dat':
            I = data.iv
        elif fileType == '3ds':
            I = [np.mean(data.I[ix]) for ix, __ in enumerate(data.en)]
        else:
            print('ERR: Bias offset for {:} not yet implemented'.format(fileType))
            return data
        for ix, (I_low, I_high) in enumerate(zip(I[:-1], I[1:])):
            if np.sign(I_low) != np.sign(I_high):
                en_low, en_high = data.en[ix], data.en[ix+1]
                biasOffset = en_high - I_high * (en_high-en_low) / (I_high - I_low)
                data.en -= biasOffset
                break
        print('Corrected for a bias offset of {:2.2f} meV'.format(biasOffset*1000))
        return data
    except:
        print('ERR: File not in standard format for processing. Could not correct for Bias offset')
        return data

def _nice_units(data, fileType):
    '''Switch to commonly used units.

    fileType    - .3ds : Use nS for LIY and didv attribute

    History:
        2017-08-10  - HP : Comment: Missing a factor of 2, phase error not
                           justified
        2018-11-13  - HP : Add nice_units to .dat files

    '''
    def use_nS(data):
        def chi(X):
            gFit = X * data.didv / lockInMod
            err = np.absolute(gFit - didv)
            return np.log(np.sum(err**2))
        try:
            lockInMod = float(data.header['Lock-in>Amplitude'])
        except KeyError:
            lockInMod = 1 # Doesn't matter due to minimization.
       # current = np.mean(data.I, axis=(1,2))
        didv = np.gradient(data.iv) / np.gradient(data.en)
        result = minimize(chi, 1)
        data.to_nS = result.x / lockInMod * 1e9
        data.didv *= data.to_nS
        try:
            data.LIY *= data.to_nS
            data.didvStd *= data.to_nS
        except AttributeError:
            pass # Just means that this is a .dat rather than a .3ds
        phi = np.arccos(1.0/result.x)

    def use_nm(data):
        fov = [float(val) for val in data.header['Scan>Scanfield'].split(';')]
        data.x = 1e9 * np.linspace(0, fov[2], data.Z.shape[1])
        data.y = 1e9 * np.linspace(0, fov[3], data.Z.shape[0])
        data.qx = stmpy.tools.fftfreq(len(data.x), data.x[-1])
        data.qy = stmpy.tools.fftfreq(len(data.y), data.y[-1])
        data.Z *= 1e9
        data._pxToNm = data.x[-1]/len(data.x)
        data._pxToInvNm = data.qx[-1]/len(data.qx)
        print('WARNING: I am not 100% sure that the q scale is right...')

    if fileType in ['3ds', 'dat']:
        use_nS(data)
    if fileType in ['3ds', 'sxm']:
        use_nm(data)
    return data

def _make_attr(self, attr, names, data):
    '''
    Trys to give object an attribute from self.data by looking through
    each key in names.  It will add only the fist match, so the order of
    names dictates the preferences.

    Inputs:
        attr    - Required : Name of new attribute
        names   - Required : List of names to search for
        data    - Required : Name of a current attribute in which the new
                             attribute is stored.

    Returns:
        1   - If successfully added the attribute
        0   - If name is not found.

    History:
        2017-08-11  - HP : Initial commit.
        2017-08-24  - HP : Now uses grid z value for Z attribute.
    '''
    dat = getattr(self, data)
    for name in names:
       if name in dat.keys():
            setattr(self, attr, dat[name])
            return 1
    return 0



####    ____SAVE FUNCTIONS____    ####

def save_spy(data, filePath, objects=[]):
    '''Save python data to file'''

    def stew(fileObj, val):
        'Quickly write binary strings with utf-8 encoding.'
        return fileObj.write(bytearray(val.encode('utf-8')))

    def write_npy(name, npy):
        if npy.dtype.name == 'object':
            stew(fileObj, 'OAR=' + name + '\n' + str(npy.shape) + '\n')
            for obj in npy:
                write_obj('unnamed', obj)
        else:
            stew(fileObj, 'NPY=' + name + '\n')
            np.save(fileObj, npy)

    def write_obj(name, obj):
        stew(fileObj, 'OBJ=' + name + '\n')
        for name, item in obj.__dict__.items():
            write_item(name, item)
        stew(fileObj, ':OBJ_END:\n')

    def write_dic(name, dic):
        stew(fileObj, 'DIC=' + name + '\n')
        for name, item in dic.items():
            write_item(name, item)
        stew(fileObj, ':DIC_END:\n')

    def write_lst(name, lst):
        stew(fileObj, 'LST=' + name + '\n')
        for ix, item in enumerate(lst):
            write_item(str(ix), item)
        stew(fileObj, ':LST_END:\n')

    def write_str(name, val):
        stew(fileObj, 'STR=' + name + '\n' + val + '\n:STR_END:\n')
        #stew(bytearray(val.encode('utf-8')) + '\n')
        #stew(':STR_END:\n')

    def write_byt(name, byt):
        stew(fileObj, 'BYT=' + name + '\n')
        fileObj.write(byt)

    def write_num(name, val):
        stew(fileObj, 'NUM=' + name + '\n')
        if isinstance(val, int):
            fmt = '>i'
        elif isinstance(val, float):
            fmt = '>d'
        elif PY2:
            if isinstance(val, long):
                fmt = '>l'
        fileObj.write(bytearray(fmt.encode('utf-8')) + pack(fmt, val))

    def write_cpx(name, val):
        stew(fileObj, 'CPX=' + name + '\n')
        fileObj.write(pack('>f', val.real) + pack('>f', val.imag))

    def write_bol(name, val):
        stew(fileObj, 'BOL=' + name + '\n')
        fileObj.write('NOTWORKING')

    def write_item(name, item):
        if type(item).__module__ == np.__name__:
            write_npy(name, item)
        elif isinstance(item, dict):
            write_dic(name, item)
        elif isinstance(item, list):
            write_lst(name, item)
        elif isinstance(item, tuple):
            print('Tuples present...')
        elif hasattr(item, 'read'):
            pass
        elif isinstance(item, str):
            write_str(name, item)
        elif isinstance(item, bytes):
            write_byt(name, item)
        elif type(item) in [int, float]:
            write_num(name, item)
        elif isinstance(item, complex):
            write_cpx(name, item)
        elif callable(item):
            print('WARING: Callable item not saved: {:}.'.format(name))
        elif any([isinstance(item, obj) for obj in objects]):
            write_obj(name, item)
        elif PY2:
            # Legacy types deprecated in python 3.x
            if isinstance(item, unicode):
                write_str(name, item)
            elif isinstance(item, long):
                write_num(name, item)
        else:
            raise(TypeError('Item {:} {:} not supported.'.format(name, type(item))))

    fileObj = open(filePath, 'wb')
    stew(fileObj, 'SPY: Stmpy I/O, Version=' + str(__version__) + '\n')
    objects.append(Spy)
    write_item('MAIN', data)
    fileObj.close()



####    ____LOAD FUNCTIONS____    ####

def load_spy(filePath):
    ''' Load .spy files into python'''
    def read_npy(fileObj):
        npy = np.load(fileObj)
        if npy.shape == ():
            npy = npy.flatten()[0]
        return npy

    def read_oar(fileObj):
        line = fileObj.readline().strip().decode('utf-8')
        shape = eval(line)
        oar = np.empty(shape=shape, dtype=object).flatten()
        for ix, __ in enumerate(oar):
            line = fileObj.readline().strip().decode('utf-8')
            oar[ix] = read_obj(fileObj)
        return oar.reshape(shape)

    def read_obj(fileObj):
        obj = Spy()
        while True:
            line = fileObj.readline().strip().decode('utf-8')
            if line == ':OBJ_END:':
                'finished'
                break
            key, val = line.split('=')
            setattr(obj, val, read_item(fileObj, key))
        return obj

    def read_dic(fileObj):
        dic = {}
        while True:
            line = fileObj.readline().strip().decode('utf-8')
            if line == ':DIC_END:':
                break
            key, val = line.split('=')
            dic[val] = read_item(fileObj, key)
        return dic

    def read_lst(fileObj):
        lst = []
        while True:
            line = fileObj.readline().strip().decode('utf-8')
            if line == ':LST_END:':
                break
            key, val = line.split('=')
            lst.append(read_item(fileObj, key))
        return lst

    def read_str(fileObj):
        st = ''
        while True:
            line = fileObj.readline()
            if line.strip().decode('utf-8') == ':STR_END:':
                break
            st += line.decode('utf-8')
        return st

    def read_byt(fileObj):
        return fileObj.readline()

    #def read_str(fileObj):
    #    return fileObj.readline().strip().decode('utf-8')

    def read_num(fileObj):
        fmt = fileObj.read(2)
        num = unpack(fmt, fileObj.read(calcsize(fmt)))[0]
        return num

    def read_cpx(fileObj):
        real = unpack('>f', fileObj.read(4))[0]
        imag = unpack('>f', fileObj.read(4))[0]
        return complex(real, imag)

    def read_item(fileObj, key):
        if   key == 'NPY':
            item = read_npy(fileObj)
        elif key == 'OAR':
            item = read_oar(fileObj)
        elif key == 'OBJ':
            item = read_obj(fileObj)
        elif key == 'DIC':
            item = read_dic(fileObj)
        elif key == 'LST':
            item = read_lst(fileObj)
        elif key == 'STR':
            item = read_str(fileObj)
        elif key == 'BYT':
            item = read_byt(fileObj)
        elif key == 'NUM':
            item = read_num(fileObj)
        elif key == 'CPX':
            item = read_cpx(fileObj)
        else:
            raise(TypeError(
                'File contains unsupported format: {:}'.format(key)))
        return item

    fileObj = open(filePath, 'rb')
    fileObj.seek(0,2)
    fileSize = fileObj.tell()
    fileObj.seek(0)
    name, version = fileObj.readline().strip().decode('utf-8').split('=')
    if float(version) < 1.0:
        raise(TypeError('Version {:} files not supported'.format(version)))
    while fileObj.tell() < fileSize:
        line = fileObj.readline().strip().decode('utf-8')
        key, val = line.split('=')
        item = read_item(fileObj, key)
    fileObj.close()
    return item

def load_stmview(name, path=''):
    '''
    Loads STM_View files into python.
    Inputs:
            name    - Required : String containing the data set name. The data
                                 contains numerous .mat files, which much have
                                 the correct appendices following name (-G, -I,
                                 -T) e.g. name = '90227A13'
            path    - Optional: String containing path to the directory where
                                the data files are located.

    Returns:
            data    - stmpy.io.Spy() object with standard attributes: LIY, en,
                      didv, I, and Z, but no header.

    History:
            2020-02-12  - HP : Initial commit
    '''
    def matigo(name, path='', extension=''):
        raw = stmpy.matio.loadmat(path + name + extension)
        end = extension.split('.')[0][1:]
        mat = raw['obj_' + name + '_' + end]
        return mat
    self = stmpy.io.Spy()
    matG = matigo(name, path, '-G.mat')
    matI = matigo(name, path, '-I.mat')
    matZ = matigo(name, path, '-T.mat')
    self.LIY = np.moveaxis(matG['map'], -1, 0)
    self.en = matG['e'][0]
    self.didv = matG['ave']
    self.I = np.moveaxis(matI['map'], -1, 0)
    self.Z = matZ['map']
    return self

def load_3ds(filePath):
    '''Load Nanonis 3ds into python.'''
    try:
        fileObj = open(filePath, 'rb')
    except:
        raise NameError('File not found.')
    self = Spy()
    self.header={}
    while True:
        line = fileObj.readline().strip().decode('utf-8')
        if line == ':HEADER_END:':
            break
        splitLine = line.split('=')
        self.header[splitLine[0]] = splitLine[1]
    self._info = {'params'	: int(self.header['# Parameters (4 byte)']),
                'paramName'	: self.header['Fixed parameters'][1:-1].split(';') +
                              self.header['Experiment parameters'][1:-1].split(';'),
                'channels'	: self.header['Channels'][1:-1].split(';'),
                'points'	: int(self.header['Points']),
                'sizex' 	: int(self.header['Grid dim'][1:-1].split(' x ')[0]),
                'sizey'	: int(self.header['Grid dim'][1:-1].split(' x ')[1]),
                'dataStart'	: fileObj.tell()
                 }
    self.grid = {}; self.scan = {}
    for channel in self._info['channels']:
        self.grid[channel] = np.zeros(
                [self._info['points'], self._info['sizey'], self._info['sizex']])
    for channel in self._info['paramName']:
        self.scan[channel] = np.zeros([self._info['sizey'], self._info['sizex']])

    try:
        for iy in range(self._info['sizey']):
            for ix in range(self._info['sizex']):
                for channel in self._info['paramName']:
                    value = unpack('>f',fileObj.read(4))[0]
                    self.scan[channel][iy,ix] = value

                for channel in self._info['channels']:
                    for ie in range(self._info['points']):
                        value = unpack('>f',fileObj.read(4))[0]
                        self.grid[channel][ie,iy,ix] = value
    except:
        print('WARNING: Data set is not complete.')

    dataRead = fileObj.tell()
    fileObj.read()
    allData = fileObj.tell()
    if dataRead == allData:
        print('File import successful.')
    else:
        print('ERR: Did not reach end of file.')
    fileObj.close()

    LIYNames =  ['LIY 1 omega (A)', 'LIY 1 omega [AVG] (A)', 'LI Demod 1 Y (A)', 'LI Demod 2 Y (A)','LI Demod 3 Y (A)']
    if _make_attr(self, 'LIY', LIYNames, 'grid'):
        self.didv = np.mean(self.LIY, axis=(1,2))
        self.didvStd = np.std(self.LIY, axis=(1,2))
    else:
        print('ERR: LIY AVG channel not found, resort to manual ' +
              'definitions.  Found channels:\n {:}'.format(self.grid.keys()))

    _make_attr(self, 'I',  ['Current (A)', 'Current [AVG] (A)'], 'grid')
    if _make_attr(self, 'Z',  ['Z (m)', 'Z [AVG] (m)'], 'grid'):
        self.Z = self.Z[0]
    else:
        _make_attr(self, 'Z', ['Scan:Z (m)'], 'scan')
        print('WARNING: Using scan channel for Z attribute.')
    self.iv = np.mean(self.I, axis=(1,2))
    try:
        self.en = np.mean(self.grid['Bias [AVG] (V)'], axis=(1,2))
    except KeyError:
        print('WARNING: Assuming energy layers are evenly spaced.')
        self.en = np.linspace(self.scan['Sweep Start'].flatten()[0],
                              self.scan['Sweep End'].flatten()[0],
                              self._info['points'])

    return self


def load_sxm(filePath):
    ''' Load Nanonis SXM files into python. '''
    try:
        fileObj = open(filePath, 'rb')
    except:
        raise NameError('File not found.')
    self = Spy()
    self.header={}
    s1 = fileObj.readline().decode('utf-8')
    if not re.match(':NANONIS_VERSION:', s1):
        raise NameError('The file %s does not have the Nanonis SXM'.format(filePath))
    self.header['version'] = int(fileObj.readline())
    while True:
        line = fileObj.readline().strip().decode('utf-8')
        if re.match('^:.*:$', line):
            tagname = line[1:-1]
        else:
            if 'Z-CONTROLLER' == tagname:
                keys = line.split('\t')
                values = fileObj.readline().strip().decode('utf-8').split('\t')
                self.header['z-controller'] = dict(zip(keys, values))
            elif tagname in ('BIAS', 'REC_TEMP', 'ACQ_TIME', 'SCAN_ANGLE'):
                self.header[tagname.lower()] = float(line)
            elif tagname in ('SCAN_PIXELS', 'SCAN_TIME', 'SCAN_RANGE', 'SCAN_OFFSET'):
                self.header[tagname.lower()] = [ float(i) for i in re.split('\s+', line) ]
            elif 'DATA_INFO' == tagname:
                if 1 == self.header['version']:
                    keys = re.split('\s\s+',line)
                else:
                    keys = line.split('\t')
                self.header['data_info'] = []
                while True:
                    line = fileObj.readline().strip().decode('utf-8')
                    if not line:
                        break
                    values = line.strip().split('\t')
                    self.header['data_info'].append(dict(zip(keys, values)))
            elif tagname in ('SCANIT_TYPE','REC_DATE', 'REC_TIME', 'SCAN_FILE', 'SCAN_DIR'):
                self.header[tagname.lower()] = line
            elif 'SCANIT_END' == tagname:
                break
            else:
                if tagname.lower() not in self.header:
                    self.header[tagname.lower()] = line
                else:
                    self.header[tagname.lower()] += '\n' + line
    if 1 == self.header['version']:
        self.header['scan_pixels'].reverse()
    fileObj.readline()
    fileObj.read(2) # Need to read the byte \x1A\x04, before reading data
    size = int(self.header['scan_pixels'][0] * self.header['scan_pixels'][1] * 4)
    shape = [int(val) for val in self.header['scan_pixels']]
    self.channels = {}
    for channel in self.header['data_info']:
        if channel['Direction'] == 'both':
            self.channels[channel['Name'] + '_Fwd'] = np.ndarray(
                    shape=shape[::-1], dtype='>f', buffer=fileObj.read(size))
            self.channels[channel['Name'] + '_Bkd'] = np.ndarray(
                    shape=shape[::-1], dtype='>f', buffer=fileObj.read(size))
        else:
            self.channels[channel['Name'] + channel['Direction']] = np.ndarray(
                    shape=shape, dtype='>f', buffer=fileObj.read(size))
    try:
        self.Z = self.channels['Z_Fwd']
        self.I = self.channels['Current_Fwd']
        self.LIY = self.channels['LIY_1_omega_Fwd']
    except KeyError: print('WARNING:  Could not create standard attributes, look in channels instead.')
    fileObj.close()
    return self


def load_dat(filePath):
    ''' Load Nanonis SXM files into python. '''
    try:
        fileObj = open(filePath, 'rb')
    except:
        raise NameError('File not found.')
    self = Spy()
    self.header={}
    self.channels = {}
    while True:
        line = fileObj.readline().decode('utf-8')
        splitLine = line.split('\t')
        if line[0:6] == '[DATA]':
            break
        elif line.rstrip() != '':
            self.header[splitLine[0]] = splitLine[1]
    channels = fileObj.readline().decode('utf-8').rstrip().split('\t')
    allData = []
    for line in fileObj:
        line = line.decode('utf-8').rstrip().split('\t')
        allData.append(np.array(line, dtype=float))
    allData = np.array(allData)
    for ix, channel in enumerate(channels):
        self.channels[channel] = allData[:,ix]
    dataRead = fileObj.tell()
    fileObj.read()
    finalRead = fileObj.tell()
    if dataRead == finalRead:
        print('File import successful.')
    else:
        print('ERR: Did not reach end of file.')
    fileObj.close()
    _make_attr(self, 'didv',
            ['LIY 1 omega (A)', 'LIY 1 omega [AVG] (A)'], 'channels')
    _make_attr(self, 'iv', ['Current (A)', 'Current [AVG] (A)'],
    'channels')
    _make_attr(self, 'en', ['Bias (V)', 'Bias calc (V)'], 'channels')
    if 'LIY 1 omega [00001] (A)' in self.channels.keys():
        try:
            sweeps = int(self.header['Bias Spectroscopy>Number of sweeps'])
        except KeyError:
            sweeps = -1
            flag = 0
            for key in self.channels.keys():
                if key.startswith('LIY 1 omega') and 'bwd' not in key:
                    sweeps += 1
                if 'bwd' in key:
                    flag = 1
            if flag == 1:
                print('WARNING: Ignoring backward sweeps.')
        self.LIY = np.zeros([len(self.en), sweeps])
        self.I = np.zeros([len(self.en), sweeps])
        for ix in range(1, sweeps+1):
            s = str(ix).zfill(5)
            try:
                self.LIY[:,ix-1] = self.channels['LIY 1 omega [' + s + '] (A)']
                self.I[:,ix-1] = self.channels['Current [' + s + '] (A)']
            except KeyError:
                print('WARNING: Number of sweeps less than expected.\n' +
                     'Found {:d}, expected {:d}.\t'.format(ix-1, sweeps) +
                     'Consequently, data.didvStd is not correct. ')
                break
        self.didvStd = np.std(self.LIY, axis=1)
    return self


def load_nsp(filePath):
    '''UNTESTED - Load Nanonis Long Term Specturm into python.'''
    try:
        fileObj = open(filePath, 'rb')
    except:
        raise NameError('File not found.')
    self = Spy()
    self.header={}
    while True:
        line = fileObj.readline().strip().decode('utf-8')
        if line == ':HEADER_END:':
            break
        elif re.match('^:.*:$', line):
            tagname = line[1:-1]
        else:
            try:
                self.header[tagname] = int(line.split('\t')[0])
            except:
                self.header[tagname] = line.split('\t')[0]

    self.freq = np.linspace(0,
            np.round(float(self.header['DATASIZECOLS'])*float(self.header['DELTA_f'])),
            float(self.header['DATASIZECOLS']))

    self.start = datetime.strptime(self.header['START_DATE'] +
            self.header['START_TIME'],'%d.%m.%Y%H:%M:%S')
    self.end = datetime.strptime(self.header['END_DATE'] +
            self.header['END_TIME'],'%d.%m.%Y%H:%M:%S')
    self.time = np.linspace(0, (self.end - self.start).total_seconds(),
            int(self.header['DATASIZEROWS']))
    self.data = np.zeros([int(self.header['DATASIZEROWS']), int(self.header['DATASIZECOLS'])])
    fileObj.read(2) #first two bytes are not data
    try:
        for ix in range(int(self.header['DATASIZEROWS'])):
            for iy in range(int(self.header['DATASIZECOLS'])):
                value = unpack('>f',fileObj.read(4))[0]
                self.data[ix,iy] = value
    except:
        print('ERR: Data set is not complete')
    fileObj.close()
    if self.header['SIGNAL'] == 'Current (A)':
        self.fftI = self.data.T
    elif self.header['SIGNAL'] == 'InternalGeophone (V)':
        self.fftV = self.data.T
    else:
        self.fftSignal = self.data.T
    return self


def load_nvi(filePath):
    '''UNTESTED - Load NISTview image data into python. '''
    nviData = sio.readsav(filePath)
    self = Spy()
    self._raw = nviData['imagetosave']
    self.map = self._raw.currentdata[0]
    self.header = {name:self._raw.header[0][name][0] for name in self._raw.header[0].dtype.names}
    self.info = {'FILENAME'    : self._raw.filename[0],
                 'FILSIZE'     : int(self._raw.header[0].filesize[0]),
                 'CHANNELS'    : self._raw.header[0].scan_channels[0],
                 'XSIZE'       : self._raw.xsize[0],
                 'YSIZE'       : self._raw.ysize[0],
                 'TEMPERATURE' : self._raw.header[0].temperature[0],
                 'LOCKIN_AMPLITUDE' : self._raw.header[0].lockin_amplitude[0],
                 'LOCKIN_FREQUENCY' : self._raw.header[0].lockin_frequency[0],
                 'DATE'        : self._raw.header[0].date[0],
                 'TIME'        : self._raw.header[0].time[0],
                 'BIAS_SETPOINT'    : self._raw.header[0].bias_setpoint[0],
                 'BIAS_OFFSET' : self._raw.header[0].bias_offset[0],
                 'BFIELD'      : self._raw.header[0].bfield[0],
                 'ZUNITS'      : self._raw.zunits[0],
					}
    return self


def load_nvl(filePath):
    '''UNTESTED - Load NISTview layer data into python. '''
    nvlData = sio.readsav(filePath)
    self = Spy()
    self._raw = nvlData['savestructure']
    self.en = self._raw.energies[0]
    self.map = self._raw.fwddata[0]
    self.ave = [np.mean(layer) for layer in self.map]
    self.header = {name:self._raw.header[0][name][0] for name in self._raw.header[0].dtype.names}
    for name in self._raw.dtype.names:
        if name not in self.header.keys():
            self.header[name] = self._raw[name][0]
    return self


def load_asc(filePath):
    '''UNTESTED - Load ASCII files into python.'''
    try:
        fileObj = open(filePath, 'rb')
    except:
        raise NameError('File not found.')
    self = Spy()
    header= {}
    channels = {}
    while True:
        line = fileObj.readline().rstrip()
        if line is '':
            break
        splitLine = line.split(':')
        header[splitLine[0]] = splitLine[1]
    channelNames = fileObj.readline().rstrip().split('      ')
    for chn in channelNames:
        channels[chn] = []
    for data in fileObj.readlines():
        dsplit = data.rstrip().split('   ')
        dfloat = [float(val) for val in dsplit]
        for chn, val in zip(channelNames, dfloat):
            channels[chn] += [val]
    for chn in channelNames:
        channels[chn] = np.array(channels[chn])
    if len(channelNames) is 2:
        self.x = channels[channelNames[0]]
        self.y = channels[channelNames[1]]
    self.header = header
    self.channels = channels
    fileObj.close()
    return self


####    ____CLASS DEFINITIONS____   ####

class Spy(object):
    def __init__(self):
        pass
