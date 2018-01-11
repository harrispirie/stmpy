from struct import unpack
import numpy as np
import scipy.io as sio
import os
import re
import stmpy
from stmpy import matio
from datetime import datetime, timedelta
from scipy.optimize import minimize

'''
I think I should use more doc strings.
'''


def load(filePath, biasOffset=True, niceUnits=False):
    '''
    Loads data into python. Please include the file extension in the path.

    Currently supports formats: 3ds, sxm, dat, nvi, nvl, mat.

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
        dataObject  - Custom object with attributes appropriate to the type of
                      data and containing experiment parameters in a header.
                      
    Usage:
        data = load('file.3ds', biasOffset=True, niceUnits=False)
    '''
    if filePath.endswith('.3ds'):
        dataObject = Nanonis3ds(filePath)
        if biasOffset:
            dataObject = _correct_bias_offset(dataObject, '.3ds')
        if niceUnits:
            dataObject = _nice_units(dataObject)
        return dataObject

    elif filePath.endswith('.sxm'):
        return NanonisSXM(filePath)

    elif filePath.endswith('.dat'):
        if biasOffset:
            return _correct_bias_offset(NanonisDat(filePath), '.dat')
        else:
            return NanonisDat(filePath)

    elif filePath[-3:] == 'NVI' or filePath[-3:] == 'nvi':
        return NISTnvi(sio.readsav(filePath))

    elif filePath[-3:] == 'NVL' or filePath[-3:] == 'nvl':
        return NISTnvl(sio.readsav(filePath))

    elif filePath.endswith('.nsp'):
        return LongTermSpectrum(filePath)

    elif filePath.endswith('.mat'):
        raw_mat = matio.loadmat(filePath)
        mappy_dict = {}
        for key in raw_mat:
            try:
                mappy_dict[key] = matio.Mappy()
                mappy_dict[key].mat2mappy(raw_mat[key])
                print('Created channel: {:}'.format(key))
            except:
                del mappy_dict[key]
                print('Could not convert: {:}'.format(key))
        if len(mappy_dict) == 1: return mappy_dict[mappy_dict.keys()[0]]
        else: return mappy_dict
    
    elif filePath.endswith('.asc'):
        return AsciiFile(filePath)

    else: raise IOError('ERR - Wrong file type.')


def save(filePath, pyObject):
    '''
Save objects from a python workspace to disk.
Currently implemented for the following python data types: nvl, mat.
Currently saves to the following file types: mat.
Please include the file extension in the path, e.g. 'file.mat'

Usage: save(filePath, data)
    '''
    if filePath.endswith('.mat'):
        if pyObject.__class__ == matio.Mappy:
            pyObject.savemat(filePath)
        elif pyObject.__class__ == NISTnvl:
            mappyObject = matio.Mappy()
            mappyObject.nvl2mappy(pyObject)
            mappyObject.savemat(filePath)
    else: raise IOError('ERR - File format not supported.')

def qkrdasciifile(filename, delimiter='\t', returnnotes=False):
    '''
    Read formatted data in a general ascii file (no specific extension) with header and endnote optionally returned.
    Data is read when a line starts with a digit (leading spaces and tabs skipped), other than which all above saved as header, below saved as endnote, and both printed out.
    Usage: data, header, endnote = qkrdasciifile('filename.txt', delimiter='\t', returnnotes=True)
    '''
    with open(filename, 'r') as f:
        lines = f.readlines()
    header = []
    ih = 0
    while not lines[ih].lstrip('- \t')[0].isdigit():
        ih = ih + 1
    header = lines[:ih]
    idt = 0
    while lines[ih+idt].lstrip('- \t')[0].isdigit():
        idt = idt + 1
        if ih+idt == len(lines):
            break
    endnote = lines[(ih+idt):]
    rows = idt
    cols = len(lines[ih].lstrip('\t').split(delimiter))
    data = np.zeros((cols, rows))
    for ir in range(rows):
        tmp = lines[ir+ih].lstrip('\t').split(delimiter)
        for ic in range(cols):
            data[ic, ir] = float(tmp[ic])
    print('%d row(s), %d column(s) data loaded'%(rows, cols))
    print('Header:')
    for headerline in header:
        print(headerline)
    print('Endnote:')
    for endline in endnote:
        print(endline)
    if returnnotes:
        return data, header, endnote
    else:
        return data

####    ____HIDDEN METHODS____    ####

def _correct_bias_offset(data, fileType):
    try:
        if fileType == '.dat':
            I = data.I
        elif fileType == '.3ds':
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

def _nice_units(data):
    '''Switch to commonly used units.

    fileType    - .3ds : Use nS for LIY and didv attribute

    History:
        2017-08-10  - HP : Comment: Missing a factor of 2, phase error not
                           justified 
    '''
    def use_nS(data):
        def chi(X):
            gFit = X * data.didv / lockInMod
            err = np.absolute(gFit - didv)
            return np.log(np.sum(err**2))
        lockInMod = float(data.header['Lock-in>Amplitude'])
        current = np.mean(data.I, axis=(1,2))
        didv = np.gradient(current) / np.gradient(data.en)
        result = minimize(chi, 1)
        data.to_nS = result.x / lockInMod * 1e9 
        data.didv *= data.to_nS
        data.LIY *= data.to_nS
        data.didvStd *= data.to_nS
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
    
    use_nS(data)
    use_nm(data)
    return data


####    ____CLASS DEFINITIONS____   ####

class Nanonis3ds(object):
    '''Data structure for Nanonis DOS maps.

    Attributes:
        self.Z      - 2D numpy array containing topography channel. Looks for
                      the 'setup Z' recored simultaneously with the DOS map, if
                      not found it will resort to the 'scan Z' measured when
                      moving between lines. 
        self.I      - 3D numpy array for current at each poont.
        self.LIY    - 3D numpy array containing lock-in Y channel.
        self.didv   - 1D numpy array for average LIY
        self.didvStd - 1D numpy array for standard deviation in dIdV.
        self.en     - 1D numpy array for energies used in bias sweep. 
        self.header - Dictionary of all recorded experimental parameters.
        self.grid   - Dictionary of all grid spectroscopy data (individual sweeps, etc.)
        self.scan   - Dictionary with all scan data.

    Methods:
        None

    History:
        2017-08-06  - HP : Added support for recangular maps by changing the
                           order in which data points are read.
        2017-08-10  - HP : Added support for non-linear spaced energies.
    '''
    def __init__(self, filePath):
        if self._load3ds(filePath):
            LIYNames =  ['LIY 1 omega (A)', 'LIY 1 omega [AVG] (A)']
            if self._make_attr('LIY', LIYNames, 'grid'):
                self.didv = np.mean(self.LIY, axis=(1,2))
                self.didvStd = np.std(self.LIY, axis=(1,2))
            else:
                print('ERR: LIY AVG channel not found, resort to manual ' + 
                      'definitions.  Found channels:\n {:}'.format(self.data.keys()))
            
            self._make_attr('I',  ['Current (A)', 'Current [AVG] (A)'], 'grid')
            if self._make_attr('Z',  ['Z (m)', 'Z [AVG] (m)'], 'grid'):
                self.Z = self.Z[0]
            else:
                self._make_attr('Z', ['Scan:Z (m)'], 'scan')
                print('WARNING: Using scan channel for Z attribute.')
            try:     
                self.en = np.mean(self.grid['Bias [AVG] (V)'], axis=(1,2))
            except KeyError:
                print('WARNING: Assuming energy layers are evenly spaced.')
                self.en = np.linspace(self.scan['Sweep Start'].flatten()[0],
                                      self.scan['Sweep End'].flatten()[0],
                                      self._info['points'])
        else: 
            raise NameError('File not found.')

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

    def _load3ds(self, filePath):
        try: 
            fileObj = open(filePath, 'rb')
        except: 
            return 0
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
        return 1

class LongTermSpectrum(object):
    '''
header: a dict containging all parameters
time: time length of the spectrum, in unit of (s)
freq: freq range of the spectrum, in unit of (Hz)
fftI, fftV, or fftSignal:
    current signal >> fftI
    voltage signal >> fftV
    other signal   >> fftSignal 
Example Usage:
import stmpy \nimport matplotlib.pyplot as plt \nimport matplotlib.dates as md \nimport numpy as np \nfrom datetime import datetime \n 
data = stmpy.load('***.nsp')
x=np.array(data.start,dtype='datetime64[s]') \nstepsize = int(np.floor(((data.end - data.start).total_seconds())/data.header['DATASIZEROWS']))
dates = x + np.arange(0,stepsize*data.header['DATASIZEROWS'],stepsize) \nnew_dates=[np.datetime64(ts).astype(datetime) for ts in dates]
datenums=md.date2num(new_dates) \nplt.subplots_adjust(bottom=0.2) \nplt.xticks( rotation=45 ) \nplt.ax=plt.gca() \nxfmt = md.DateFormatter('%Y-%m-%d %H:%M:%S')
plt.ax.xaxis.set_major_formatter(xfmt) \nplt.pcolormesh(datenums, data.freq, data.fftI)#depending on which kind of data used
plt.clim(0,1e-12)#can be different \nplt.ylabel('Frequency (Hz)') \nplt.savefig('pic name.png',dpi = 600,bbox_inches = 'tight') \nplt.show()
    '''
    def __init__(self, filePath):
        self._loadnsp(filePath)
        if self.header['SIGNAL'] == 'Current (A)':
            self.fftI = self.data.T
        elif self.header['SIGNAL'] == 'InternalGeophone (V)':
            self.fftV = self.data.T
        else:
            self.fftSignal = self.data.T

    def _loadnsp(self, filePath):
        try: fileObj = open(filePath, 'rb')
        except: return 0
        self.header = {}
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

        self.freq = np.linspace(0, np.round(float(self.header['DATASIZECOLS'])*float(self.header['DELTA_f'])),float(self.header['DATASIZECOLS']))
        
        self.start = datetime.strptime(self.header['START_DATE']+self.header['START_TIME'],'%d.%m.%Y%H:%M:%S')
        self.end = datetime.strptime(self.header['END_DATE']+self.header['END_TIME'],'%d.%m.%Y%H:%M:%S')
        self.time = np.linspace(0, (self.end - self.start).total_seconds(), int(self.header['DATASIZEROWS']))

        self.data = np.zeros([int(self.header['DATASIZEROWS']),int(self.header['DATASIZECOLS'])])
        fileObj.read(2) #first two bytes are not data
        try:
            for ix in range(int(self.header['DATASIZEROWS'])):
                for iy in range(int(self.header['DATASIZECOLS'])):
                    value = unpack('>f',fileObj.read(4))[0]
                    self.data[ix,iy] = value
        except:
            print('Error: Data set is not complete')


class NanonisSXM(object):
    def __init__(self, filename):
        self.header = {}
        self.header['filename'] = filename
        self._open()
    
    def _open(self):
        self._file = open(os.path.normpath(self.header['filename']), 'rb')
        s1 = self._file.readline().decode('utf-8')
        if not re.match(':NANONIS_VERSION:', s1):
            print('The file %s does not have the Nanonis SXM'.format(self.header['filename']))
            return
        self.header['version'] = int(self._file.readline())
        while True:
            line = self._file.readline().strip().decode('utf-8')
            if re.match('^:.*:$', line):
                tagname = line[1:-1]
            else:
                if 'Z-CONTROLLER' == tagname:
                    keys = line.split('\t')
                    values = self._file.readline().strip().decode('utf-8').split('\t')
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
                        line = self._file.readline().strip().decode('utf-8')
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
        self._file.readline()
        self._file.read(2) # Need to read the byte \x1A\x04, before reading data
        size = int( self.header['scan_pixels'][0] * self.header['scan_pixels'][1] * 4)
        shape = [int(val) for val in self.header['scan_pixels']]
        self.channels = {}
        for channel in self.header['data_info']:
            if channel['Direction'] == 'both':
                self.channels[channel['Name'] + '_Fwd'] = np.ndarray(
                        shape=shape[::-1], dtype='>f', buffer=self._file.read(size))
                self.channels[channel['Name'] + '_Bkd'] = np.ndarray(
                        shape=shape[::-1], dtype='>f', buffer=self._file.read(size))
            else:
                self.channels[channel['Name'] + channel['Direction']] = np.ndarray(shape=shape, dtype='>f', buffer=self._file.read(size))
        try:
            self.Z = self.channels['Z_Fwd']
            self.I = self.channels['Current_Fwd']
            self.LIY = self.channels['LIY_1_omega_Fwd']
        except KeyError: print('WARNING:  Could not create standard attributes, look in channels instead.')
        self._file.close()


class NanonisDat(object):
    def __init__(self,filePath):
        if self._loadDat(filePath):
            self._make_attr('didv', 
                    ['LIY 1 omega (A)', 'LIY 1 omega [AVG] (A)'], 'channels')
            self._make_attr('I', ['Current (A)', 'Current [AVG] (A)'],
            'channels')
            self._make_attr('en', ['Bias (V)', 'Bias calc (V)'], 'channels')
            if 'LIY 1 omega [00001] (A)' in self.channels.keys():
                sweeps = int(self.header['Bias Spectroscopy>Number of sweeps'])
                self.LIY = np.zeros([len(self.en), sweeps])
                for ix in range(1, sweeps+1):
                    s = str(ix).zfill(5)
                    self.LIY[:,ix-1] = self.channels['LIY 1 omega [' + s + '] (A)']
                self.didvStd = np.std(self.LIY, axis=1)

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
            2017-08-29  - HP : Initial commit (copied from Nanonis3ds)
        '''
        dat = getattr(self, data)
        for name in names:
            if name in dat.keys():
                setattr(self, attr, dat[name])
                return 1
        return 0

    def _loadDat(self, filePath):
        self.channels = {}
        self.header = {}
        fileObj = open(filePath,'r')
        while True:
            line = fileObj.readline()
            splitLine = line.split('\t')
            if line[0:6] == '[DATA]': 
                break
            elif line.rstrip() != '': 
                self.header[splitLine[0]] = splitLine[1]
        channels = fileObj.readline().rstrip().split('\t')
        allData = []
        for line in fileObj:
            line = line.rstrip().split('\t')
            allData.append(np.array(line, dtype = float))
        allData = np.array(allData)
        for ix, channel in enumerate(channels):
            self.channels[channel] = allData[:,ix]
        dataRead = fileObj.tell()
        fileObj.read()
        finalRead = fileObj.tell()
        if dataRead == finalRead: 
            print('File import successful.')
            fileObj.close()
            return 1
        else: 
            print('ERR: Did not reach end of file.')
            fileObj.close()
            return 0

        
        

class NISTnvi(object):
    def __init__(self,nviData):
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
        
class NISTnvl(object):
    def __init__(self,nvlData):
        self._raw = nvlData['savestructure']
        self.en = self._raw.energies[0]
        self.map = self._raw.fwddata[0]
        self.ave = [np.mean(layer) for layer in self.map]
        self.header = {name:self._raw.header[0][name][0] for name in self._raw.header[0].dtype.names}
        for name in self._raw.dtype.names:
            if name not in self.header.keys():
                self.header[name] = self._raw[name][0]
        self.info = {}
        try:
            self.info['FILENAME']   = self._raw.filename[0]
        except:
            1
        try:
            self.info['FILSIZE']    = int(self._raw.header[0].filesize[0])
        except:
            1
        try:
            self.info['CHANNELS']   = self._raw.header[0].scan_channels[0]
        except:
            1
        try:
            self.info['XSIZE']      = self._raw.xsize[0]
        except:
            1
        try:
            self.info['YSIZE']      = self._raw.ysize[0]
        except:
            1
        try:
            self.info['TEMPERATURE']= self._raw.header[0].temperature[0]
        except:
            1
        try:
            self.info['LOCKIN_AMPLITUDE']= self._raw.header[0].lockin_amplitude[0]
        except:
            1
        try:
            self.info['LOCKIN_FREQUENCY']= self._raw.header[0].lockin_frequency[0]
        except:
            1
        try:
            self.info['DATE']       = self._raw.header[0].date[0]
        except:
            1
        try:
            self.info['TIME']       = self._raw.header[0].time[0]
        except:
            1
        try:
            self.info['BIAS_SETPOINT'] = self._raw.header[0].bias_setpoint[0]
        except:
            1
        try:
            self.info['BIAS_OFFSET']= self._raw.header[0].bias_offset[0]
        except:
            1
        try:
            self.info['BFIELD']     = self._raw.header[0].bfield[0]
        except:
            1
        try:
            self.info['WINDOWTITLE'] = self._raw.windowtitle[0]
        except:
            1
        try:
            self.info['XYUNITS']    = self._raw.xyunits[0]
        except:
            1
        try:
            self.info['EUNITS']     = self._raw.eunits[0]
        except:
            1

class AsciiFile(object):
    def __init__(self, filePath):
        self.load(filePath)
    
    def load(self, filePath):
        fid = open(filePath, 'r')
        header= {}
        channels = {}
        while True:
            line = fid.readline().rstrip()
            if line is '':
                break
            splitLine = line.split(':')
            header[splitLine[0]] = splitLine[1]
        channelNames = fid.readline().rstrip().split('      ')
        for chn in channelNames:
            channels[chn] = []
        for data in fid.readlines():
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
        fid.close()


