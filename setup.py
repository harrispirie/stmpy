from setuptools import setup

setup(
  name = 'getstmpy',
  packages = ['stmpy'],
  version = '0.1',
  license='MIT',
  description = 'Scanning tunneling microscopy data analysis suite',
  author = 'Harris Pirie',
  author_email = 'hpirie@live.com',
  url = 'https://github.com/harrispirie/stmpy',
  download_url = 'https://github.com/harrispirie/stmpy/archive/v0.1.tar.gz',
  keywords = ['STM', 'Python', 'Data Analysis'],
  install_requires=[
          'numpy',
          'scipy',
          'matplotlib',
          'opencv-python',
          'scikit-image'
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3'
  ],
)
