from setuptools import setup, find_packages

setup(
  name = 'getstmpy',
  version = '0.1.7',
  packages = find_packages(),
  package_data = {
    "":["*.txt", "*.mat"]
    },
  include_package_data=True,
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
