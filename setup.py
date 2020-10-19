import setuptools
setuptools.setup(
     name='blackbox_mpc',
     version='0.2',
     author="Ossama Ahmed, Jonas Rothfuss",
     author_email="ossama.ahmed@mail.mcgill.ca, jonas.rothfuss@inf.ethz.ch",
     description="BlackBox MPC - Model Predictive Control with"
                  "sampling based optimizers",
     url="https://github.com/ossamaAhmed/blackbox_mpc",
     packages=setuptools.find_packages(),
     install_requires=[
        'tensorflow==2.0.0',
        'tensorflow-probability==0.8.0rc0',
        'gym',
        'numpy',
        'sphinx',
        'matplotlib',
        'sphinx_rtd_theme',
        'sphinxcontrib-bibtex',
      ],
    zip_safe=False
    )