from setuptools import setup, find_packages

VERSION = '0.0.1' 
DESCRIPTION = 'Auto regressive bagging and boosting package'
LONG_DESCRIPTION = 'Forecating using historical data as predictors'

# Setting up
setup(
        name="ARBB", 
        version=VERSION,
        author="Mustafa Aslan",
        author_email="<mustafaslan63@email.com>",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        url='https://github.com/mustafaslanCoto/ARML',
        install_requires=["xgboost", "lightgbm", "catboost", "pandas",
                          "numpy", "sklearn", "datetime", "hyperopt","statsmodels", "seaborn", "statsforecast",
                          "matplotlib", "window_ops", "cubist", "scipy", "numba"], # add any additional packages that 
        # needs to be installed along with your package. Eg: 'caer'
        
        keywords=['python', 'first package'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Data Scientists",
            "Programming Language :: Python :: 2",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
        ]
)