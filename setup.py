from setuptools import setup

setup(
    name='bpr-knn',
    version='0.0.2',
    description='Bayesian Personalised Ranking with adaptive KNN',
    author='Sun',
    author_email='jis252@eng.ucsd.edu',
    url='https://github.com/JingzheSun/Recommender_System_Adaptive_KNN',
    license='Free',
    packages=['bpr_knn'],
    install_requires=[
        'numpy'
    ],
)