from distutils.core import setup
setup(
    name='cwtLayerKeras',         # How you named your package folder (MyLib)
    packages=['cwtLayerKeras'],   # Chose the same as "name"
    version='0.1',      # Start with a small number and increase it with every change you make
    # Chose a license from here: https://help.github.com/articles/licensing-a-repository
    license='GNU AGPLv3',
    # Give a short description about your library
    description='Keras layer to compute the cwt(continuous wavelet transformation) scalogram of signals',
    author='fmolivato',                   # Type in your name
    author_email='f.olivato.97@gmail.com',      # Type in your E-Mail
    # Provide either the link to your github or to your website
    url='https://github.com/fmolivato/cwtLayerKeras',
    # I explain this later on
    download_url='https://github.com/fmolivato/cwtLayerKeras/archive/refs/tags/v_01.tar.gz',
    # Keywords that define your package best
    keywords=['keras', 'tensorflow2', 'scalogram', 'wavelet'],
    install_requires=[            # I get to this in a second
            'numpy',
            'tensorflow',
    ],
    classifiers=[
        # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        'Development Status :: 3 - Alpha',
        # Define that your audience are developers
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: GNU AGPLv3 License',   # Again, pick a license
        # Specify which pyhton versions that you want to support
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
)
