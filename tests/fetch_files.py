# Author: Paul David Harris
# Created: 04/05/2026
# Purpose: Download files from Zenodo repository

import pooch

DATASET_DIR = u'.'

repo = pooch.create(path=DATASET_DIR, base_url='doi:10.5281/zenodo.20038738')
repo.load_registry_from_doi()

files = ('HP3_TE300_SPC630.hdf5',         
         )

for file in files:
    repo.fetch(file)
