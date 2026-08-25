# Author: Paul David Harris
# Created: 04/05/2026
# Purpose: Download files from Zenodo repository

import pooch

DATASET_DIR = u'data'

repo = pooch.create(path=DATASET_DIR, base_url='doi:10.5281/zenodo.20038738')
repo.load_registry_from_doi()

files = ('HP3_TE300_SPC630.hdf5', 'dsdna_d7_d17_50_50_1.hdf5', 
         '12d_New_30p_320mW_steer_3.hdf5', '0023uLRpitc_NTP_20dT_0.5GndCl.hdf5',
         'dsdna_d7d17_50_50_1.spc', 'dsdna_d7d17_50_50_1.set',
         
         )

for file in files:
    repo.fetch(file)
