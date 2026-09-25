# Brain MRI VQA Dataset

In-house patients from **Technical University of Munich University Hospital** are filtered out due to privacy reasons.

The MRI data should be downloaded from the original sources and placed under the `imaging-data` with the following structure:

```
imaging-data/
├── longitudinal
│   ├── glioma
│   │   ├── postop
│   │   │   ├── BraTS-GLI-02095
│   │   │   │   ├── BraTS-GLI-02095-100
│   │   │   │   ├── BraTS-GLI-02095-101
│   │   │   │   └── BraTS-GLI-02095-102
│   │   │   ├── BraTS-GLI-02122
│   │   │   │   ├── BraTS-GLI-02122-100
│   │   │   │   └── BraTS-GLI-02122-101
│   │   │   ├── BraTS-GLI-02275
│   │   │   │   ├── BraTS-GLI-02275-100
│   │   │   │   ├── BraTS-GLI-02275-101
│   │   │   │   └── BraTS-GLI-02275-102
│   │   │   ├── BraTS-GLI-02589
│   │   │   │   ├── BraTS-GLI-02589-100
│   │   │   │   └── BraTS-GLI-02589-101
│   │   │   └── BraTS-GLI-03047
│   │   │       ├── BraTS-GLI-03047-100
│   │   │       └── BraTS-GLI-03047-101
│   │   └── preop
│   │       ├── BraTS-GLI-00001
│   │       │   ├── BraTS-GLI-00001-000
│   │       │   └── BraTS-GLI-00001-001
│   │       ├── BraTS-GLI-00467
│   │       │   ├── BraTS-GLI-00467-000
│   │       │   └── BraTS-GLI-00467-001
│   │       ├── BraTS-GLI-00474
│   │       │   ├── BraTS-GLI-00474-000
│   │       │   └── BraTS-GLI-00474-001
│   │       ├── BraTS-GLI-00535
│   │       │   ├── BraTS-GLI-00535-000
│   │       │   └── BraTS-GLI-00535-001
│   │       ├── BraTS-GLI-00647
│   │       │   ├── BraTS-GLI-00647-000
│   │       │   └── BraTS-GLI-00647-001
│   │       ├── BraTS-GLI-00702
│   │       │   ├── BraTS-GLI-00702-000
│   │       │   └── BraTS-GLI-00702-001
│   │       ├── BraTS-GLI-00712
│   │       │   ├── BraTS-GLI-00712-000
│   │       │   └── BraTS-GLI-00712-001
│   │       ├── BraTS-GLI-00721
│   │       │   ├── BraTS-GLI-00721-000
│   │       │   └── BraTS-GLI-00721-001
│   │       ├── BraTS-GLI-00762
│   │       │   ├── BraTS-GLI-00762-000
│   │       │   └── BraTS-GLI-00762-001
│   │       ├── BraTS-GLI-00779
│   │       │   ├── BraTS-GLI-00779-000
│   │       │   └── BraTS-GLI-00779-001
│   │       ├── Patient-001-2
│   │       │   ├── week-044
│   │       │   └── week-056
│   │       ├── Patient-002-2
│   │       │   ├── week-040-2
│   │       │   └── week-047
│   │       ├── Patient-003-2
│   │       │   ├── week-027
│   │       │   └── week-038
│   │       ├── Patient-004
│   │       │   ├── week-000-2
│   │       │   └── week-020
│   │       ├── Patient-004-3
│   │       │   ├── week-071
│   │       │   └── week-086
│   │       ├── Patient-009-2
│   │       │   ├── week-024
│   │       │   └── week-066
│   │       ├── Patient-011-2
│   │       │   ├── week-024
│   │       │   └── week-028
│   │       ├── Patient-013-2
│   │       │   ├── week-017
│   │       │   └── week-030
│   │       ├── Patient-017
│   │       │   ├── week-001
│   │       │   └── week-016
│   │       ├── Patient-017-2
│   │       │   ├── week-016
│   │       │   └── week-035
│   │       ├── Patient-022
│   │       │   ├── week-001
│   │       │   └── week-037
│   │       ├── Patient-023-2
│   │       │   ├── week-046
│   │       │   └── week-084
│   │       ├── Patient-023-3
│   │       │   ├── week-084
│   │       │   └── week-097
│   │       ├── Patient-027
│   │       │   ├── week-000-2
│   │       │   └── week-014
│   │       ├── Patient-029
│   │       │   ├── week-000-2
│   │       │   └── week-010
│   │       ├── Patient-029-3
│   │       │   ├── week-207
│   │       │   └── week-223
│   │       ├── Patient-030
│   │       │   ├── week-001
│   │       │   └── week-017
│   │       ├── Patient-030-2
│   │       │   ├── week-041
│   │       │   └── week-052
│   │       ├── Patient-035
│   │       │   ├── week-008
│   │       │   └── week-019
│   │       ├── Patient-036
│   │       │   ├── week-002
│   │       │   └── week-005
│   │       ├── Patient-041
│   │       │   ├── week-020
│   │       │   └── week-038
│   │       ├── Patient-042
│   │       │   ├── week-010
│   │       │   └── week-022
│   │       ├── Patient-043
│   │       │   ├── week-018
│   │       │   └── week-030
│   │       ├── Patient-054
│   │       │   ├── week-053
│   │       │   └── week-066
│   │       ├── Patient-063
│   │       │   ├── week-047
│   │       │   └── week-077
│   │       ├── Patient-063-2
│   │       │   ├── week-105
│   │       │   └── week-118
│   │       ├── Patient-064
│   │       │   ├── week-002
│   │       │   └── week-020
│   │       ├── Patient-064-2
│   │       │   ├── week-049
│   │       │   └── week-062
│   │       ├── Patient-072
│   │       │   ├── week-029
│   │       │   └── week-037
│   │       └── Patient-078
│   │           ├── week-029
│   │           └── week-037
│   └── metastasis
│       ├── postop
│       └── preop
│           ├── BraTS-MET-00577
│           │   ├── BraTS-MET-00577-000
│           │   └── BraTS-MET-00577-001
│           ├── BraTS-MET-00613
│           │   ├── BraTS-MET-00613-000
│           │   └── BraTS-MET-00613-001
│           ├── BraTS-MET-00623
│           │   ├── BraTS-MET-00623-000
│           │   └── BraTS-MET-00623-001
│           ├── BraTS-MET-00644
│           │   ├── BraTS-MET-00644-000
│           │   └── BraTS-MET-00644-001
│           ├── proteas-p25
│           │   ├── p25-00
│           │   ├── p25-01
│           │   └── p25-02
│           ├── proteas-p36
│           │   ├── p36-00
│           │   ├── p36-01
│           │   ├── p36-02
│           │   └── p36-03
│           ├── UCSF-MET-00568
│           │   ├── UCSF-MET-00568-000
│           │   └── UCSF-MET-00568-001
│           ├── UCSF-MET-00588
│           │   ├── UCSF-MET-00588-000
│           │   └── UCSF-MET-00588-001
│           ├── UCSF-MET-00598
│           │   ├── UCSF-MET-00598-000
│           │   └── UCSF-MET-00598-001
│           └── UCSF-MET-00626
│               ├── UCSF-MET-00626-000
│               └── UCSF-MET-00626-001
└── single-timepoint
    ├── glioma
    │   ├── postop
    │   │   ├── BraTS-GLI-02255-100
    │   │   ├── BraTS-GLI-02280-100
    │   │   ├── BraTS-GLI-02397-100
    │   │   ├── BraTS-GLI-02680-100
    │   │   ├── BraTS-GLI-02715-100
    │   │   ├── BraTS-GLI-03046-100
    │   │   └── BraTS-GLI-03058-100
    │   └── preop
    │       ├── BraTS-GLI-00671-000
    │       ├── egd-0005
    │       ├── egd-0010
    │       ├── egd-0015
    │       ├── ucsf-pdgm-0004
    │       ├── ucsf-pdgm-0010
    │       ├── ucsf-pdgm-0020
    │       ├── upenn-gbm-00005
    │       ├── upenn-gbm-00010
    │       └── upenn-gbm-00025
    ├── meningioma
    │   ├── postop
    │   └── preop
    │       ├── BraTS-MEN-00008-000
    │       ├── BraTS-MEN-00031-000
    │       ├── BraTS-MEN-00261-000
    │       ├── BraTS-MEN-00373-000
    │       ├── BraTS-MEN-00500-000
    │       ├── BraTS-MEN-01211-000
    │       └── BraTS-MEN-01402-000
    └── metastasis
        ├── postop
        └── preop
            ├── BraTS-MET-00141-000
            ├── BraTS-MET-00200-000
            ├── BraTS-MET-00208-000
            ├── BraTS-MET-00560-000
            ├── BraTS-MET-00575-000
            ├── BraTS-MET-00650-000
            ├── BraTS-MET-00767-000
            ├── BraTS-MET-00777-000
            ├── BraTS-MET-00786-000
            ├── BraTS-MET-00792-000
            ├── UCSF-MET-00554-000
            ├── UCSF-MET-00610-000
            ├── UCSF-MET-00650-000
            ├── UCSF-MET-00735-000
```


Dataset sources and patient IDs:

## Erasmus Glioma Database
DOI: https://doi.org/10.1016/j.dib.2021.107191 

- egd-0005
- egd-0010
- egd-0015

## UPenn-GBM
DOI: https://doi.org/10.1038/s41597-022-01560-7 

- upenn-gbm-00005
- upenn-gbm-00010
- upenn-gbm-00025

## UCSF-PDGM
DOI: https://doi.org/10.1148/ryai.220058 

- ucsf-pgdm-0014
- ucsf-pgdm-0010
- ucsf-pgdm-0020

## LUMIERE
DOI: https://doi.org/10.1038/s41597-022-01881-7 

- Patient-001: week-044  week-056

- Patient-002-2: week-040-2  week-047

- Patient-003: week-027  week-038

- Patient-004: week-000-2  week-020

- Patient-004: week-071  week-086

- Patient-009: week-024  week-066

- Patient-011: week-024  week-028

- Patient-013: week-017  week-030

- Patient-017: week-001  week-016

- Patient-017-2: week-016  week-035

- Patient-022: week-001  week-037

- Patient-023: week-046  week-084

- Patient-023: week-084  week-097

- Patient-027: week-000-2  week-014

- Patient-029: week-000-2  week-010

- Patient-029-3: week-207  week-223

- Patient-030: week-001  week-017

- Patient-030: week-041  week-052

- Patient-035: week-008  week-019

- Patient-036: week-002  week-005

- Patient-041: week-020  week-038

- Patient-042: week-010  week-022

- Patient-043: week-018  week-030

- Patient-054: week-053  week-066

- Patient-063: week-047  week-077

- Patient-063: week-105  week-118

- Patient-064: week-002  week-020

- Patient-064: week-049  week-062

- Patient-072: week-029  week-037

- Patient-078: week-029  week-037


## BraTS-Glioma Pre-Op
DOI: https://arxiv.org/abs/2107.02314 

Single-timepoint:
- BraTS-GLI-00702-000

Longitudinal:
- BraTS-GLI-00001-000
- BraTS-GLI-00001-001
- BraTS-GLI-00467-000
- BraTS-GLI-00467-001
- BraTS-GLI-00474-000
- BraTS-GLI-00474-001
- BraTS-GLI-00535-000
- BraTS-GLI-00535-001
- BraTS-GLI-00647-000
- BraTS-GLI-00647-001
- BraTS-GLI-00671-000
- BraTS-GLI-00702-000
- BraTS-GLI-00702-001
- BraTS-GLI-00712-000
- BraTS-GLI-00712-001
- BraTS-GLI-00721-000
- BraTS-GLI-00721-001
- BraTS-GLI-00762-000
- BraTS-GLI-00762-001
- BraTS-GLI-00779-000
- BraTS-GLI-00779-001

## BraTS-Glioma Post-Op
DOI: https://arxiv.org/abs/2405.18368 

Single-timepoint:
- BraTS-GLI-02255-100  
- BraTS-GLI-02280-100  
- BraTS-GLI-02397-100  
- BraTS-GLI-02680-100  
- BraTS-GLI-02715-100  
- BraTS-GLI-03046-100  
- BraTS-GLI-03058-100

Longitudinal:
- BraTS-GLI-02095-100
- BraTS-GLI-02095-101
- BraTS-GLI-02095-102
- BraTS-GLI-02122-100
- BraTS-GLI-02122-101 
- BraTS-GLI-02275-100
- BraTS-GLI-02275-101
- BraTS-GLI-02275-102
- BraTS-GLI-02589-100
- BraTS-GLI-02589-101
- BraTS-GLI-03047-100
- BraTS-GLI-03047-101

## BraTS-MET
DOI: https://arxiv.org/abs/2306.00838 

Single-timepoint: 
- BraTS-MET-00141-000
- BraTS-MET-00200-000
- BraTS-MET-00208-000
- BraTS-MET-00560-000
- BraTS-MET-00575-000
- BraTS-MET-00650-000
- BraTS-MET-00767-000
- BraTS-MET-00777-000
- BraTS-MET-00786-000
- BraTS-MET-00792-000

Longitudinal:
- BraTS-MET-00577-000
- BraTS-MET-00577-001
- BraTS-MET-00623-000
- BraTS-MET-00623-001
- BraTS-MET-00644-000
- BraTS-MET-00644-001

## UCSF-BMSR
DOI: https://doi.org/10.1148/ryai.230126 

Single-timepoint: 
- UCSF-MET-00554-000
- UCSF-MET-00610-000
- UCSF-MET-00650-000
- UCSF-MET-00735-000

Longitudinal:
- UCSF-MET-00568-000
- UCSF-MET-00568-001
- UCSF-MET-00588-000
- UCSF-MET-00588-001
- UCSF-MET-00598-000
- UCSF-MET-00598-001
- UCSF-MET-00626-000
- UCSF-MET-00626-001

## PROTEAS Metastasis
DOI: https://doi.org/10.1038/s41597-025-06131-0

Longitudinal:
- p25
- p36

## BraTS Meningioma
DOI: https://arxiv.org/abs/2305.07642

- BraTS-MEN-00008-000  
- BraTS-MEN-00031-000  
- BraTS-MEN-00261-000  
- BraTS-MEN-00373-000  
- BraTS-MEN-00402-000
- BraTS-MEN-00500-000  
- BraTS-MEN-01211-000  
