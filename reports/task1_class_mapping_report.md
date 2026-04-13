# Task 1 Class Mapping Report

## Inventory Ontology

- insan
- kara_tasiti
- deniz_tasiti
- hava_araci
- uap_uai_ozel
- ignore

## Training Ontology

- 0 = kara_tasiti
- 1 = insan
- 2 = uap_uai_ozel

## Local TEKNOFEST Mapping

- local class 0 -> kara_tasiti
- local class 1 -> insan
- local class 2 -> uap_uai_ozel
- local class 3 -> uap_uai_ozel

## Public Dataset Decisions

- VisDrone pedestrian/person/people -> insan
- VisDrone vehicle-like classes -> kara_tasiti
- UAVDT vehicle classes -> kara_tasiti
- HIT-UAV person -> insan; car/bicycle/other vehicle -> kara_tasiti
- SeaDronesSee boat/jetski/lifesaving appliance/buoy -> deniz_tasiti; swimmer -> insan

## Notes

- deniz_tasiti and hava_araci stay visible in inventory truth but are ignored in the first RGB training runs.
- UAP/UAI public coverage is assumed absent unless local custom labels provide it.