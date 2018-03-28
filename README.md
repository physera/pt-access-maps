# pt-access-maps
Supplementary material for Physera's blog post on increasing access to physical therapy

## Data you'll need to run this
* CSV of lat/long for physical therapy locations.  We produced this by downloading the NPI and Physician Compare datasets (available at (https://www.cms.gov/Regulations-and-Guidance/Administrative-Simplification/NationalProvIdentStand/DataDissemination.html) and https://data.medicare.gov/data/physician-compare), filtering to only physical therapy related specialties, and converting the business street addresses to lat/long using a geocoding service.  
* US County shapefile from the census bureau: https://www.census.gov/geo/maps-data/data/cbf/cbf_counties.html
* US County level population density data from the census bureau: https://factfinder.census.gov/bkmk/table/1.0/en/DEC/10_SF1/GCTPH1.US05PR
* US County level updated population estimates from the census bureau: http://bit.ly/2GvVeUM
