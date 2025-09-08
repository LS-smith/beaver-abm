# Beaver Agent-Based Model (ABM)

A spatially-explicit agent-based model for simulating beaver population dynamics, territory establishment, dam construction, and subsequent flooding across a dynamic landscape.

## Overview

This repository contains the implementation of a beaver population dynamics model developed as part of a thesis project. The model simulates individual beaver agents with realistic life cyles, territorial behaviour, mating patterns, and dam-building activities within a spatially-explicit environment using real geographic data.

### Key Features

- **Agent-based modelling**: Each beaver agent has unique characteristics (age, sex, territory, mate)
- **Spatial representation**: Uses a DEM, habitat suitability map formed from land cover data, and water network data
- **Life cycle simulation**: Includes birth, maturation, reproduction, dispersal, and death
- **Territory establishment**: Territory formation and abandonment based on habitat quality
- **Dam construction**: Agents build dams that modify the landscape and flood landcover grid
- **Flooding impact**: Maps flooded areas over time to indicate potential conflict zones

## Model Architecture

The model is built using the [Mesa](https://mesa.readthedocs.io/) agent-based modeling framework and consists of several key components:

### Core Classes

- **`Beaver`** (`src/Agent.py`): Base beaver agent class with lifecycle methods
  - `Kit`: Young beavers (0-2 years)
  - `Juvenile`: Adolescent beavers (2-3 years) 
  - `Adult`: Mature beavers (3+ years) capable of reproduction
- **`Dam`** (`src/Agent.py`): Dam structures that modify local hydrology
- **`Flood_Model`** (`src/Model.py`): Main model class managing the simulation environment

### Input Data Requirements

The model uses three primary spatial datasets that are preprocessed to create the required input layers:

#### Primary Data Sources
- **Digital Elevation Model (DEM)**: `DTM.tif` - High-resolution terrain elevation data (5m resolution)
- **Land Cover Data**: `landcover.tif` - Vegetation and land use classification raster (10m resolution)
- **Water Network**: `Water_network.shp` - Vector line data of stream/river geometries with gradient attributes

#### Preprocessed Datasets (Generated from Primary Sources)
The model preprocessing scripts (`src/preprocess_hsm.py`, `src/distance_to_water.py`) generate:

- **Habitat Suitability Model**: `hsm_5m.tif` - Beaver habitat quality ratings (1-5 scale) derived from land cover data
- **Distance to Water**: `distance_to_water_5m.tif` - Proximity to water bodies in meters, calculated from water network data

All spatial datasets should be placed in the `data/` directory with consistent coordinate reference systems and 5m resolution.

### Data Sources

*Note: Due to file size limitations, spatial datasets are not included in this repository. Users must obtain the following data sources:*

#### Primary Data Sources
- **Digital Elevation Model (DTM)**: Available from [Ordnance Survey Terrain 5 DTM](https://www.ordnancesurvey.co.uk/business-government/products/terrain-5) or equivalent high-resolution elevation data for chosen study area
- **Land Cover Data**: [UK Centre for Ecology & Hydrology (UKCEH) Land Cover Maps](https://www.ceh.ac.uk/our-science/projects/land-cover-map) or equivalent landcover data for chosen study area
- **Water Network**: [Ordnance Survey Open Rivers](https://www.ordnancesurvey.co.uk/business-government/products/open-map-rivers) or equivalent stream network data with gradient attributes

*Preprocessing scripts in `src/` can be adapted for equivalent datasets from other regions or data providers.*

## Study Area

This model was developed and tested using spatial data from the Tay catchment in Scotland. The Tay is Scotland's longest river and largest catchment by discharge, covering approximately 5000 km² of diverse Highland and Lowland landscapes. The catchment's extensive river networks and riparian woodland provide a diverse network of potential beaver habitat.

The Tay catchment represents an ideal study system for beaver population modelling due to its:
- Recent beaver reintroduction efforts in Scotland and associated flooding impacts
- Diverse landscape gradients and habitat types, including expanses of human-modified landscapes
- Well-documented hydrological and terrain datasets

The model grid is derived from high-resolution spatial datasets covering the entire Tay catchment, allowing for landscape-scale simulations of beaver population dynamics and their impacts on regional hydrology.

## Installation

### Prerequisites

- Python 3.8+
- Git

### Setup

1. Clone the repository:
```bash
git clone https://github.com/LS-smith/beaver-abm.git
cd beaver-abm
```

2. Create a virtual environment (recommended):
```bash
python -m venv beaver-env
source beaver-env/bin/activate  # On Windows: beaver-env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r data/requirements.txt
```

### Required Python Packages

- `mesa` - Agent-based modeling framework
- `rasterio` - Geospatial raster data I/O
- `geopandas` - Vector geospatial data handling
- `numpy` - Numerical computing
- `scipy` - Scientific computing
- `shapely` - Geometric operations
- `scikit-image` - Image processing (for DEM downsampling)

## Usage

### Running the Model

The model is designed to be run through the main application script:

```bash
cd beaver-abm
python src/App.py
```

This will:
1. Load the DEM data from `./data/DTM.tif`
2. Initialise the model with 50 initial beavers
3. Run the simulation for 120 time steps (months)
4. Save results to `./out/Beaver_data.csv`

### Customizing the Simulation

You can modify the simulation parameters by editing `src/App.py`:

```python
# In App.py, modify these lines:
model = Flood_Model(
    dem=dem, 
    dem_transform=dem_transform, 
    initial_beavers=50,  # Change starting population
    seed=None           # Set seed for reproducibility (e.g., seed=42)
)

# Change simulation length
for i in range(120):  # Currently 120 months (10 years)
    model.step()
```

### Model Parameters

Key parameters that can be adjusted in `src/App.py`:

- `initial_beavers`: Starting population size (default: 50, placed as 25 mated pairs)
- `seed`: Random seed for reproducibility (default: None for random behaviour)
- Simulation length: Number of time steps (default: 120 months = 10 years)

Additional parameters in the model classes:
- Territory size distribution (lognormal with mean ~3km bank length)
- Dispersal distance (exponential with mean 1.4km)
- Breeding season (April-June, random month per female)
- Dam construction criteria (gradient ≤4%, within territory)
- Mortality rates (background 0.2% per month + age-related)

## Model Behaviour

### Beaver Life Cycle

1. **Birth**: Kits (1-4 per litter) are born to mated pairs during breeding season (April-June)
2. **Maturation**: Agents transition through life stages:
   - Kit (0-24 months): Stay with parents, cannot reproduce
   - Juvenile (24-36 months): Can disperse and mate, limited dam building
   - Adult (36+ months): Full behavioural repertoire
3. **Dispersal**: Young adults search for mates within 20km radius, disperse up to 7km if unsuccessful
4. **Territory Formation**: Successful pairs establish territories (0.5-20km bank length) near suitable habitat
5. **Reproduction**: Mated pairs produce 1-4 offspring annually during breeding season
6. **Mortality**: Agents die based on exponential age distribution (mean 84 months) plus background mortality

### Territory Dynamics

- Territories are established near water bodies (within 100m) with habitat suitability ratings 2-4
- Territory size follows lognormal distribution (mean ~3km bank length, range 0.5-20km)
- Territory quality based on habitat suitability model and distance to water
- Territories are abandoned after exponential timer (mean 48 months ≈ 4 years)
- Competition for prime habitat drives dispersal patterns
- Partners and offspring share the same territory

### Dam Construction

- Adult beavers build dams within their territories on waterways with gradient ≤4%
- Dam placement uses real stream network geometry from Water_network.shp
- Dams have realistic depths (normal distribution: mean 1.6m, range 0.55-2.0m)
- Dam construction includes flood modelling to assess land inundation
- Abandoned dams decay over ~2 years but can be repaired if territory is reoccupied
- Maximum 5 dam construction attempts per beaver per time step

## Output and Analysis

The model collects data on:

- Population demographics over time
- Spatial distribution of territories and dams
- Territory establishment and abandonment patterns
- Dam construction locations and timing
- Flood inundation areas

## File Structure

```
beaver-abm/
├── README.md
├── data/
│   ├── requirements.txt
│   ├── DTM.tif              # Digital Elevation Model
│   ├── hsm_5m.tif           # Habitat Suitability Model
│   ├── distance_to_water_5m.tif
│   ├── Water_network.shp    # Stream network
│   └── landcover.tif        # Land cover data
├── src/
│   ├── Agent.py             # Beaver and Dam agent classes
│   ├── Model.py             # Main simulation model
│   ├── App.py               # Application runner and visualization
│   ├── distance_to_water.py # Spatial analysis utilities
│   └── preprocess_hsm.py    # Data preprocessing scripts
└── out/                     # Output directory for results
```

## Contributing

This model was developed for academic research purposes. If you use this code in your research, please cite appropriately.

## License

This work is licensed under a Creative Commons Attribution 4.0 International License (CC BY 4.0).

See: https://creativecommons.org/licenses/by/4.0/

## Contact

**Leah Smith** - Masters Student  
Email: [leah.smith-4@postgrad.manchester.ac.uk]  
GitHub: [@LS-smith](https://github.com/LS-smith)

---

*This model was developed as part of a Masters thesis on beaver population dynamics and flooding impacts. The model is under active development and refinement.*