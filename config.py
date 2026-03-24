"""
Master Thesis Configuration — European Automotive Price Forecasting (2021–2025)
Hedonic Pricing Models vs Machine Learning with Temporal Generalization Testing
"""
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent
DATA_RAW    = ROOT / 'data' / 'raw'
DATA_PROC   = ROOT / 'data' / 'processed'
DATA_SPLITS = ROOT / 'data' / 'splits'
FIGURES     = ROOT / 'figures'
MODELS_DIR  = ROOT / 'models'
NOTEBOOKS   = ROOT / 'notebooks'

for p in [DATA_RAW, DATA_PROC, DATA_SPLITS, FIGURES, MODELS_DIR, NOTEBOOKS]:
    p.mkdir(parents=True, exist_ok=True)

# ── Raw file names ─────────────────────────────────────────────────────────
FILE_MAP = {
    2021: DATA_RAW / 'autoscout24_2021.csv',
    2022: DATA_RAW / 'autoscout24_2022.csv',
    2023: DATA_RAW / 'autoscout24_2023.csv',
    2024: DATA_RAW / 'autoscout24_2024.csv',
    2025: DATA_RAW / 'autoscout24_full.csv',
}
CSV_DELIMITERS = {2021: ';', 2022: ';', 2023: ';', 2024: ';', 2025: ';'}

MATERIALS_FILE = DATA_RAW / 'ev_materials_prices_2021_2025.csv'

# ── Schema mapping: each year's column name → canonical name ───────────────
# Canonical columns: make, model, price, currency, condition, body_type,
#   mileage_km, production_year, power_hp, engine_cc, fuel_type, transmission,
#   drivetrain, doors, color, country, co2_emission, seats, gears,
#   first_registration, seller_type, equipment_list
SCHEMA_MAP = {
    2021: {
        'Vehicle_brand': 'make',
        'Vehicle_model': 'model',
        'Price': 'price',
        'Currency': 'currency',
        'Condition': 'condition',
        'Type': 'body_type',
        'Mileage_km': 'mileage_km',
        'Production_year': 'production_year',
        'Power_HP': 'power_hp',
        'Displacement_cm3': 'engine_cc',
        'Fuel_type': 'fuel_type',
        'Transmission': 'transmission',
        'Drive': 'drivetrain',
        'Doors_number': 'doors',
        'Colour': 'color',
        'country': 'country_raw',
        'CO2_emissions': 'co2_emission',
        'Features': 'equipment_list',
        'First_owner': 'first_owner',
        'Vehicle_version': 'model_version',
    },
    2022: {
        'make': 'make',
        'model': 'model',
        'price': 'price',
        'body_type': 'body_type',
        'type': 'condition',
        'mileage': 'mileage_km',
        'first_registration': 'first_registration',
        'power': 'power_hp',
        'engine_size': 'engine_cc',
        'fuel_type': 'fuel_type',
        'gearbox': 'transmission',
        'drivetrain': 'drivetrain',
        'doors': 'doors',
        'colour': 'color',
        'country': 'country_raw',
        'co_emissions': 'co2_emission',
        'seats': 'seats',
        'gears': 'gears',
        'seller': 'seller_type',
        'paint': 'paint_type',
        'upholstery': 'upholstery',
        'previous_owner': 'nr_prev_owners',
        'full_service_history': 'full_service_history',
        'non_smoker_vehicle': 'non_smoker',
        'emission_class': 'emission_class',
        'comfort_&_convenience': 'eq_comfort',
        'entertainment_&_media': 'eq_entertainment',
        'safety_&_security': 'eq_safety',
        'extras': 'eq_extra',
    },
    2023: {
        'Brand': 'make',
        'ModelC': 'model',
        'Price': 'price',
        'Carrosserie': 'body_type',
        'État': 'condition',
        'Kilométrage': 'mileage_km',
        'Année': 'production_year',
        'Puissance kW': 'power_kw',
        'Transmission': 'transmission',
        'Carburant': 'fuel_type',
        'Portes': 'doors',
        'Couleur extérieure': 'color',
        'Country': 'country_raw',
        'Climatisation': 'has_ac',
        'Climatisation automatique': 'has_auto_ac',
        'Système de navigation': 'has_navigation',
        'Volant multifonctions': 'has_multifunction_steering',
        'Capteurs d\'aide au stationnement arrière': 'has_rear_parking_sensors',
        'Capteurs d\'aide au stationnement avant': 'has_front_parking_sensors',
    },
    2024: {
        'Make': 'make',
        'Model': 'model',
        'Price': 'price',
        'Body': 'body_type',
        'Condition': 'condition',
        'Mileage_km': 'mileage_km',
        'Year': 'production_year',
        'Power_hp': 'power_hp',
        'Engine_Size_cc': 'engine_cc',
        'Fuel_Type': 'fuel_type',
        'Gearbox': 'transmission',
        'Drivetrain': 'drivetrain',
        'Doors': 'doors',
        'Color': 'color',
        'Country': 'country_raw',
        'Fuel_Consumption_l': 'fuel_consumption',
        'Seats': 'seats',
        'Gears': 'gears',
        'Cylinders': 'cylinders',
        'Seller': 'seller_type',
        'Upholstery': 'upholstery',
        'Previous_Owners': 'nr_prev_owners',
        'Full_Service_History': 'full_service_history',
        'Non_Smoker_Vehicle': 'non_smoker',
    },
    2025: {
        'make': 'make',
        'model': 'model',
        'price': 'price',
        'price_currency': 'currency',
        'body_type': 'body_type',
        'vehicle_type': 'condition',
        'mileage_km_raw': 'mileage_km',
        'registration_date': 'registration_date',
        'production_year': 'production_year',
        'power_hp': 'power_hp',
        'cylinders_volume_cc': 'engine_cc',
        'fuel_category': 'fuel_type',
        'transmission': 'transmission',
        'drive_train': 'drivetrain',
        'nr_doors': 'doors',
        'body_color': 'color',
        'country_code': 'country_raw',
        'co2_emission_grper_km': 'co2_emission',
        'nr_seats': 'seats',
        'gears': 'gears',
        'cylinders': 'cylinders',
        'seller_type': 'seller_type',
        'paint_type': 'paint_type',
        'upholstery': 'upholstery',
        'nr_prev_owners': 'nr_prev_owners',
        'has_full_service_history': 'full_service_history',
        'non_smoking': 'non_smoker',
        'had_accident': 'had_accident',
        'envir_standard': 'emission_class',
        'electric_range_km': 'electric_range_km',
        'power_kw': 'power_kw',
        'weight_kg': 'weight_kg',
        'equipment_comfort': 'eq_comfort',
        'equipment_entertainment': 'eq_entertainment',
        'equipment_safety': 'eq_safety',
        'equipment_extra': 'eq_extra',
        'registration_date': 'first_registration',
        'city': 'city',
        'zip': 'zip',
    },
}

# ── Price bounds (EUR) — adjusted for realistic European market ────────────
# Lower bound: €800 excludes scrap/parts-only listings
# Upper bound: €300,000 excludes ultra-luxury/exotics that distort distributions
PRICE_MIN_EUR = 800
PRICE_MAX_EUR = 300_000

# ── FX rates: currency → EUR, by year ─────────────────────────────────────
AVG_FX_TO_EUR = {
    'PLN': {2021: 1/4.5596, 2022: 1/4.6877, 2023: 1/4.5430, 2024: 1/4.2684, 2025: 1/4.2500},
    'GBP': {2021: 1.163,    2022: 1.173,    2023: 1.151,    2024: 1.252,    2025: 1.270},
    'CHF': {2021: 0.910,    2022: 0.992,    2023: 1.069,    2024: 1.046,    2025: 1.060},
    'SEK': {2021: 0.099,    2022: 0.093,    2023: 0.087,    2024: 0.089,    2025: 0.088},
    'DKK': {2021: 0.134,    2022: 0.135,    2023: 0.134,    2024: 0.134,    2025: 0.134},
    'CZK': {2021: 0.039,    2022: 0.041,    2023: 0.042,    2024: 0.041,    2025: 0.041},
    'EUR': {2021: 1.0,      2022: 1.0,      2023: 1.0,      2024: 1.0,      2025: 1.0},
}

# ── Country ISO-2 lookup ───────────────────────────────────────────────────
COUNTRY_TO_ISO2 = {
    'poland': 'PL', 'polska': 'PL', 'germany': 'DE', 'deutschland': 'DE',
    'france': 'FR', 'italy': 'IT', 'italia': 'IT', 'spain': 'ES', 'españa': 'ES',
    'netherlands': 'NL', 'nl': 'NL', 'belgium': 'BE', 'belgique': 'BE',
    'austria': 'AT', 'österreich': 'AT', 'switzerland': 'CH', 'sweden': 'SE',
    'denmark': 'DK', 'norway': 'NO', 'finland': 'FI', 'czech republic': 'CZ',
    'czechia': 'CZ', 'slovakia': 'SK', 'hungary': 'HU', 'romania': 'RO',
    'luxembourg': 'LU', 'great britain': 'GB', 'portugal': 'PT', 'greece': 'GR',
    'croatia': 'HR', 'ireland': 'IE', 'slovenia': 'SI', 'bulgaria': 'BG',
    'polska': 'PL', 'other': 'XX', 'united states': 'US', 'canada': 'CA',
    'de': 'DE', 'fr': 'FR', 'it': 'IT', 'es': 'ES', 'be': 'BE', 'at': 'AT',
    'lu': 'LU', 'pl': 'PL', 'nl': 'NL', 'ch': 'CH', 'se': 'SE', 'dk': 'DK',
    'no': 'NO', 'fi': 'FI', 'cz': 'CZ', 'sk': 'SK', 'hu': 'HU', 'ro': 'RO',
    'hr': 'HR', 'ie': 'IE', 'si': 'SI', 'bg': 'BG', 'gb': 'GB', 'pt': 'PT',
    'gr': 'GR',
}

# ── Fuel harmonization ─────────────────────────────────────────────────────
FUEL_HARMONIZE = {
    # Gasoline variants
    'gasoline': 'Gasoline', 'essence': 'Gasoline', 'benzine': 'Gasoline',
    'petrol': 'Gasoline', 'benzin': 'Gasoline',
    'super 95': 'Gasoline', 'super plus 98': 'Gasoline',
    'super e10 95': 'Gasoline', 'super plus e10 98': 'Gasoline',
    'regular/benzine 91': 'Gasoline', 'regular/benzine e10 91': 'Gasoline',
    'essence 91': 'Gasoline', 'essence e10 91': 'Gasoline',
    'essence (filtre à particules)': 'Gasoline',
    'essence 91 (filtre à particules)': 'Gasoline',
    'essence e10 91 (filtre à particules)': 'Gasoline',
    'super 95 (filtre à particules)': 'Gasoline',
    'super e10 95 (filtre à particules)': 'Gasoline',
    'super plus 98 (filtre à particules)': 'Gasoline',
    'super plus e10 98 (filtre à particules)': 'Gasoline',
    # Diesel variants
    'diesel': 'Diesel',
    'diesel (filtre à particules)': 'Diesel',
    'biodiesel': 'Diesel', 'diesel écologique': 'Diesel',
    'diesel écologique (filtre à particules)': 'Diesel',
    # Electric
    'electric': 'Electric', 'electricity': 'Electric',
    'electrique': 'Electric', 'électrique': 'Electric', 'elektrisch': 'Electric',
    'electrique (filtre à particules)': 'Electric',
    'électrique (filtre à particules)': 'Electric',
    # Hybrid / PHEV (including fuel_category format from 2025)
    'electric/gasoline': 'Hybrid_PHEV', 'electric/diesel': 'Hybrid_Diesel',
    'hybrid': 'Hybrid_PHEV', 'hybride': 'Hybrid_PHEV',
    'plug-in hybrid': 'Hybrid_PHEV', 'hybride rechargeable': 'Hybrid_PHEV',
    'gasoline/electric': 'Hybrid_PHEV', 'diesel/electric': 'Hybrid_Diesel',
    # Gas
    'gasoline + lpg': 'LPG', 'lpg': 'LPG',
    'liquid petroleum gas (lpg)': 'LPG',
    'gaz de pétrole liquéfié': 'LPG', 'gpl': 'LPG',
    'cng': 'CNG', 'gasoline + cng': 'CNG',
    'gnl': 'CNG', 'gaz naturel h': 'CNG', 'domestic gas h': 'CNG', 'biogas': 'CNG',
    # Other
    'others': 'Other', 'other': 'Other', 'hydrogen': 'Other',
    'hydrogène': 'Other', 'ethanol': 'Other', 'e85': 'Other',
    'autres': 'Other', 'autres (filtre à particules)': 'Other',
    'vegetable oil': 'Other', 'vegetable oil (filtre à particules)': 'Other',
}

EV_FUELS   = {'Electric'}
PHEV_FUELS = {'Hybrid_PHEV', 'Hybrid_Diesel'}
ICE_FUELS  = {'Gasoline', 'Diesel', 'LPG', 'CNG'}

# ── Powertrain segmentation ───────────────────────────────────────────────
def get_powertrain(fuel_type):
    if fuel_type in EV_FUELS:   return 'EV'
    if fuel_type in PHEV_FUELS: return 'PHEV'
    if fuel_type in ICE_FUELS:  return 'ICE'
    return 'Other'

# ── Body type harmonization ───────────────────────────────────────────────
BODY_HARMONIZE = {
    'compact': 'Hatchback', 'small_cars': 'Hatchback', 'citadine': 'Hatchback',
    'hatchback': 'Hatchback',
    'sedan': 'Sedan', 'saloon': 'Sedan', 'berline': 'Sedan', 'limousine': 'Sedan',
    'estate': 'Estate', 'station wagon': 'Estate', 'break': 'Estate', 'kombi': 'Estate',
    'suv': 'SUV', 'off-road': 'SUV', 'suv / tout-terrain': 'SUV', 'off-road/pick-up': 'SUV',
    'coupe': 'Coupe', 'coupé': 'Coupe', 'sports car': 'Coupe',
    'convertible': 'Convertible', 'cabriolet': 'Convertible', 'cabrio/roadster': 'Convertible',
    'van': 'Van', 'minivan': 'Van', 'monospace': 'Van', 'utilitaire': 'Van',
    'transporter': 'Van', 'bus': 'Van',
    'pickup': 'Pickup', 'pick-up': 'Pickup',
    'other': 'Other',
}

# ── Transmission harmonization ─────────────────────────────────────────────
TRANSMISSION_HARMONIZE = {
    'manual': 'Manual', 'boîte manuelle': 'Manual', 'manuell': 'Manual',
    'automatic': 'Automatic', 'boîte automatique': 'Automatic', 'automatik': 'Automatic',
    'semi-automatic': 'Semi-Automatic', 'semi-automatique': 'Semi-Automatic',
}

# ── Condition harmonization ────────────────────────────────────────────────
CONDITION_HARMONIZE = {
    'used': 'Used', 'occasion': 'Used', 'gebraucht': 'Used',
    'new': 'New', 'neuf': 'New', 'neu': 'New',
    'pre-registered': 'Pre-Registered', 'pre-registration': 'Pre-Registered',
    "day's registration": 'Pre-Registered', 'demonstration': 'Used',
    'ancêtre': 'Used', 'oldtimer': 'Used', 'classic': 'Used',
}

# ── EV materials ───────────────────────────────────────────────────────────
MATERIALS = ['Lithium', 'Cobalt', 'Nickel', 'Graphite', 'Neodymium (Nd)', 'Dysprosium (Dy)']
BATTERY_MATERIALS = ['Lithium', 'Cobalt', 'Nickel', 'Graphite']
MOTOR_MATERIALS   = ['Neodymium (Nd)', 'Dysprosium (Dy)']

# ── Modelling ──────────────────────────────────────────────────────────────
RANDOM_STATE = 42
TARGET_COL   = 'price_eur'
LOG_TARGET   = 'log_price'

# Years included (2021/2022 dropped due to incompatible schemas & populations)
DATA_YEARS = [2023, 2024, 2025]

# Primary evaluation: stratified random split 70/15/15
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
STRATIFY_COLS = ['country_code', 'make_grouped', 'data_year']

# Secondary evaluation: leave-one-year-out cross-validation

# ── Color grouping ─────────────────────────────────────────────────────────
NEUTRAL_COLORS = {'black', 'white', 'grey', 'gray', 'silver', 'noir', 'blanc',
                  'gris', 'argent', 'schwarz', 'weiß', 'weiss', 'silber', 'grau'}
