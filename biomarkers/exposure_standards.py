"""
Exposure Data Standards and Schema Definitions

This module defines standardized schemas for exposure data ingestion across:
- Environmental monitoring (EPA, satellite)
- Chemical biomarkers (NHANES, biomonitoring)
- Lifestyle/behavioral data (wearables, surveys)
- Spatial-temporal harmonization

Follows FAIR principles with UCUM units, standard ontology IDs (DSSTox/CHEBI/ExO),
and temporal alignment specifications for multi-modal integration.

Author: AI Pipeline Team
Date: September 2025
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ExposureType(Enum):
    """Standardized exposure types with ExO/CHEBI mappings"""

    AIR_QUALITY = "air_quality"
    CHEMICAL_BIOMARKER = "chemical_biomarker"
    BUILT_ENVIRONMENT = "built_environment"
    LIFESTYLE_BEHAVIORAL = "lifestyle_behavioral"
    OCCUPATIONAL = "occupational"
    DIETARY = "dietary"
    RADIATION = "radiation"


class TemporalResolution(Enum):
    """Standard temporal resolutions for exposure measurements"""

    INSTANTANEOUS = "instantaneous"
    MINUTE = "1min"
    HOURLY = "1h"
    DAILY = "1d"
    WEEKLY = "1w"
    MONTHLY = "1mo"
    ANNUAL = "1y"


class SpatialResolution(Enum):
    """Standard spatial resolutions"""

    POINT = "point"  # GPS coordinates
    CENSUS_BLOCK = "census_block"
    ZIP_CODE = "zip_code"
    COUNTY = "county"
    GRID_1KM = "grid_1km"
    GRID_10KM = "grid_10km"


@dataclass
class ExposureRecord:
    """Standardized exposure record with full provenance"""

    # Core identifiers
    subject_id: str
    exposure_id: str  # Unique exposure measurement ID
    analyte_id: str  # DSSTox/CHEBI/ExO ID
    analyte_name: str

    # Measurement
    measured_at: datetime
    measurement_window: timedelta  # Duration of measurement/average
    value: float
    unit: str  # UCUM-compatible unit
    detection_limit: Optional[float] = None

    # Quality and uncertainty
    measurement_quality: str = "good"  # good/fair/poor/invalid
    uncertainty: Optional[float] = None
    measurement_method: Optional[str] = None

    # Spatial context
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    location_type: str = "unknown"  # home/work/transit/outdoor/indoor
    spatial_resolution: SpatialResolution = SpatialResolution.POINT

    # Temporal context
    temporal_resolution: TemporalResolution = TemporalResolution.INSTANTANEOUS
    season: Optional[str] = None  # spring/summer/fall/winter
    time_of_day: Optional[str] = None  # morning/afternoon/evening/night

    # Data source and provenance
    data_source: str  # EPA_AQS/NHANES/device_serial/survey_id
    collection_protocol: Optional[str] = None
    lab_method: Optional[str] = None
    data_version: str = "1.0"

    # Biological context (for biomarkers)
    sample_type: Optional[str] = None  # blood/urine/saliva/hair
    fasting_status: Optional[bool] = None
    medication_interference: Optional[bool] = None

    # Metadata
    exposure_type: ExposureType = ExposureType.AIR_QUALITY
    created_at: datetime = field(default_factory=datetime.now)
    notes: Optional[str] = None


@dataclass
class ExposureDataset:
    """Collection of exposure records with metadata"""

    records: List[ExposureRecord]
    dataset_id: str
    dataset_name: str
    exposure_types: List[ExposureType]

    # Temporal coverage
    start_date: datetime
    end_date: datetime
    temporal_resolution: TemporalResolution

    # Spatial coverage
    spatial_extent: Dict[str, float]  # min_lat, max_lat, min_lon, max_lon
    spatial_resolution: SpatialResolution

    # Subject information
    n_subjects: int
    subject_demographics: Optional[pd.DataFrame] = None

    # Data quality
    completeness_score: float = 0.0  # 0-1
    data_quality_flags: List[str] = field(default_factory=list)

    # Provenance
    data_sources: List[str] = field(default_factory=list)
    processing_steps: List[str] = field(default_factory=list)
    validation_status: str = "unvalidated"

    created_at: datetime = field(default_factory=datetime.now)


class ExposureStandardizer:
    """Standardize exposure data from various sources to common schema"""

    def __init__(self):
        self.ontology_mappings = self._load_ontology_mappings()
        self.unit_conversions = self._load_unit_conversions()

    def _load_ontology_mappings(self) -> Dict[str, Dict[str, str]]:
        """Load mappings from common names to standard ontology IDs"""
        return {
            "air_pollutants": {
                "PM2.5": "CHEBI:132076",
                "PM10": "CHEBI:132078",
                "NO2": "CHEBI:17632",
                "O3": "CHEBI:25812",
                "SO2": "CHEBI:18422",
                "CO": "CHEBI:17245",
            },
            "metals": {
                "Lead": "CHEBI:25016",
                "Mercury": "CHEBI:16170",
                "Cadmium": "CHEBI:22977",
                "Arsenic": "CHEBI:22632",
            },
            "pfas": {
                "PFOA": "CHEBI:39421",
                "PFOS": "CHEBI:39422",
                "PFNA": "CHEBI:131200",
                "PFHxS": "CHEBI:131204",
            },
            "pesticides": {
                "Atrazine": "CHEBI:15930",
                "Glyphosate": "CHEBI:27744",
                "2,4-D": "CHEBI:28854",
            },
        }

    def _load_unit_conversions(self) -> Dict[str, Dict[str, float]]:
        """Load UCUM-compatible unit conversion factors"""
        return {
            "concentration": {
                "ug/m3": 1.0,  # Base unit
                "mg/m3": 1000.0,
                "ng/m3": 0.001,
                "ppb": 1.0,  # Context-dependent conversion
                "ppm": 1000.0,
            },
            "biomarker": {
                "ng/mL": 1.0,  # Base unit
                "ug/L": 1.0,
                "pg/mL": 0.001,
                "mg/L": 1000.0,
                "nmol/L": 1.0,  # Molar units
            },
        }

    def standardize_epa_aqs(self, raw_data: pd.DataFrame) -> ExposureDataset:
        """Standardize EPA AQS air quality data"""

        logger.info(f"Standardizing EPA AQS data: {len(raw_data)} records")

        records = []
        for _, row in raw_data.iterrows():

            # Map pollutant to standard ID
            pollutant = row.get("Parameter Name", "")
            analyte_id = self._map_to_ontology_id(pollutant, "air_pollutants")

            # Parse datetime
            date_str = row.get("Date Local", "")
            time_str = row.get("Time Local", "00:00")
            measured_at = pd.to_datetime(f"{date_str} {time_str}")

            # Standardize units
            value = float(row.get("Arithmetic Mean", 0))
            unit = row.get("Units of Measure", "ug/m3")
            standardized_value, standardized_unit = self._standardize_units(
                value, unit, "concentration"
            )

            record = ExposureRecord(
                subject_id=f"MONITOR_{row.get('Site Num', 'unknown')}",
                exposure_id=f"EPA_AQS_{row.get('Local Site Name', '')}_{measured_at.strftime('%Y%m%d')}",
                analyte_id=analyte_id,
                analyte_name=pollutant,
                measured_at=measured_at,
                measurement_window=timedelta(days=1),  # Daily averages
                value=standardized_value,
                unit=standardized_unit,
                latitude=float(row.get("Latitude", 0)),
                longitude=float(row.get("Longitude", 0)),
                location_type="outdoor",
                spatial_resolution=SpatialResolution.POINT,
                temporal_resolution=TemporalResolution.DAILY,
                data_source="EPA_AQS",
                measurement_method=row.get("Method Name", ""),
                exposure_type=ExposureType.AIR_QUALITY,
                measurement_quality=self._assess_aqs_quality(row),
            )
            records.append(record)

        return self._create_dataset(records, "EPA_AQS", [ExposureType.AIR_QUALITY])

    def standardize_nhanes_chemicals(self, raw_data: pd.DataFrame) -> ExposureDataset:
        """Standardize NHANES chemical biomarker data"""

        logger.info(f"Standardizing NHANES chemical data: {len(raw_data)} records")

        records = []
        for _, row in raw_data.iterrows():

            # Map chemical to standard ID
            chemical = row.get("Chemical", "")
            analyte_id = self._map_chemical_to_ontology_id(chemical)

            # Handle detection limits
            value = float(row.get("Result", 0))
            detection_limit = row.get("Detection Limit", None)
            if detection_limit:
                detection_limit = float(detection_limit)

            # Determine sample collection date (approximate from cycle)
            cycle = row.get("Cycle", "2017-2018")
            measured_at = self._estimate_collection_date(cycle)

            record = ExposureRecord(
                subject_id=f"NHANES_{row.get('SEQN', 'unknown')}",
                exposure_id=f"NHANES_{chemical}_{row.get('SEQN', 'unknown')}",
                analyte_id=analyte_id,
                analyte_name=chemical,
                measured_at=measured_at,
                measurement_window=timedelta(hours=1),  # Point measurement
                value=value,
                unit=row.get("Unit", "ng/mL"),
                detection_limit=detection_limit,
                sample_type=row.get("Sample Type", "serum"),
                fasting_status=row.get("Fasting", None),
                spatial_resolution=SpatialResolution.COUNTY,  # NHANES sampling
                temporal_resolution=TemporalResolution.INSTANTANEOUS,
                data_source="NHANES",
                collection_protocol=f"NHANES_{cycle}",
                lab_method=row.get("Lab Method", ""),
                exposure_type=ExposureType.CHEMICAL_BIOMARKER,
                measurement_quality=self._assess_nhanes_quality(row),
            )
            records.append(record)

        return self._create_dataset(
            records, "NHANES", [ExposureType.CHEMICAL_BIOMARKER]
        )

    def standardize_wearable_data(self, raw_data: pd.DataFrame) -> ExposureDataset:
        """Standardize wearable/lifestyle exposure data"""

        logger.info(f"Standardizing wearable data: {len(raw_data)} records")

        records = []
        for _, row in raw_data.iterrows():

            # Handle various wearable metrics
            metric = row.get("metric", "")
            value = float(row.get("value", 0))

            # Map metric to exposure concept
            analyte_id, analyte_name = self._map_wearable_metric(metric)

            record = ExposureRecord(
                subject_id=f"SUBJECT_{row.get('subject_id', 'unknown')}",
                exposure_id=f"WEARABLE_{metric}_{row.get('timestamp', '')}",
                analyte_id=analyte_id,
                analyte_name=analyte_name,
                measured_at=pd.to_datetime(row.get("timestamp")),
                measurement_window=timedelta(minutes=1),
                value=value,
                unit=row.get("unit", "count"),
                location_type=row.get("location_context", "unknown"),
                temporal_resolution=TemporalResolution.MINUTE,
                data_source=f"DEVICE_{row.get('device_id', 'unknown')}",
                exposure_type=ExposureType.LIFESTYLE_BEHAVIORAL,
                measurement_quality="good",
            )
            records.append(record)

        return self._create_dataset(
            records, "WEARABLE", [ExposureType.LIFESTYLE_BEHAVIORAL]
        )

    def _map_to_ontology_id(self, analyte_name: str, category: str) -> str:
        """Map analyte name to standard ontology ID"""
        mappings = self.ontology_mappings.get(category, {})
        return mappings.get(analyte_name, f"UNKNOWN:{analyte_name}")

    def _map_chemical_to_ontology_id(self, chemical: str) -> str:
        """Map chemical name to CHEBI ID across all categories"""
        for category, mappings in self.ontology_mappings.items():
            if chemical in mappings:
                return mappings[chemical]
        return f"UNKNOWN:{chemical}"

    def _map_wearable_metric(self, metric: str) -> tuple[str, str]:
        """Map wearable metric to exposure concept"""
        wearable_mappings = {
            "heart_rate": ("ExO:0000123", "Heart Rate"),
            "steps": ("ExO:0000124", "Physical Activity"),
            "sleep_duration": ("ExO:0000125", "Sleep Duration"),
            "stress_level": ("ExO:0000126", "Perceived Stress"),
        }
        return wearable_mappings.get(metric, (f"ExO:UNKNOWN:{metric}", metric))

    def _standardize_units(
        self, value: float, unit: str, unit_type: str
    ) -> tuple[float, str]:
        """Convert to standard UCUM units"""
        conversions = self.unit_conversions.get(unit_type, {})

        if unit in conversions:
            factor = conversions[unit]
            base_units = {"concentration": "ug/m3", "biomarker": "ng/mL"}
            return value * factor, base_units.get(unit_type, unit)

        return value, unit

    def _assess_aqs_quality(self, row: pd.Series) -> str:
        """Assess EPA AQS data quality"""
        # Simple quality assessment based on available QC flags
        null_count = row.isnull().sum()
        if null_count > 3:
            return "poor"
        elif row.get("Observation Count", 0) < 10:
            return "fair"
        else:
            return "good"

    def _assess_nhanes_quality(self, row: pd.Series) -> str:
        """Assess NHANES data quality"""
        # Check for below detection limit flags
        if row.get("Comment", "").lower().find("below") >= 0:
            return "fair"
        return "good"

    def _estimate_collection_date(self, cycle: str) -> datetime:
        """Estimate collection date from NHANES cycle"""
        # Simple mapping - in practice would use actual survey dates
        cycle_dates = {
            "2017-2018": datetime(2018, 1, 1),
            "2015-2016": datetime(2016, 1, 1),
            "2013-2014": datetime(2014, 1, 1),
        }
        return cycle_dates.get(cycle, datetime(2018, 1, 1))

    def _create_dataset(
        self,
        records: List[ExposureRecord],
        dataset_name: str,
        exposure_types: List[ExposureType],
    ) -> ExposureDataset:
        """Create standardized dataset from records"""

        if not records:
            logger.warning(f"No records provided for dataset {dataset_name}")
            return ExposureDataset(
                records=[],
                dataset_id=f"{dataset_name}_{datetime.now().strftime('%Y%m%d')}",
                dataset_name=dataset_name,
                exposure_types=exposure_types,
                start_date=datetime.now(),
                end_date=datetime.now(),
                temporal_resolution=TemporalResolution.DAILY,
                spatial_extent={},
                spatial_resolution=SpatialResolution.POINT,
                n_subjects=0,
            )

        # Calculate temporal coverage
        dates = [r.measured_at for r in records]
        start_date = min(dates)
        end_date = max(dates)

        # Calculate spatial extent
        lats = [r.latitude for r in records if r.latitude is not None]
        lons = [r.longitude for r in records if r.longitude is not None]

        spatial_extent = {}
        if lats and lons:
            spatial_extent = {
                "min_lat": min(lats),
                "max_lat": max(lats),
                "min_lon": min(lons),
                "max_lon": max(lons),
            }

        # Count unique subjects
        subjects = set(r.subject_id for r in records)

        # Assess data completeness
        total_fields = len(ExposureRecord.__dataclass_fields__)
        complete_records = sum(
            1
            for r in records
            if sum(1 for field, value in r.__dict__.items() if value is not None)
            >= total_fields * 0.8
        )
        completeness_score = complete_records / len(records) if records else 0.0

        return ExposureDataset(
            records=records,
            dataset_id=f"{dataset_name}_{datetime.now().strftime('%Y%m%d')}",
            dataset_name=dataset_name,
            exposure_types=exposure_types,
            start_date=start_date,
            end_date=end_date,
            temporal_resolution=records[0].temporal_resolution,
            spatial_extent=spatial_extent,
            spatial_resolution=records[0].spatial_resolution,
            n_subjects=len(subjects),
            completeness_score=completeness_score,
            data_sources=list(set(r.data_source for r in records)),
        )


class TemporalAligner:
    """Align exposure data with biomarker sampling times"""

    def __init__(self, default_window: timedelta = timedelta(days=30)):
        self.default_window = default_window

    def align_exposures_to_biomarkers(
        self, exposure_dataset: ExposureDataset, biomarker_times: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Align exposure measurements to biomarker sampling times

        Args:
            exposure_dataset: Standardized exposure data
            biomarker_times: DataFrame with columns [subject_id, sample_time]

        Returns:
            DataFrame with aligned exposure values for each biomarker sample
        """

        logger.info("Aligning exposure data to biomarker sampling times")

        aligned_data = []

        for _, biomarker_row in biomarker_times.iterrows():
            subject_id = biomarker_row["subject_id"]
            sample_time = pd.to_datetime(biomarker_row["sample_time"])

            # Find relevant exposure records for this subject
            subject_exposures = [
                r for r in exposure_dataset.records if r.subject_id == subject_id
            ]

            if not subject_exposures:
                continue

            # Calculate exposure windows based on exposure type
            aligned_exposures = {}

            for exposure_type in exposure_dataset.exposure_types:
                type_exposures = [
                    r for r in subject_exposures if r.exposure_type == exposure_type
                ]

                if not type_exposures:
                    continue

                # Define exposure window based on type
                if exposure_type == ExposureType.AIR_QUALITY:
                    window = timedelta(days=7)  # Weekly average
                elif exposure_type == ExposureType.CHEMICAL_BIOMARKER:
                    window = timedelta(days=1)  # Same day or closest
                elif exposure_type == ExposureType.LIFESTYLE_BEHAVIORAL:
                    window = timedelta(days=14)  # 2-week average
                else:
                    window = self.default_window

                # Find exposures within window
                window_start = sample_time - window
                window_end = sample_time

                relevant_exposures = [
                    r
                    for r in type_exposures
                    if window_start <= r.measured_at <= window_end
                ]

                if relevant_exposures:
                    # Calculate summary statistics for exposure window
                    for analyte_id in set(r.analyte_id for r in relevant_exposures):
                        analyte_exposures = [
                            r for r in relevant_exposures if r.analyte_id == analyte_id
                        ]

                        values = [r.value for r in analyte_exposures]
                        weights = [
                            1.0 / max(1, (sample_time - r.measured_at).days)
                            for r in analyte_exposures
                        ]  # Weight by recency

                        weighted_mean = np.average(values, weights=weights)

                        aligned_exposures[f"{analyte_id}_mean"] = weighted_mean
                        aligned_exposures[f"{analyte_id}_max"] = max(values)
                        aligned_exposures[f"{analyte_id}_n_measurements"] = len(values)

            if aligned_exposures:
                aligned_row = {
                    "subject_id": subject_id,
                    "sample_time": sample_time,
                    **aligned_exposures,
                }
                aligned_data.append(aligned_row)

        result_df = pd.DataFrame(aligned_data)
        logger.info(
            f"Aligned exposure data: {len(result_df)} biomarker samples with exposure data"
        )

        return result_df


def create_exposure_demo_data() -> Dict[str, pd.DataFrame]:
    """Create demonstration exposure data for testing"""

    # EPA AQS demo data
    epa_data = pd.DataFrame(
        {
            "Parameter Name": ["PM2.5", "NO2", "O3"] * 10,
            "Date Local": pd.date_range("2023-01-01", periods=30).strftime("%Y-%m-%d"),
            "Time Local": ["12:00"] * 30,
            "Arithmetic Mean": np.random.normal(15, 5, 30),  # PM2.5 levels
            "Units of Measure": ["ug/m3"] * 30,
            "Site Num": ["001", "002", "003"] * 10,
            "Local Site Name": ["Downtown", "Suburban", "Industrial"] * 10,
            "Latitude": [40.7, 40.8, 40.6] * 10,
            "Longitude": [-74.0, -74.1, -73.9] * 10,
            "Method Name": ["FRM"] * 30,
            "Observation Count": np.random.randint(20, 25, 30),
        }
    )

    # NHANES demo data
    nhanes_data = pd.DataFrame(
        {
            "SEQN": [f"{1000 + i}" for i in range(50)],
            "Chemical": ["PFOA", "PFOS", "Lead", "Mercury"] * 12 + ["PFOA", "PFOS"],
            "Result": np.random.lognormal(0, 1, 50),
            "Unit": ["ng/mL"] * 50,
            "Detection Limit": [0.1] * 50,
            "Sample Type": ["serum"] * 50,
            "Cycle": ["2017-2018"] * 50,
            "Lab Method": ["LC-MS/MS"] * 50,
            "Fasting": [True, False] * 25,
        }
    )

    # Wearable demo data
    wearable_data = pd.DataFrame(
        {
            "subject_id": [f"SUBJ_{i:03d}" for i in range(20)] * 50,
            "timestamp": pd.date_range("2023-01-01", periods=1000, freq="1H")[:1000],
            "metric": ["heart_rate", "steps", "sleep_duration"] * 333 + ["heart_rate"],
            "value": np.random.normal(70, 10, 1000),  # Heart rate example
            "unit": ["bpm", "count", "hours"] * 333 + ["bpm"],
            "device_id": ["DEVICE_001"] * 1000,
            "location_context": ["home", "work", "other"] * 333 + ["home"],
        }
    )

    return {"epa_aqs": epa_data, "nhanes": nhanes_data, "wearable": wearable_data}


def run_exposure_standards_demo():
    """Demonstrate exposure data standardization"""

    print("\n🌍 EXPOSURE DATA STANDARDIZATION DEMONSTRATION")
    print("=" * 60)

    # Create demo data
    print("📊 Generating demonstration exposure data...")
    demo_data = create_exposure_demo_data()

    # Initialize standardizer
    standardizer = ExposureStandardizer()

    # Standardize each data source
    print("\n🔄 Standardizing data from multiple sources...")

    # EPA AQS data
    epa_dataset = standardizer.standardize_epa_aqs(demo_data["epa_aqs"])
    print(f"   EPA AQS: {len(epa_dataset.records)} exposure records")
    print(
        f"   Temporal coverage: {epa_dataset.start_date.date()} to {epa_dataset.end_date.date()}"
    )
    print(f"   Spatial extent: {epa_dataset.spatial_extent}")

    # NHANES data
    nhanes_dataset = standardizer.standardize_nhanes_chemicals(demo_data["nhanes"])
    print(f"   NHANES: {len(nhanes_dataset.records)} biomarker records")
    print(f"   Subjects: {nhanes_dataset.n_subjects}")
    print(f"   Completeness: {nhanes_dataset.completeness_score:.2f}")

    # Wearable data
    wearable_dataset = standardizer.standardize_wearable_data(demo_data["wearable"])
    print(f"   Wearables: {len(wearable_dataset.records)} lifestyle records")
    print(f"   Temporal resolution: {wearable_dataset.temporal_resolution.value}")

    # Demonstrate temporal alignment
    print("\n⏰ Demonstrating temporal alignment...")
    biomarker_times = pd.DataFrame(
        {
            "subject_id": ["MONITOR_001", "NHANES_1000", "SUBJECT_001"] * 5,
            "sample_time": pd.date_range("2023-01-15", periods=15, freq="7D"),
        }
    )

    aligner = TemporalAligner()
    aligned_data = aligner.align_exposures_to_biomarkers(epa_dataset, biomarker_times)

    print(f"   Aligned {len(aligned_data)} biomarker samples with exposure data")
    if not aligned_data.empty:
        print(f"   Exposure features: {list(aligned_data.columns)[2:]}")

    # Show example standardized records
    print("\n📋 Example standardized exposure records:")

    if epa_dataset.records:
        example_record = epa_dataset.records[0]
        print("\n   EPA AQS Record:")
        print(
            f"      Analyte: {example_record.analyte_name} ({example_record.analyte_id})"
        )
        print(f"      Value: {example_record.value} {example_record.unit}")
        print(
            f"      Location: ({example_record.latitude:.2f}, {example_record.longitude:.2f})"
        )
        print(f"      Quality: {example_record.measurement_quality}")

    if nhanes_dataset.records:
        example_record = nhanes_dataset.records[0]
        print("\n   NHANES Record:")
        print(
            f"      Chemical: {example_record.analyte_name} ({example_record.analyte_id})"
        )
        print(f"      Concentration: {example_record.value} {example_record.unit}")
        print(f"      Sample type: {example_record.sample_type}")
        print(f"      Detection limit: {example_record.detection_limit}")

    print("\n✅ Exposure standardization demonstration complete!")
    print("\nKey capabilities demonstrated:")
    print("  • Multi-source data harmonization (EPA, NHANES, wearables)")
    print("  • Ontology mapping (CHEBI/ExO IDs)")
    print("  • UCUM unit standardization")
    print("  • Temporal alignment with biomarker sampling")
    print("  • Quality assessment and provenance tracking")

    return {
        "epa_dataset": epa_dataset,
        "nhanes_dataset": nhanes_dataset,
        "wearable_dataset": wearable_dataset,
        "aligned_data": aligned_data,
    }


if __name__ == "__main__":
    run_exposure_standards_demo()
