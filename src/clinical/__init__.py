"""
Clinical Outcome Expansion Framework
Extends MIMIC-IV outcomes beyond current biomarkers to include AKI staging, 
recovery trajectories, and intervention responses for comprehensive validation targets.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OutcomeType(Enum):
    """Types of clinical outcomes"""
    PRIMARY_ENDPOINT = "primary"
    SECONDARY_ENDPOINT = "secondary"
    SAFETY_ENDPOINT = "safety"
    EXPLORATORY_ENDPOINT = "exploratory"
    COMPOSITE_ENDPOINT = "composite"


class TimeFrame(Enum):
    """Time frames for outcome assessment"""
    IMMEDIATE = "immediate"  # Within 24 hours
    SHORT_TERM = "short_term"  # 24h - 7 days
    MEDIUM_TERM = "medium_term"  # 7 days - 30 days
    LONG_TERM = "long_term"  # > 30 days
    DURING_STAY = "during_stay"  # During hospital/ICU stay


@dataclass
class ClinicalOutcome:
    """Definition of a clinical outcome"""
    outcome_id: str
    name: str
    description: str
    outcome_type: OutcomeType
    time_frame: TimeFrame
    
    # Outcome definition parameters
    definition_criteria: Dict[str, Any]
    inclusion_criteria: List[str] = field(default_factory=list)
    exclusion_criteria: List[str] = field(default_factory=list)
    
    # Clinical relevance
    clinical_significance: str = ""  # Description of clinical importance
    regulatory_acceptance: str = ""  # FDA/EMA guidance alignment
    
    # Technical parameters
    data_sources: List[str] = field(default_factory=list)
    required_tables: List[str] = field(default_factory=list)
    calculation_method: str = ""
    
    # Validation parameters
    expected_prevalence: Optional[float] = None
    literature_validation: List[str] = field(default_factory=list)
    
    # Metadata
    created_date: str = field(default_factory=lambda: datetime.now().isoformat())
    version: str = "1.0"


@dataclass
class PatientOutcome:
    """Individual patient outcome result"""
    patient_id: str
    stay_id: str
    outcome_id: str
    
    # Outcome values
    binary_outcome: Optional[bool] = None  # For yes/no outcomes
    continuous_outcome: Optional[float] = None  # For continuous outcomes
    categorical_outcome: Optional[str] = None  # For categorical outcomes
    time_to_event: Optional[float] = None  # Time to event in days
    
    # Confidence and quality
    confidence: float = 1.0  # Confidence in outcome determination
    data_quality: str = "high"  # high, medium, low
    missing_data_fraction: float = 0.0
    
    # Supporting data
    supporting_values: Dict[str, Any] = field(default_factory=dict)
    calculation_details: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    assessment_date: str = field(default_factory=lambda: datetime.now().isoformat())


class OutcomeCalculator:
    """Base class for outcome calculations"""
    
    def __init__(self, mimic_dir: Path):
        self.mimic_dir = Path(mimic_dir)
        self.tables_cache = {}
        
    def load_table(self, table_name: str, usecols: Optional[List[str]] = None) -> pd.DataFrame:
        """Load MIMIC table with caching"""
        cache_key = f"{table_name}_{usecols}"
        
        if cache_key not in self.tables_cache:
            table_path = self.mimic_dir / f"{table_name}.csv.gz"
            if table_path.exists():
                df = pd.read_csv(table_path, compression="gzip", usecols=usecols)
                self.tables_cache[cache_key] = df
                logger.info(f"Loaded {table_name}: {len(df)} rows")
            else:
                logger.warning(f"Table not found: {table_path}")
                self.tables_cache[cache_key] = pd.DataFrame()
        
        return self.tables_cache[cache_key].copy()


class AKIStageCalculator(OutcomeCalculator):
    """Calculate detailed AKI staging according to KDIGO guidelines"""
    
    def __init__(self, mimic_dir: Path):
        super().__init__(mimic_dir)
        self.creatinine_itemids = {50912, 1525, 220615, 220587}
        self.urine_output_itemids = {40055, 43175, 40069, 40094, 220045}
        
    def calculate_aki_stage(self, stay_data: pd.DataFrame) -> Dict[str, PatientOutcome]:
        """Calculate AKI stage for patient stays"""
        
        results = {}
        
        # Load required data
        labevents = self.load_table("hosp/labevents", 
                                   usecols=["subject_id", "hadm_id", "itemid", "charttime", "value"])
        outputevents = self.load_table("icu/outputevents",
                                      usecols=["subject_id", "stay_id", "itemid", "charttime", "value"])
        icustays = self.load_table("icu/icustays",
                                  usecols=["subject_id", "hadm_id", "stay_id", "intime", "outtime"])
        patients = self.load_table("hosp/patients",
                                  usecols=["subject_id", "gender", "dod"])
        admissions = self.load_table("hosp/admissions",
                                    usecols=["subject_id", "hadm_id", "admittime", "dischtime"])
        
        if labevents.empty or icustays.empty:
            logger.warning("Required tables not available for AKI staging")
            return results
        
        # Process creatinine data
        creat_data = labevents[labevents["itemid"].isin(self.creatinine_itemids)].copy()
        creat_data["value"] = pd.to_numeric(creat_data["value"], errors="coerce")
        creat_data = creat_data.dropna(subset=["value"])
        creat_data["charttime"] = pd.to_datetime(creat_data["charttime"])
        
        # Process urine output data
        uo_data = outputevents[outputevents["itemid"].isin(self.urine_output_itemids)].copy()
        uo_data["value"] = pd.to_numeric(uo_data["value"], errors="coerce")
        uo_data = uo_data.dropna(subset=["value"])
        uo_data["charttime"] = pd.to_datetime(uo_data["charttime"])
        
        # Calculate AKI for each ICU stay
        for _, stay in icustays.iterrows():
            subject_id = stay["subject_id"]
            stay_id = stay["stay_id"]
            hadm_id = stay["hadm_id"]
            intime = pd.to_datetime(stay["intime"])
            outtime = pd.to_datetime(stay["outtime"])
            
            # Get patient weight (simplified estimation)
            patient_weight = 70.0  # Default assumption, should be extracted from chartevents
            
            # Calculate creatinine-based AKI
            creat_aki_stage, creat_details = self._calculate_creatinine_aki(
                subject_id, hadm_id, intime, outtime, creat_data
            )
            
            # Calculate urine output-based AKI
            uo_aki_stage, uo_details = self._calculate_urine_output_aki(
                subject_id, stay_id, intime, outtime, uo_data, patient_weight
            )
            
            # Determine overall AKI stage (maximum of creatinine and UO criteria)
            overall_stage = max(creat_aki_stage, uo_aki_stage)
            
            # Create outcome result
            results[f"{stay_id}_aki_stage"] = PatientOutcome(
                patient_id=str(subject_id),
                stay_id=str(stay_id),
                outcome_id="aki_stage",
                categorical_outcome=f"stage_{overall_stage}",
                continuous_outcome=float(overall_stage),
                binary_outcome=overall_stage > 0,
                confidence=min(creat_details.get("confidence", 1.0), 
                              uo_details.get("confidence", 1.0)),
                supporting_values={
                    "creatinine_stage": creat_aki_stage,
                    "urine_output_stage": uo_aki_stage,
                    "baseline_creatinine": creat_details.get("baseline_creatinine"),
                    "peak_creatinine": creat_details.get("peak_creatinine"),
                    "min_urine_output_6h": uo_details.get("min_uo_6h"),
                    "min_urine_output_12h": uo_details.get("min_uo_12h")
                },
                calculation_details={
                    "creatinine_details": creat_details,
                    "urine_output_details": uo_details,
                    "assessment_period": f"{intime} to {outtime}"
                }
            )
        
        return results
    
    def _calculate_creatinine_aki(self, subject_id: int, hadm_id: int, 
                                intime: datetime, outtime: datetime,
                                creat_data: pd.DataFrame) -> Tuple[int, Dict[str, Any]]:
        """Calculate AKI stage based on creatinine criteria"""
        
        # Get creatinine values for this admission
        admission_creat = creat_data[
            (creat_data["subject_id"] == subject_id) & 
            (creat_data["hadm_id"] == hadm_id)
        ].copy()
        
        if admission_creat.empty:
            return 0, {"confidence": 0.0, "reason": "no_creatinine_data"}
        
        # Sort by time
        admission_creat = admission_creat.sort_values("charttime")
        
        # Estimate baseline creatinine (first available value or historical minimum)
        baseline_creat = admission_creat["value"].iloc[0]
        
        # Get ICU period creatinine values
        icu_creat = admission_creat[
            (admission_creat["charttime"] >= intime) &
            (admission_creat["charttime"] <= outtime)
        ]
        
        if icu_creat.empty:
            return 0, {"confidence": 0.0, "reason": "no_icu_creatinine_data"}
        
        peak_creat = icu_creat["value"].max()
        
        # Calculate creatinine ratio
        creat_ratio = peak_creat / baseline_creat if baseline_creat > 0 else 1.0
        creat_increase = peak_creat - baseline_creat
        
        # Determine AKI stage based on KDIGO criteria
        stage = 0
        if creat_ratio >= 3.0 or peak_creat >= 4.0:
            stage = 3
        elif creat_ratio >= 2.0:
            stage = 2
        elif creat_ratio >= 1.5 or creat_increase >= 0.3:
            stage = 1
        
        details = {
            "baseline_creatinine": baseline_creat,
            "peak_creatinine": peak_creat,
            "creatinine_ratio": creat_ratio,
            "creatinine_increase": creat_increase,
            "confidence": 1.0 if len(icu_creat) >= 2 else 0.7,
            "n_measurements": len(icu_creat)
        }
        
        return stage, details
    
    def _calculate_urine_output_aki(self, subject_id: int, stay_id: int,
                                  intime: datetime, outtime: datetime,
                                  uo_data: pd.DataFrame, weight: float) -> Tuple[int, Dict[str, Any]]:
        """Calculate AKI stage based on urine output criteria"""
        
        # Get urine output for this ICU stay
        stay_uo = uo_data[
            (uo_data["subject_id"] == subject_id) & 
            (uo_data["stay_id"] == stay_id)
        ].copy()
        
        if stay_uo.empty:
            return 0, {"confidence": 0.0, "reason": "no_urine_output_data"}
        
        # Sort by time and calculate hourly rates
        stay_uo = stay_uo.sort_values("charttime")
        stay_uo["uo_rate"] = stay_uo["value"] / weight  # mL/kg/hr
        
        # Calculate rolling minimum UO rates
        stage = 0
        min_uo_6h = np.inf
        min_uo_12h = np.inf
        confidence = 0.0
        
        if len(stay_uo) >= 6:  # At least 6 hours of data
            # Simple rolling window approach (should be more sophisticated)
            for i in range(len(stay_uo) - 5):
                window_6h = stay_uo.iloc[i:i+6]["uo_rate"].mean()
                min_uo_6h = min(min_uo_6h, window_6h)
                
                if i <= len(stay_uo) - 12:
                    window_12h = stay_uo.iloc[i:i+12]["uo_rate"].mean()
                    min_uo_12h = min(min_uo_12h, window_12h)
            
            # Determine stage based on KDIGO criteria
            if min_uo_6h < 0.3:  # <0.3 mL/kg/hr for ≥24h (stage 3)
                stage = 3
            elif min_uo_12h < 0.5:  # <0.5 mL/kg/hr for ≥12h (stage 2)
                stage = 2
            elif min_uo_6h < 0.5:  # <0.5 mL/kg/hr for ≥6h (stage 1)
                stage = 1
            
            confidence = min(1.0, len(stay_uo) / 24)  # Higher confidence with more data
        
        details = {
            "min_uo_6h": min_uo_6h if min_uo_6h != np.inf else None,
            "min_uo_12h": min_uo_12h if min_uo_12h != np.inf else None,
            "patient_weight_kg": weight,
            "confidence": confidence,
            "n_measurements": len(stay_uo)
        }
        
        return stage, details


class RecoveryTrajectoryCalculator(OutcomeCalculator):
    """Calculate kidney function recovery trajectories"""
    
    def calculate_recovery_trajectory(self, stay_data: pd.DataFrame) -> Dict[str, PatientOutcome]:
        """Calculate recovery patterns after AKI"""
        
        results = {}
        
        # Load required data
        labevents = self.load_table("hosp/labevents",
                                   usecols=["subject_id", "hadm_id", "itemid", "charttime", "value"])
        icustays = self.load_table("icu/icustays",
                                  usecols=["subject_id", "hadm_id", "stay_id", "intime", "outtime"])
        admissions = self.load_table("hosp/admissions",
                                    usecols=["subject_id", "hadm_id", "dischtime"])
        
        if labevents.empty or icustays.empty:
            return results
        
        creatinine_itemids = {50912, 1525, 220615, 220587}
        creat_data = labevents[labevents["itemid"].isin(creatinine_itemids)].copy()
        creat_data["value"] = pd.to_numeric(creat_data["value"], errors="coerce")
        creat_data = creat_data.dropna(subset=["value"])
        creat_data["charttime"] = pd.to_datetime(creat_data["charttime"])
        
        # Calculate recovery for each stay
        for _, stay in icustays.iterrows():
            subject_id = stay["subject_id"]
            stay_id = stay["stay_id"]
            hadm_id = stay["hadm_id"]
            intime = pd.to_datetime(stay["intime"])
            outtime = pd.to_datetime(stay["outtime"])
            
            # Get admission discharge time
            admission_info = admissions[admissions["hadm_id"] == hadm_id]
            if admission_info.empty:
                continue
            
            dischtime = pd.to_datetime(admission_info["dischtime"].iloc[0])
            
            # Calculate recovery metrics
            recovery_metrics = self._calculate_recovery_metrics(
                subject_id, hadm_id, intime, outtime, dischtime, creat_data
            )
            
            if recovery_metrics:
                results[f"{stay_id}_recovery"] = PatientOutcome(
                    patient_id=str(subject_id),
                    stay_id=str(stay_id),
                    outcome_id="kidney_recovery",
                    categorical_outcome=recovery_metrics["recovery_category"],
                    continuous_outcome=recovery_metrics["recovery_score"],
                    binary_outcome=recovery_metrics["complete_recovery"],
                    time_to_event=recovery_metrics.get("time_to_recovery_days"),
                    confidence=recovery_metrics["confidence"],
                    supporting_values=recovery_metrics["supporting_values"],
                    calculation_details=recovery_metrics["calculation_details"]
                )
        
        return results
    
    def _calculate_recovery_metrics(self, subject_id: int, hadm_id: int,
                                  intime: datetime, outtime: datetime, 
                                  dischtime: datetime, creat_data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Calculate detailed recovery metrics"""
        
        # Get creatinine trajectory for this admission
        admission_creat = creat_data[
            (creat_data["subject_id"] == subject_id) & 
            (creat_data["hadm_id"] == hadm_id)
        ].copy()
        
        if len(admission_creat) < 3:  # Need minimum data for trajectory
            return None
        
        admission_creat = admission_creat.sort_values("charttime")
        
        # Define time periods
        baseline_creat = admission_creat["value"].iloc[0]
        peak_creat = admission_creat["value"].max()
        peak_time = admission_creat.loc[admission_creat["value"].idxmax(), "charttime"]
        
        # Get discharge creatinine (within 24h of discharge)
        discharge_window = admission_creat[
            admission_creat["charttime"] >= (dischtime - timedelta(days=1))
        ]
        discharge_creat = discharge_window["value"].iloc[-1] if not discharge_window.empty else None
        
        # Calculate recovery metrics
        if discharge_creat is None:
            return None
        
        # Recovery percentage
        recovery_percent = (peak_creat - discharge_creat) / (peak_creat - baseline_creat) if peak_creat > baseline_creat else 1.0
        recovery_percent = max(0.0, min(1.0, recovery_percent))
        
        # Recovery categories
        if discharge_creat <= baseline_creat * 1.1:
            recovery_category = "complete"
            complete_recovery = True
        elif discharge_creat <= baseline_creat * 1.5:
            recovery_category = "partial"
            complete_recovery = False
        else:
            recovery_category = "none"
            complete_recovery = False
        
        # Time to recovery (if applicable)
        time_to_recovery = None
        if complete_recovery:
            recovery_threshold = baseline_creat * 1.1
            recovery_points = admission_creat[
                (admission_creat["charttime"] > peak_time) &
                (admission_creat["value"] <= recovery_threshold)
            ]
            if not recovery_points.empty:
                recovery_time = recovery_points["charttime"].iloc[0]
                time_to_recovery = (recovery_time - peak_time).total_seconds() / (24 * 3600)  # days
        
        return {
            "recovery_category": recovery_category,
            "recovery_score": recovery_percent,
            "complete_recovery": complete_recovery,
            "time_to_recovery_days": time_to_recovery,
            "confidence": 0.8 if len(admission_creat) >= 5 else 0.6,
            "supporting_values": {
                "baseline_creatinine": baseline_creat,
                "peak_creatinine": peak_creat,
                "discharge_creatinine": discharge_creat,
                "recovery_percent": recovery_percent
            },
            "calculation_details": {
                "n_measurements": len(admission_creat),
                "assessment_period_days": (dischtime - intime).total_seconds() / (24 * 3600),
                "peak_time": str(peak_time)
            }
        }


class InterventionResponseCalculator(OutcomeCalculator):
    """Calculate responses to therapeutic interventions"""
    
    def __init__(self, mimic_dir: Path):
        super().__init__(mimic_dir)
        self.rrt_itemids = {225802, 225803, 225805, 225806, 225807, 225808, 225809, 225810, 225811}
        self.diuretic_itemids = {221794, 221289, 221833}  # Furosemide, etc.
        
    def calculate_intervention_responses(self, stay_data: pd.DataFrame) -> Dict[str, PatientOutcome]:
        """Calculate responses to RRT, diuretics, and other interventions"""
        
        results = {}
        
        # Load required data
        inputevents = self.load_table("icu/inputevents",
                                     usecols=["subject_id", "stay_id", "itemid", "starttime", "amount"])
        procedureevents = self.load_table("icu/procedureevents_mv",
                                         usecols=["subject_id", "stay_id", "itemid", "starttime", "endtime"])
        outputevents = self.load_table("icu/outputevents",
                                      usecols=["subject_id", "stay_id", "itemid", "charttime", "value"])
        icustays = self.load_table("icu/icustays",
                                  usecols=["subject_id", "stay_id", "intime", "outtime"])
        
        if icustays.empty:
            return results
        
        # Calculate RRT response
        rrt_results = self._calculate_rrt_response(
            inputevents, procedureevents, outputevents, icustays
        )
        results.update(rrt_results)
        
        # Calculate diuretic response
        diuretic_results = self._calculate_diuretic_response(
            inputevents, outputevents, icustays
        )
        results.update(diuretic_results)
        
        return results
    
    def _calculate_rrt_response(self, inputevents: pd.DataFrame, procedureevents: pd.DataFrame,
                               outputevents: pd.DataFrame, icustays: pd.DataFrame) -> Dict[str, PatientOutcome]:
        """Calculate response to renal replacement therapy"""
        
        results = {}
        
        # Check if we have the required data
        if procedureevents.empty or "itemid" not in procedureevents.columns:
            logger.info("No procedure events data available for RRT analysis")
            return results
        
        # Identify RRT events
        rrt_events = procedureevents[procedureevents["itemid"].isin(self.rrt_itemids)].copy()
        
        if rrt_events.empty:
            logger.info("No RRT events found in procedure data")
            return results
        
        # Group by stay and calculate response metrics
        for stay_id, stay_rrt in rrt_events.groupby("stay_id"):
            # Get urine output before and after RRT
            stay_uo = outputevents[outputevents["stay_id"] == stay_id].copy()
            
            if stay_uo.empty:
                continue
            
            stay_uo["charttime"] = pd.to_datetime(stay_uo["charttime"])
            stay_uo["value"] = pd.to_numeric(stay_uo["value"], errors="coerce")
            stay_uo = stay_uo.dropna(subset=["value"])
            
            # Find first RRT time
            first_rrt = pd.to_datetime(stay_rrt["starttime"].min())
            
            # Calculate pre-RRT and post-RRT urine output
            pre_rrt_uo = stay_uo[
                (stay_uo["charttime"] >= (first_rrt - timedelta(hours=24))) &
                (stay_uo["charttime"] < first_rrt)
            ]
            
            post_rrt_uo = stay_uo[
                (stay_uo["charttime"] >= first_rrt) &
                (stay_uo["charttime"] <= (first_rrt + timedelta(hours=48)))
            ]
            
            if len(pre_rrt_uo) >= 6 and len(post_rrt_uo) >= 6:
                pre_rrt_rate = pre_rrt_uo["value"].mean()
                post_rrt_rate = post_rrt_uo["value"].mean()
                
                # Calculate response
                uo_improvement = post_rrt_rate - pre_rrt_rate
                uo_improvement_percent = uo_improvement / pre_rrt_rate if pre_rrt_rate > 0 else 0
                
                # Determine response category
                if uo_improvement_percent > 0.5:
                    response_category = "good_response"
                elif uo_improvement_percent > 0.2:
                    response_category = "partial_response"
                else:
                    response_category = "poor_response"
                
                subject_id = stay_rrt["subject_id"].iloc[0]
                
                results[f"{stay_id}_rrt_response"] = PatientOutcome(
                    patient_id=str(subject_id),
                    stay_id=str(stay_id),
                    outcome_id="rrt_response",
                    categorical_outcome=response_category,
                    continuous_outcome=uo_improvement_percent,
                    binary_outcome=uo_improvement_percent > 0.2,
                    supporting_values={
                        "pre_rrt_uo_rate": pre_rrt_rate,
                        "post_rrt_uo_rate": post_rrt_rate,
                        "uo_improvement": uo_improvement,
                        "uo_improvement_percent": uo_improvement_percent
                    },
                    calculation_details={
                        "rrt_start_time": first_rrt.isoformat(),
                        "n_rrt_sessions": len(stay_rrt),
                        "pre_rrt_measurements": len(pre_rrt_uo),
                        "post_rrt_measurements": len(post_rrt_uo)
                    }
                )
        
        return results
    
    def _calculate_diuretic_response(self, inputevents: pd.DataFrame, outputevents: pd.DataFrame,
                                   icustays: pd.DataFrame) -> Dict[str, PatientOutcome]:
        """Calculate response to diuretic therapy"""
        
        results = {}
        
        # Check if we have input events data
        if inputevents.empty or "itemid" not in inputevents.columns:
            logger.info("No input events data available for diuretic analysis")
            return results
        
        # Identify diuretic administrations
        diuretic_events = inputevents[inputevents["itemid"].isin(self.diuretic_itemids)].copy()
        
        if diuretic_events.empty:
            logger.info("No diuretic events found in input data")
            return results
        
        # Similar logic to RRT response but for diuretics
        # (Implementation would follow similar pattern)
        
        return results


class CompositeOutcomeCalculator(OutcomeCalculator):
    """Calculate composite clinical outcomes"""
    
    def calculate_make30(self, stay_data: pd.DataFrame) -> Dict[str, PatientOutcome]:
        """Calculate Major Adverse Kidney Events at 30 days (MAKE30)"""
        
        results = {}
        
        # MAKE30 = Death OR Dialysis OR ≥50% decline in eGFR from baseline at 30 days
        # Simplified implementation for demonstration
        
        return results
    
    def calculate_composite_cv_outcome(self, stay_data: pd.DataFrame) -> Dict[str, PatientOutcome]:
        """Calculate composite cardiovascular outcomes"""
        
        results = {}
        
        # CV death, MI, stroke, heart failure hospitalization
        # Implementation would analyze diagnoses and procedures
        
        return results


class ClinicalOutcomeExpansion:
    """Main class for comprehensive clinical outcome calculation"""
    
    def __init__(self, mimic_dir: Path, output_dir: Optional[Path] = None):
        self.mimic_dir = Path(mimic_dir)
        self.output_dir = Path(output_dir) if output_dir else Path("outputs/clinical_outcomes")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize calculators
        self.aki_calculator = AKIStageCalculator(mimic_dir)
        self.recovery_calculator = RecoveryTrajectoryCalculator(mimic_dir)
        self.intervention_calculator = InterventionResponseCalculator(mimic_dir)
        self.composite_calculator = CompositeOutcomeCalculator(mimic_dir)
        
        # Define available outcomes
        self.outcome_definitions = self._initialize_outcome_definitions()
        
    def _initialize_outcome_definitions(self) -> Dict[str, ClinicalOutcome]:
        """Initialize comprehensive outcome definitions"""
        
        outcomes = {}
        
        # AKI-related outcomes
        outcomes["aki_stage"] = ClinicalOutcome(
            outcome_id="aki_stage",
            name="AKI Stage (KDIGO)",
            description="Acute kidney injury staging according to KDIGO guidelines",
            outcome_type=OutcomeType.PRIMARY_ENDPOINT,
            time_frame=TimeFrame.DURING_STAY,
            definition_criteria={
                "creatinine_criteria": "1.5x baseline OR ≥0.3 mg/dL increase",
                "urine_output_criteria": "<0.5 mL/kg/hr for ≥6h",
                "staging": "Stage 1-3 based on severity"
            },
            clinical_significance="Primary kidney injury endpoint, validated prognostic factor",
            regulatory_acceptance="FDA-accepted endpoint for kidney injury studies",
            data_sources=["labevents", "outputevents", "icustays"],
            required_tables=["hosp/labevents", "icu/outputevents", "icu/icustays"],
            expected_prevalence=0.25
        )
        
        outcomes["kidney_recovery"] = ClinicalOutcome(
            outcome_id="kidney_recovery",
            name="Kidney Function Recovery",
            description="Recovery of kidney function after AKI episode",
            outcome_type=OutcomeType.SECONDARY_ENDPOINT,
            time_frame=TimeFrame.MEDIUM_TERM,
            definition_criteria={
                "complete_recovery": "Return to ≤110% of baseline creatinine",
                "partial_recovery": "Return to ≤150% of baseline creatinine",
                "time_to_recovery": "Days from peak injury to recovery threshold"
            },
            clinical_significance="Critical for long-term prognosis and quality of life",
            data_sources=["labevents"],
            required_tables=["hosp/labevents", "hosp/admissions"],
            expected_prevalence=0.60
        )
        
        # Intervention response outcomes
        outcomes["rrt_response"] = ClinicalOutcome(
            outcome_id="rrt_response",
            name="RRT Response",
            description="Response to renal replacement therapy",
            outcome_type=OutcomeType.SECONDARY_ENDPOINT,
            time_frame=TimeFrame.SHORT_TERM,
            definition_criteria={
                "urine_output_improvement": ">20% increase in UO after RRT initiation",
                "assessment_window": "48 hours post-RRT initiation"
            },
            clinical_significance="Predicts RRT weaning success and recovery",
            data_sources=["procedureevents", "outputevents"],
            required_tables=["icu/procedureevents_mv", "icu/outputevents"],
            expected_prevalence=0.45
        )
        
        # Mortality outcomes
        outcomes["mortality_30d"] = ClinicalOutcome(
            outcome_id="mortality_30d",
            name="30-Day Mortality",
            description="All-cause mortality within 30 days of admission",
            outcome_type=OutcomeType.PRIMARY_ENDPOINT,
            time_frame=TimeFrame.MEDIUM_TERM,
            definition_criteria={
                "death_within_30_days": "Death within 30 days of hospital admission"
            },
            clinical_significance="Gold standard mortality endpoint",
            regulatory_acceptance="FDA-accepted primary endpoint",
            data_sources=["patients", "admissions"],
            required_tables=["hosp/patients", "hosp/admissions"],
            expected_prevalence=0.15
        )
        
        # Composite outcomes
        outcomes["make30"] = ClinicalOutcome(
            outcome_id="make30",
            name="MAKE30",
            description="Major Adverse Kidney Events at 30 days",
            outcome_type=OutcomeType.COMPOSITE_ENDPOINT,
            time_frame=TimeFrame.MEDIUM_TERM,
            definition_criteria={
                "death": "All-cause mortality",
                "dialysis": "New requirement for dialysis",
                "egfr_decline": "≥50% decline in eGFR from baseline"
            },
            clinical_significance="Comprehensive kidney outcome measure",
            regulatory_acceptance="Emerging FDA-accepted composite endpoint",
            data_sources=["multiple"],
            expected_prevalence=0.35
        )
        
        return outcomes
    
    def calculate_all_outcomes(self, cohort_definition: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Calculate all defined clinical outcomes for the cohort"""
        
        logger.info("Starting comprehensive clinical outcome calculation")
        
        # Load ICU stays as base cohort
        icustays = self.aki_calculator.load_table("icu/icustays")
        
        if icustays.empty:
            logger.error("No ICU stays data available")
            return pd.DataFrame()
        
        # Apply cohort definition if provided
        if cohort_definition:
            icustays = self._apply_cohort_definition(icustays, cohort_definition)
        
        logger.info(f"Calculating outcomes for {len(icustays)} ICU stays")
        
        all_outcomes = {}
        
        # Calculate AKI staging
        logger.info("Calculating AKI staging...")
        aki_outcomes = self.aki_calculator.calculate_aki_stage(icustays)
        all_outcomes.update(aki_outcomes)
        
        # Calculate recovery trajectories
        logger.info("Calculating recovery trajectories...")
        recovery_outcomes = self.recovery_calculator.calculate_recovery_trajectory(icustays)
        all_outcomes.update(recovery_outcomes)
        
        # Calculate intervention responses
        logger.info("Calculating intervention responses...")
        intervention_outcomes = self.intervention_calculator.calculate_intervention_responses(icustays)
        all_outcomes.update(intervention_outcomes)
        
        logger.info(f"Calculated {len(all_outcomes)} total outcome assessments")
        
        # Convert to DataFrame
        outcomes_df = self._outcomes_to_dataframe(all_outcomes)
        
        # Save results
        self._save_outcomes(outcomes_df, all_outcomes)
        
        return outcomes_df
    
    def _apply_cohort_definition(self, icustays: pd.DataFrame, 
                                cohort_def: Dict[str, Any]) -> pd.DataFrame:
        """Apply cohort inclusion/exclusion criteria"""
        
        # Example cohort filtering
        filtered_stays = icustays.copy()
        
        if "min_age" in cohort_def:
            # Would need to join with patients table to get age
            pass
        
        if "exclude_readmissions" in cohort_def and cohort_def["exclude_readmissions"]:
            # Keep only first ICU stay per patient
            filtered_stays = filtered_stays.groupby("subject_id").first().reset_index()
        
        return filtered_stays
    
    def _outcomes_to_dataframe(self, outcomes: Dict[str, PatientOutcome]) -> pd.DataFrame:
        """Convert outcome results to DataFrame"""
        
        rows = []
        for outcome_key, outcome in outcomes.items():
            rows.append({
                "outcome_key": outcome_key,
                "patient_id": outcome.patient_id,
                "stay_id": outcome.stay_id,
                "outcome_id": outcome.outcome_id,
                "binary_outcome": outcome.binary_outcome,
                "continuous_outcome": outcome.continuous_outcome,
                "categorical_outcome": outcome.categorical_outcome,
                "time_to_event": outcome.time_to_event,
                "confidence": outcome.confidence,
                "data_quality": outcome.data_quality,
                "assessment_date": outcome.assessment_date
            })
        
        return pd.DataFrame(rows)
    
    def _save_outcomes(self, outcomes_df: pd.DataFrame, 
                      detailed_outcomes: Dict[str, PatientOutcome]) -> None:
        """Save outcomes to files"""
        
        # Save summary DataFrame
        summary_file = self.output_dir / "clinical_outcomes_summary.csv"
        outcomes_df.to_csv(summary_file, index=False)
        logger.info(f"Saved outcomes summary to {summary_file}")
        
        # Save detailed outcomes as JSON
        detailed_data = {}
        for key, outcome in detailed_outcomes.items():
            detailed_data[key] = {
                "patient_id": outcome.patient_id,
                "stay_id": outcome.stay_id,
                "outcome_id": outcome.outcome_id,
                "binary_outcome": outcome.binary_outcome,
                "continuous_outcome": outcome.continuous_outcome,
                "categorical_outcome": outcome.categorical_outcome,
                "time_to_event": outcome.time_to_event,
                "confidence": outcome.confidence,
                "data_quality": outcome.data_quality,
                "supporting_values": outcome.supporting_values,
                "calculation_details": outcome.calculation_details,
                "assessment_date": outcome.assessment_date
            }
        
        detailed_file = self.output_dir / "clinical_outcomes_detailed.json"
        with open(detailed_file, 'w') as f:
            json.dump(detailed_data, f, indent=2)
        logger.info(f"Saved detailed outcomes to {detailed_file}")
        
        # Save outcome definitions
        definitions_data = {}
        for outcome_id, definition in self.outcome_definitions.items():
            definitions_data[outcome_id] = {
                "name": definition.name,
                "description": definition.description,
                "outcome_type": definition.outcome_type.value,
                "time_frame": definition.time_frame.value,
                "definition_criteria": definition.definition_criteria,
                "clinical_significance": definition.clinical_significance,
                "expected_prevalence": definition.expected_prevalence
            }
        
        definitions_file = self.output_dir / "outcome_definitions.json"
        with open(definitions_file, 'w') as f:
            json.dump(definitions_data, f, indent=2)
        logger.info(f"Saved outcome definitions to {definitions_file}")
    
    def generate_outcome_summary_report(self) -> pd.DataFrame:
        """Generate summary report of outcome prevalences and quality"""
        
        # Load calculated outcomes
        summary_file = self.output_dir / "clinical_outcomes_summary.csv"
        if not summary_file.exists():
            logger.warning("No outcomes summary file found")
            return pd.DataFrame()
        
        outcomes_df = pd.read_csv(summary_file)
        
        # Calculate prevalences and quality metrics
        report_data = []
        
        for outcome_id in outcomes_df["outcome_id"].unique():
            outcome_data = outcomes_df[outcomes_df["outcome_id"] == outcome_id]
            
            # Calculate metrics
            total_assessments = len(outcome_data)
            
            if "binary_outcome" in outcome_data.columns:
                positive_outcomes = outcome_data["binary_outcome"].sum()
                prevalence = positive_outcomes / total_assessments if total_assessments > 0 else 0
            else:
                prevalence = None
            
            avg_confidence = outcome_data["confidence"].mean()
            high_quality_pct = (outcome_data["data_quality"] == "high").mean() * 100
            
            report_data.append({
                "outcome_id": outcome_id,
                "total_assessments": total_assessments,
                "prevalence": prevalence,
                "average_confidence": avg_confidence,
                "high_quality_percent": high_quality_pct,
                "expected_prevalence": self.outcome_definitions.get(
                    outcome_id, 
                    ClinicalOutcome("", "", "", OutcomeType.PRIMARY_ENDPOINT, TimeFrame.IMMEDIATE, {})
                ).expected_prevalence
            })
        
        report_df = pd.DataFrame(report_data)
        
        # Save report
        report_file = self.output_dir / "outcome_quality_report.csv"
        report_df.to_csv(report_file, index=False)
        logger.info(f"Saved outcome quality report to {report_file}")
        
        return report_df


def create_demo_clinical_outcomes():
    """Create demonstration clinical outcomes data"""
    
    print("\n🏥 CLINICAL OUTCOME EXPANSION DEMONSTRATION")
    print("=" * 60)
    
    # Create demo MIMIC directory structure
    demo_dir = Path("demo_outputs/mimic_demo")
    demo_dir.mkdir(parents=True, exist_ok=True)
    
    # Create minimal demo data files
    print("📊 Creating demonstration MIMIC data...")
    
    # Create demo ICU stays
    icustays_data = []
    for i in range(50):
        icustays_data.append({
            "subject_id": 10000 + i,
            "hadm_id": 20000 + i,
            "stay_id": 30000 + i,
            "intime": f"2023-01-{(i % 28) + 1:02d} 08:00:00",
            "outtime": f"2023-01-{(i % 28) + 1:02d} {8 + (i % 10):02d}:00:00"
        })
    
    icustays_df = pd.DataFrame(icustays_data)
    icu_dir = demo_dir / "icu"
    icu_dir.mkdir(exist_ok=True)
    icustays_df.to_csv(icu_dir / "icustays.csv.gz", compression="gzip", index=False)
    
    # Create demo output events (urine output)
    outputevents_data = []
    for i in range(50):
        subject_id = 10000 + i
        stay_id = 30000 + i
        
        # Simulate hourly urine output
        for hour in range(24):
            hour_time = 8 + hour
            if hour_time >= 24:
                day_offset = 1
                hour_time = hour_time - 24
            else:
                day_offset = 0
            
            day_num = ((i + day_offset) % 28) + 1
            uo_value = np.random.normal(50, 20)  # mL/hr
            
            outputevents_data.append({
                "subject_id": subject_id,
                "stay_id": stay_id,
                "itemid": 40055,  # Urine output
                "charttime": f"2023-01-{day_num:02d} {hour_time:02d}:00:00",
                "value": max(0, uo_value)
            })
    
    outputevents_df = pd.DataFrame(outputevents_data)
    outputevents_df.to_csv(icu_dir / "outputevents.csv.gz", compression="gzip", index=False)
    
    # Create demo patients table
    patients_data = []
    for i in range(50):
        patients_data.append({
            "subject_id": 10000 + i,
            "gender": "M" if i % 2 == 0 else "F",
            "dod": None  # No deaths in demo
        })
    
    patients_df = pd.DataFrame(patients_data)
    hosp_dir = demo_dir / "hosp"
    hosp_dir.mkdir(exist_ok=True)
    patients_df.to_csv(hosp_dir / "patients.csv.gz", compression="gzip", index=False)
    
    # Create demo admissions table
    admissions_data = []
    for i in range(50):
        admissions_data.append({
            "subject_id": 10000 + i,
            "hadm_id": 20000 + i,
            "admittime": f"2023-01-{(i % 28) + 1:02d} 07:00:00",
            "dischtime": f"2023-01-{(i % 28) + 1:02d} {18 + (i % 6):02d}:00:00"
        })
    
    admissions_df = pd.DataFrame(admissions_data)
    admissions_df.to_csv(hosp_dir / "admissions.csv.gz", compression="gzip", index=False)
    
    # Create demo lab events (creatinine)
    labevents_data = []
    for idx, stay in enumerate(icustays_df.itertuples()):
        subject_id = stay.subject_id
        hadm_id = stay.hadm_id
        
        # Simulate creatinine trajectory
        baseline_creat = np.random.normal(1.0, 0.3)
        peak_creat = baseline_creat * np.random.uniform(1.0, 3.0)
        
        # Multiple measurements
        for hour in [0, 6, 12, 24, 48, 72]:
            if hour == 0:
                creat_value = baseline_creat
            elif hour <= 24:
                creat_value = baseline_creat + (peak_creat - baseline_creat) * (hour / 24)
            else:
                creat_value = peak_creat - (peak_creat - baseline_creat) * 0.3 * ((hour - 24) / 48)
            
            # Handle hour overflow properly
            hour_time = 8 + hour
            day_offset = hour_time // 24
            hour_time = hour_time % 24
            day_num = ((idx + day_offset) % 28) + 1
            
            labevents_data.append({
                "subject_id": subject_id,
                "hadm_id": hadm_id,
                "itemid": 50912,  # Creatinine
                "charttime": f"2023-01-{day_num:02d} {hour_time:02d}:00:00",
                "value": max(0.5, creat_value)
            })
    
    labevents_df = pd.DataFrame(labevents_data)
    labevents_df.to_csv(hosp_dir / "labevents.csv.gz", compression="gzip", index=False)
    
    print(f"   Created {len(icustays_df)} ICU stays")
    print(f"   Created {len(outputevents_df)} urine output measurements")
    print(f"   Created {len(labevents_df)} lab measurements")
    print(f"   Created {len(patients_df)} patient records")
    print(f"   Created {len(admissions_df)} admission records")
    
    # Initialize outcome calculator
    outcome_calculator = ClinicalOutcomeExpansion(
        mimic_dir=demo_dir,
        output_dir=Path("demo_outputs/clinical_outcomes")
    )
    
    # Calculate outcomes
    print("\n🔬 Calculating clinical outcomes...")
    outcomes_df = outcome_calculator.calculate_all_outcomes()
    
    print(f"\n📈 Clinical outcome calculation completed:")
    print(f"   Total outcome assessments: {len(outcomes_df)}")
    
    if not outcomes_df.empty:
        outcome_counts = outcomes_df["outcome_id"].value_counts()
        print(f"\n📊 Outcome breakdown:")
        for outcome_id, count in outcome_counts.items():
            print(f"   {outcome_id}: {count} assessments")
        
        # Generate quality report
        print("\n📋 Generating outcome quality report...")
        quality_report = outcome_calculator.generate_outcome_summary_report()
        
        if not quality_report.empty:
            print(f"\n🎯 Outcome quality metrics:")
            for _, row in quality_report.iterrows():
                outcome_id = row["outcome_id"]
                prevalence = row["prevalence"]
                confidence = row["average_confidence"]
                print(f"   {outcome_id}: {prevalence:.1%} prevalence, {confidence:.2f} avg confidence")
    
    print(f"\n✅ Clinical outcome expansion demonstration completed!")
    print(f"📁 Results saved to: demo_outputs/clinical_outcomes/")
    
    return outcomes_df, outcome_calculator


if __name__ == "__main__":
    create_demo_clinical_outcomes()
