"""
Inference pipeline for microplastic hotspot prediction and advisory generation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
import json
from datetime import datetime, timedelta
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PredictionEngine:
    """Engine for making microplastic hotspot predictions"""
    
    def __init__(self, model_dir: str = "models/"):
        self.model_dir = Path(model_dir)
        self.classifier = None
        self.feature_engineer = None
        self.is_initialized = False
        
    def initialize(self):
        """Initialize the prediction engine with trained models"""
        try:
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
            
            from models.train_classifier import MicroplasticClassifier
            from features.engineer_features import FeatureEngineerer
            
            # Load trained classifier
            self.classifier = MicroplasticClassifier()
            self.classifier.load_models(str(self.model_dir))
            
            # Initialize feature engineer
            self.feature_engineer = FeatureEngineerer()
            
            self.is_initialized = True
            logger.info("Prediction engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize prediction engine: {e}")
            raise
    
    def predict_hotspot_detailed(self, 
                       latitude: float, 
                       longitude: float, 
                       date: str,
                       ocean: str = "Unknown",
                       region: str = "Unknown") -> Dict[str, Any]:
        """
        Predict microplastic hotspot probability for a specific location and time
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate  
            date: Date string (YYYY-MM-DD format)
            ocean: Ocean name
            region: Region name
            
        Returns:
            Dictionary with prediction results and confidence scores
        """
        if not self.is_initialized:
            self.initialize()
        
        # Create input DataFrame
        input_data = pd.DataFrame({
            'Latitude': [latitude],
            'Longitude': [longitude],
            'Date': [pd.to_datetime(date)],
            'Oceans': [ocean],
            'Regions': [region]
        })
        
        # Engineer features
        try:
            features_df = self.feature_engineer.create_all_features(input_data)
            
            # Prepare features for prediction
            # For inference, we don't have a target column, so we prepare just the features
            numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
            X_inference = features_df[numeric_cols].copy()
            
            # Handle missing values
            X_inference = X_inference.fillna(X_inference.median())
            
            # Handle infinite values
            X_inference = X_inference.replace([np.inf, -np.inf], np.nan)
            X_inference = X_inference.fillna(X_inference.median())
            
            # Get feature names that match training data
            feature_names = self.classifier.feature_names
            
            # Ensure we have the same features as training (fill missing with 0)
            for feature in feature_names:
                if feature not in X_inference.columns:
                    X_inference[feature] = 0
            
            # Select only the features used in training
            X_inference = X_inference[feature_names]
            
            X = X_inference.values
            
            # Make predictions
            predictions = self.classifier.predict(X, return_probabilities=True)
            
            # Extract probabilities and predictions
            prob = predictions.get('probabilities', [0])[0]
            pred = predictions.get('predictions', [0])[0]
            
            # Format results
            result = {
                'location': {
                    'latitude': latitude,
                    'longitude': longitude,
                    'ocean': ocean,
                    'region': region
                },
                'prediction_date': date,
                'prediction': {
                    'is_hotspot': bool(pred),
                    'hotspot_probability': float(prob),
                    'confidence': self._calculate_confidence(prob)
                },
                'confidence_level': self._get_confidence_level(prob)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return self._create_error_result(latitude, longitude, date, str(e))
    
    def predict_hotspot(self, latitude: float, longitude: float) -> float:
        """
        Simple prediction method for hotspot probability (backwards compatibility)
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            
        Returns:
            Hotspot probability as float
        """
        try:
            # Use current date
            current_date = datetime.now().strftime('%Y-%m-%d')
            
            # Make full prediction
            result = self.predict_hotspot_detailed(
                latitude=latitude,
                longitude=longitude, 
                date=current_date,
                ocean="Unknown",
                region="Unknown"
            )
            
            # Extract probability
            if result.get('prediction'):
                return result['prediction']['hotspot_probability']
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Simple prediction failed: {e}")
            return 0.0
    
    def batch_predict(self, locations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Make batch predictions for multiple locations
        
        Args:
            locations: List of location dictionaries with keys: 
                      'latitude', 'longitude', 'date', 'ocean', 'region'
                      
        Returns:
            List of prediction results
        """
        results = []
        
        for i, location in enumerate(locations):
            try:
                result = self.predict_hotspot(
                    latitude=location['latitude'],
                    longitude=location['longitude'], 
                    date=location['date'],
                    ocean=location.get('ocean', 'Unknown'),
                    region=location.get('region', 'Unknown')
                )
                result['batch_index'] = i
                results.append(result)
                
            except Exception as e:
                logger.error(f"Batch prediction failed for location {i}: {e}")
                error_result = self._create_error_result(
                    location['latitude'], location['longitude'], 
                    location['date'], str(e)
                )
                error_result['batch_index'] = i
                results.append(error_result)
        
        return results
    
    def _calculate_confidence(self, probability: float) -> str:
        """Calculate confidence level based on probability"""
        if probability <= 0.3 or probability >= 0.7:
            return "High"
        elif probability <= 0.4 or probability >= 0.6:
            return "Medium"
        else:
            return "Low"
    
    def _get_confidence_level(self, probability: float) -> str:
        """Get overall confidence level for advisory generation"""
        if probability <= 0.2:
            return "Very Low Risk"
        elif probability <= 0.4:
            return "Low Risk"
        elif probability <= 0.6:
            return "Moderate Risk"
        elif probability <= 0.8:
            return "High Risk"
        else:
            return "Very High Risk"
    
    def _create_error_result(self, lat: float, lon: float, date: str, error: str) -> Dict[str, Any]:
        """Create error result structure"""
        return {
            'location': {
                'latitude': lat,
                'longitude': lon,
                'ocean': 'Unknown',
                'region': 'Unknown'
            },
            'prediction_date': date,
            'error': error,
            'predictions': {},
            'ensemble_result': None,
            'confidence_level': 'Error'
        }


class AdvisoryGenerator:
    """Generate human-readable ecological risk advisories using LLM integration"""
    
    def __init__(self, llm_config: Optional[Dict] = None):
        self.llm_config = llm_config or self._get_default_llm_config()
        self.templates = self._load_advisory_templates()
        
    def _get_default_llm_config(self) -> Dict:
        """Get default LLM configuration"""
        return {
            'provider': 'openai',  # or 'local', 'anthropic', etc.
            'model': 'gpt-3.5-turbo',
            'temperature': 0.3,
            'max_tokens': 500,
            'api_key': None  # Should be set via environment variable
        }
    
    def _load_advisory_templates(self) -> Dict[str, str]:
        """Load advisory templates for different risk levels and stakeholders"""
        return {
            'high_risk': {
                'environmental_agency': """
URGENT ENVIRONMENTAL ALERT - High Microplastic Contamination Risk

Location: {latitude:.4f}°, {longitude:.4f}° ({ocean}, {region})
Risk Assessment Date: {date}
Confidence Level: {confidence}
Hotspot Probability: {probability:.1%}

IMMEDIATE ACTIONS REQUIRED:
• Deploy monitoring equipment to confirm contamination levels
• Restrict fishing activities in the immediate area
• Initiate water quality testing protocols
• Alert marine protected area authorities if applicable

ECOLOGICAL IMPLICATIONS:
• High risk of microplastic ingestion by marine life
• Potential bioaccumulation in food web
• Threat to endangered species habitats
• Risk of contamination spread to adjacent areas

RECOMMENDED MONITORING PERIOD: 30-60 days with weekly assessments
""",
                'fisheries': """
FISHERIES ADVISORY - Elevated Microplastic Risk Zone

Coordinates: {latitude:.4f}°, {longitude:.4f}°
Water Body: {ocean} Ocean, {region}
Advisory Date: {date}
Risk Level: HIGH ({probability:.1%} probability)

FISHING RECOMMENDATIONS:
• AVOID fishing in this area for the next 4-6 weeks
• If fishing is essential, increase catch inspection protocols
• Do not consume fish caught from this zone without testing
• Report any unusual fish behavior or appearance

ECONOMIC CONSIDERATIONS:
• Temporary fishing restrictions may be necessary
• Consider alternative fishing grounds
• Monitor market advisories for seafood from this region
• Maintain detailed catch logs for traceability

Contact local fisheries management for updated guidelines.
""",
                'public': """
MARINE SAFETY ADVISORY

Location: {ocean} Ocean near {region}
Date: {date}
Contamination Risk: HIGH

What this means:
Marine areas near coordinates {latitude:.4f}°, {longitude:.4f}° show high risk 
of microplastic contamination ({probability:.1%} probability).

Precautions for the public:
• Avoid swimming in the immediate area
• Do not collect seafood for personal consumption
• Keep pets away from shoreline if applicable
• Report oil sheens or unusual debris to authorities

This is a predictive alert based on environmental modeling. 
Local authorities are being notified for verification.

For updates: [Contact Information]
"""
            },
            'moderate_risk': {
                'environmental_agency': """
ENVIRONMENTAL MONITORING NOTICE - Moderate Microplastic Risk

Location: {latitude:.4f}°, {longitude:.4f}° ({ocean}, {region})
Assessment Date: {date}
Confidence Level: {confidence}
Hotspot Probability: {probability:.1%}

RECOMMENDED ACTIONS:
• Increase monitoring frequency in this area
• Schedule routine water quality assessments
• Monitor marine life behavior patterns
• Coordinate with research institutions for data collection

ENVIRONMENTAL CONTEXT:
• Moderate risk of microplastic accumulation
• Standard precautionary measures advised
• Suitable for research and monitoring activities
• Continue regular ecosystem health assessments

NEXT REVIEW: 90 days or as conditions change
""",
                'fisheries': """
FISHERIES NOTICE - Moderate Microplastic Alert

Location: {latitude:.4f}°, {longitude:.4f}° 
Water Body: {ocean}, {region}
Date: {date}
Risk Level: MODERATE ({probability:.1%} probability)

OPERATIONAL GUIDANCE:
• Normal fishing operations may continue
• Implement enhanced catch inspection protocols
• Monitor fish health and behavior closely
• Report any anomalies to fisheries management

QUALITY ASSURANCE:
• Follow standard seafood safety procedures
• Maintain catch documentation
• Consider voluntary testing for high-value catches
• Stay informed of area status changes

This is a precautionary advisory based on predictive modeling.
""",
                'public': """
MARINE AREA NOTICE

Location: {ocean} Ocean, {region} area
Date: {date} 
Status: MODERATE RISK

Current Status:
Predictive models indicate moderate risk of microplastic presence 
near {latitude:.4f}°, {longitude:.4f}° ({probability:.1%} probability).

Public Information:
• Normal recreational activities may continue
• Follow standard marine safety practices  
• Be aware of local conditions
• Report unusual marine debris or conditions

This is an informational alert based on environmental modeling.
Local conditions may vary.

Stay informed: [Contact Information]
"""
            },
            'low_risk': {
                'environmental_agency': """
ENVIRONMENTAL STATUS UPDATE - Low Microplastic Risk

Location: {latitude:.4f}°, {longitude:.4f}° ({ocean}, {region})
Assessment Date: {date}
Confidence Level: {confidence}
Hotspot Probability: {probability:.1%}

STATUS SUMMARY:
• Low probability of microplastic accumulation
• Area suitable for normal marine activities
• Routine monitoring protocols sufficient
• No immediate environmental concerns identified

ONGOING MONITORING:
• Continue standard observation schedules
• Maintain baseline data collection
• Coordinate with regional monitoring networks
• Update assessment as new data becomes available

NEXT SCHEDULED REVIEW: 6 months
""",
                'fisheries': """
FISHERIES STATUS - Low Risk Area

Location: {latitude:.4f}°, {longitude:.4f}°
Water Body: {ocean}, {region}
Date: {date}
Risk Level: LOW ({probability:.1%} probability)

OPERATIONAL STATUS:
• Normal fishing operations approved
• Standard quality control procedures apply
• No additional restrictions necessary
• Area suitable for commercial activities

QUALITY STANDARDS:
• Follow regular seafood safety protocols
• Maintain standard documentation
• Continue routine product monitoring
• Report any unusual observations

Area remains open for normal fishing activities.
""",
                'public': """
MARINE CONDITIONS REPORT

Location: {ocean} Ocean, {region}
Date: {date}
Status: GOOD CONDITIONS

Current Assessment:
Low risk of microplastic contamination predicted for area near 
{latitude:.4f}°, {longitude:.4f}° ({probability:.1%} probability).

Public Information:
• Area suitable for normal recreational activities
• Standard marine safety practices recommended
• Good conditions for water sports and fishing
• Continue enjoying marine resources responsibly

This assessment is based on predictive environmental modeling.

For current conditions: [Contact Information]
"""
            }
        }
    
    def generate_advisory(self, 
                         prediction_result: Dict[str, Any], 
                         stakeholder: str = "public",
                         use_llm: bool = True) -> Dict[str, Any]:
        """
        Generate ecological risk advisory based on prediction results
        
        Args:
            prediction_result: Output from PredictionEngine
            stakeholder: Target stakeholder ('environmental_agency', 'fisheries', 'public')
            use_llm: Whether to use LLM for enhanced advisory generation
            
        Returns:
            Dictionary containing advisory text and metadata
        """
        if prediction_result.get('error'):
            return self._generate_error_advisory(prediction_result, stakeholder)
        
        # Extract key information
        location = prediction_result['location']
        prediction_data = prediction_result.get('prediction', {})
        confidence_level = prediction_result.get('confidence_level', 'Unknown')
        
        if not prediction_data:
            return self._generate_no_prediction_advisory(prediction_result, stakeholder)
        
        probability = prediction_data.get('hotspot_probability', 0)
        
        # Determine risk level
        risk_level = self._determine_risk_level(probability)
        
        # Get base template
        base_advisory = self._get_base_advisory(risk_level, stakeholder, prediction_result)
        
        # Enhance with LLM if requested and available
        if use_llm:
            try:
                enhanced_advisory = self._enhance_with_llm(base_advisory, prediction_result, stakeholder)
                base_advisory = enhanced_advisory
            except Exception as e:
                logger.warning(f"LLM enhancement failed, using template: {e}")
        
        # Create advisory package
        advisory = {
            'advisory_text': base_advisory,
            'metadata': {
                'generation_time': datetime.now().isoformat(),
                'risk_level': risk_level,
                'stakeholder': stakeholder,
                'confidence_level': confidence_level,
                'hotspot_probability': probability,
                'location': location,
                'prediction_date': prediction_result['prediction_date'],
                'llm_enhanced': use_llm,
                'advisory_id': self._generate_advisory_id(prediction_result)
            },
            'recommendations': self._extract_recommendations(base_advisory, risk_level, stakeholder),
            'follow_up': self._generate_follow_up_actions(risk_level, stakeholder)
        }
        
        return advisory
    
    def _determine_risk_level(self, probability: float) -> str:
        """Determine risk level category from probability"""
        if probability >= 0.6:
            return 'high_risk'
        elif probability >= 0.3:
            return 'moderate_risk'
        else:
            return 'low_risk'
    
    def _get_base_advisory(self, risk_level: str, stakeholder: str, prediction_result: Dict) -> str:
        """Get base advisory template and fill with data"""
        template = self.templates[risk_level][stakeholder]
        
        location = prediction_result['location']
        ensemble_result = prediction_result.get('ensemble_result', {})
        
        return template.format(
            latitude=location['latitude'],
            longitude=location['longitude'],  
            ocean=location['ocean'],
            region=location['region'],
            date=prediction_result['prediction_date'],
            confidence=prediction_result.get('confidence_level', 'Unknown'),
            probability=ensemble_result.get('hotspot_probability', 0)
        )
    
    def _enhance_with_llm(self, base_advisory: str, prediction_result: Dict, stakeholder: str) -> str:
        """Enhance advisory using LLM (placeholder for actual LLM integration)"""
        # This is a placeholder for actual LLM integration
        # In practice, you would integrate with OpenAI API, local LLM, etc.
        
        # For demonstration, we'll add some contextual information
        location = prediction_result['location']
        probability = prediction_result.get('ensemble_result', {}).get('hotspot_probability', 0)
        
        enhancement_prompt = f"""
        Please enhance this marine microplastic advisory with additional context:
        
        Base Advisory:
        {base_advisory}
        
        Additional Context:
        - Location: {location['latitude']:.4f}°, {location['longitude']:.4f}°
        - Ocean: {location['ocean']}
        - Region: {location['region']}
        - Hotspot Probability: {probability:.1%}
        - Target Stakeholder: {stakeholder}
        
        Please provide more specific ecological context, potential causes, 
        and actionable recommendations while maintaining the professional tone.
        """
        
        # Placeholder for actual LLM call
        # enhanced_text = call_llm_api(enhancement_prompt)
        
        # For now, return the base advisory with a note
        enhanced_advisory = base_advisory + "\n\n--- Enhanced Analysis ---\n"
        enhanced_advisory += "This advisory incorporates predictive modeling of ocean currents, "
        enhanced_advisory += "seasonal patterns, and historical microplastic distribution data. "
        enhanced_advisory += f"The {probability:.1%} probability reflects current environmental conditions "
        enhanced_advisory += "and should be validated with on-site measurements."
        
        return enhanced_advisory
    
    def _extract_recommendations(self, advisory_text: str, risk_level: str, stakeholder: str) -> List[str]:
        """Extract key recommendations from advisory text"""
        recommendations = []
        
        # Parse recommendations based on risk level and stakeholder
        if risk_level == 'high_risk':
            if stakeholder == 'environmental_agency':
                recommendations = [
                    "Deploy monitoring equipment immediately",
                    "Restrict fishing activities in the area", 
                    "Initiate water quality testing protocols",
                    "Alert marine protected area authorities"
                ]
            elif stakeholder == 'fisheries':
                recommendations = [
                    "Avoid fishing in this area for 4-6 weeks",
                    "Increase catch inspection protocols if fishing is essential",
                    "Do not consume fish without testing",
                    "Report unusual fish behavior"
                ]
            elif stakeholder == 'public':
                recommendations = [
                    "Avoid swimming in the immediate area",
                    "Do not collect seafood for personal consumption",
                    "Keep pets away from shoreline",
                    "Report unusual debris to authorities"
                ]
        elif risk_level == 'moderate_risk':
            if stakeholder == 'environmental_agency':
                recommendations = [
                    "Increase monitoring frequency",
                    "Schedule routine water quality assessments",
                    "Monitor marine life behavior patterns"
                ]
            elif stakeholder == 'fisheries':
                recommendations = [
                    "Continue normal operations with enhanced inspection",
                    "Monitor fish health closely",
                    "Maintain detailed catch documentation"
                ]
            elif stakeholder == 'public':
                recommendations = [
                    "Follow standard marine safety practices",
                    "Stay informed of local conditions",
                    "Report unusual marine debris"
                ]
        else:  # low_risk
            recommendations = [
                "Continue normal activities",
                "Follow standard safety procedures",
                "Maintain routine monitoring"
            ]
        
        return recommendations
    
    def _generate_follow_up_actions(self, risk_level: str, stakeholder: str) -> Dict[str, Any]:
        """Generate follow-up action plan"""
        follow_up = {
            'timeline': '30 days',
            'next_assessment': '7 days',
            'monitoring_frequency': 'weekly',
            'escalation_threshold': 0.8
        }
        
        if risk_level == 'high_risk':
            follow_up.update({
                'timeline': '7 days',
                'next_assessment': '24 hours',
                'monitoring_frequency': 'daily',
                'escalation_threshold': 0.9,
                'emergency_contacts': True
            })
        elif risk_level == 'moderate_risk':
            follow_up.update({
                'timeline': '14 days', 
                'next_assessment': '3 days',
                'monitoring_frequency': 'every 3 days',
                'escalation_threshold': 0.7
            })
        
        return follow_up
    
    def _generate_advisory_id(self, prediction_result: Dict) -> str:
        """Generate unique advisory ID"""
        location = prediction_result['location']
        date = prediction_result['prediction_date']
        
        # Create ID from location and date
        lat_str = f"{location['latitude']:.2f}".replace('.', '')
        lon_str = f"{location['longitude']:.2f}".replace('.', '')
        date_str = date.replace('-', '')
        
        return f"MPA-{date_str}-{lat_str}{lon_str}"
    
    def _generate_error_advisory(self, prediction_result: Dict, stakeholder: str) -> Dict[str, Any]:
        """Generate advisory for error cases"""
        return {
            'advisory_text': f"Unable to generate prediction advisory due to: {prediction_result['error']}",
            'metadata': {
                'generation_time': datetime.now().isoformat(),
                'risk_level': 'error',
                'stakeholder': stakeholder,
                'error': prediction_result['error']
            },
            'recommendations': ["Contact technical support", "Retry with valid input data"],
            'follow_up': {'timeline': 'immediate', 'action': 'resolve_error'}
        }
    
    def _generate_no_prediction_advisory(self, prediction_result: Dict, stakeholder: str) -> Dict[str, Any]:
        """Generate advisory when no prediction is available"""
        return {
            'advisory_text': "No prediction results available for advisory generation.",
            'metadata': {
                'generation_time': datetime.now().isoformat(),
                'risk_level': 'unknown',
                'stakeholder': stakeholder
            },
            'recommendations': ["Ensure model predictions are available", "Check input data quality"],
            'follow_up': {'timeline': 'immediate', 'action': 'obtain_prediction'}
        }


def main():
    """Main function for inference pipeline demonstration"""
    try:
        # Initialize components
        prediction_engine = PredictionEngine()
        advisory_generator = AdvisoryGenerator()
        
        # Example predictions
        test_locations = [
            {
                'latitude': 35.0,
                'longitude': -140.0,
                'date': '2024-03-15',
                'ocean': 'Pacific',
                'region': 'North Pacific Gyre'
            },
            {
                'latitude': 40.7,
                'longitude': -74.0, 
                'date': '2024-03-15',
                'ocean': 'Atlantic',
                'region': 'New York Bight'
            }
        ]
        
        print("=== Marine Microplastic Prediction & Advisory System ===\n")
        
        for i, location in enumerate(test_locations, 1):
            print(f"Location {i}: {location['latitude']:.2f}°, {location['longitude']:.2f}°")
            
            # Make prediction
            prediction = prediction_engine.predict_hotspot_detailed(**location)
            
            if prediction.get('prediction'):
                prob = prediction['prediction']['hotspot_probability']
                risk = prediction['confidence_level']
                print(f"Hotspot Probability: {prob:.1%}")
                print(f"Risk Level: {risk}")
                
                # Generate advisories for different stakeholders
                stakeholders = ['environmental_agency', 'fisheries', 'public']
                
                for stakeholder in stakeholders:
                    advisory = advisory_generator.generate_advisory(
                        prediction, stakeholder, use_llm=True
                    )
                    
                    print(f"\n--- {stakeholder.replace('_', ' ').title()} Advisory ---")
                    print(advisory['advisory_text'][:300] + "...")
                    print(f"Advisory ID: {advisory['metadata']['advisory_id']}")
            else:
                print("Prediction failed or unavailable")
            
            print("\n" + "="*60 + "\n")
        
        logger.info("Inference pipeline demonstration completed")
        
    except Exception as e:
        logger.error(f"Inference pipeline failed: {e}")
        print(f"Error: {e}")


if __name__ == "__main__":
    main()