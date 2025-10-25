#!/usr/bin/env python3
"""
Simple Marine Microplastic Advisory System
Input location → Get nearby place + hotspot probability + marine authority contact
"""

import sys
sys.path.append('src')

from inference.predict_and_advise import PredictionEngine
from datetime import datetime
import math

def get_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points using Haversine formula"""
    R = 6371  # Earth's radius in kilometers
    
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    return R * c

def find_nearest_place(lat, lon):
    """Find nearest known marine location"""
    
    # Database of major marine locations worldwide
    marine_locations = [
        # North America - Atlantic
        (40.7589, -73.9851, "New York Harbor", "United States"),
        (25.7617, -80.1918, "Miami Beach", "United States"),
        (44.6532, -63.5986, "Halifax Harbor", "Canada"),
        (42.3601, -71.0589, "Boston Harbor", "United States"),
        
        # North America - Pacific
        (33.9425, -118.4081, "Los Angeles Coast", "United States"),
        (37.7749, -122.4194, "San Francisco Bay", "United States"),
        (47.6062, -122.3321, "Seattle Harbor", "United States"),
        (49.2827, -123.1207, "Vancouver Harbor", "Canada"),
        
        # Europe
        (51.5074, -0.1278, "Thames Estuary", "United Kingdom"),
        (52.3676, 4.9041, "Amsterdam Port", "Netherlands"),
        (53.5511, 9.9937, "Hamburg Harbor", "Germany"),
        (59.9139, 10.7522, "Oslo Fjord", "Norway"),
        (41.9028, 12.4964, "Rome Coast", "Italy"),
        (43.2965, 5.3698, "Marseille Harbor", "France"),
        
        # Asia-Pacific
        (35.6762, 139.6503, "Tokyo Bay", "Japan"),
        (-33.8688, 151.2093, "Sydney Harbor", "Australia"),
        (22.3193, 114.1694, "Hong Kong Harbor", "China"),
        (1.3521, 103.8198, "Singapore Strait", "Singapore"),
        (37.5665, 126.9780, "Seoul Coast", "South Korea"),
        (13.7563, 100.5018, "Bangkok Gulf", "Thailand"),
        
        # Other regions
        (-23.5505, -46.6333, "Santos Port", "Brazil"),
        (-34.6037, -58.3816, "Buenos Aires Coast", "Argentina"),
        (30.0444, 31.2357, "Alexandria Harbor", "Egypt"),
        (-26.2041, 28.0473, "Johannesburg Region", "South Africa"),
    ]
    
    # Find closest location
    min_distance = float('inf')
    nearest_place = None
    nearest_country = None
    
    for place_lat, place_lon, place_name, country in marine_locations:
        distance = get_distance(lat, lon, place_lat, place_lon)
        if distance < min_distance:
            min_distance = distance
            nearest_place = place_name
            nearest_country = country
    
    return nearest_place, nearest_country, min_distance

def get_marine_authority_contact(country):
    """Get marine authority contact information by country"""
    
    contacts = {
        "United States": {
            "authority": "U.S. Coast Guard",
            "emergency": "+1-800-424-8802",
            "non_emergency": "+1-202-372-2100",
            "website": "www.uscg.mil"
        },
        "Canada": {
            "authority": "Canadian Coast Guard",
            "emergency": "+1-800-267-7270",
            "non_emergency": "+1-613-993-0999",
            "website": "www.ccg-gcc.gc.ca"
        },
        "United Kingdom": {
            "authority": "Maritime and Coastguard Agency",
            "emergency": "999 (UK Emergency)",
            "non_emergency": "+44-203-817-2000",
            "website": "www.gov.uk/maritime-and-coastguard-agency"
        },
        "Australia": {
            "authority": "Australian Maritime Safety Authority",
            "emergency": "+61-1800-803-772",
            "non_emergency": "+61-2-6279-5000",
            "website": "www.amsa.gov.au"
        },
        "Japan": {
            "authority": "Japan Coast Guard",
            "emergency": "118 (Marine Emergency)",
            "non_emergency": "+81-3-3591-6361",
            "website": "www.kaiho.mlit.go.jp"
        },
        "Germany": {
            "authority": "German Maritime Search and Rescue",
            "emergency": "+49-421-536870",
            "non_emergency": "+49-381-4563-0",
            "website": "www.havariekommando.de"
        },
        "France": {
            "authority": "French Maritime Prefecture",
            "emergency": "196 (Sea Emergency)",
            "non_emergency": "+33-2-98-22-40-40",
            "website": "www.premar-atlantique.gouv.fr"
        },
        "Netherlands": {
            "authority": "Dutch Coast Guard",
            "emergency": "112 (Emergency)",
            "non_emergency": "+31-223-542300",
            "website": "www.kustwacht.nl"
        },
        "Norway": {
            "authority": "Norwegian Coastal Administration",
            "emergency": "120 (Sea Emergency)",
            "non_emergency": "+47-07847",
            "website": "www.kystverket.no"
        },
        "Singapore": {
            "authority": "Maritime and Port Authority of Singapore",
            "emergency": "999 (Emergency)",
            "non_emergency": "+65-6325-2488",
            "website": "www.mpa.gov.sg"
        },
        "China": {
            "authority": "China Maritime Safety Administration",
            "emergency": "12395 (Maritime Emergency)",
            "non_emergency": "+86-10-6529-2218",
            "website": "www.msa.gov.cn"
        },
        "South Korea": {
            "authority": "Korea Coast Guard",
            "emergency": "122 (Coast Guard)",
            "non_emergency": "+82-32-835-2000",
            "website": "www.kcg.go.kr"
        },
        "Italy": {
            "authority": "Italian Coast Guard",
            "emergency": "1530 (Coast Guard)",
            "non_emergency": "+39-06-5908-4409",
            "website": "www.guardiacostiera.gov.it"
        },
        "Brazil": {
            "authority": "Brazilian Navy",
            "emergency": "185 (Fire/Rescue)",
            "non_emergency": "+55-21-2104-5000",
            "website": "www.marinha.mil.br"
        },
        "Argentina": {
            "authority": "Argentine Naval Prefecture",
            "emergency": "106 (Naval Emergency)",
            "non_emergency": "+54-11-4576-7000",
            "website": "www.prefecturanaval.gov.ar"
        },
        "Thailand": {
            "authority": "Royal Thai Navy",
            "emergency": "191 (Police Emergency)",
            "non_emergency": "+66-2-475-8236",
            "website": "www.navy.mi.th"
        },
        "Egypt": {
            "authority": "Egyptian Maritime Transport Sector",
            "emergency": "122 (Emergency)",
            "non_emergency": "+20-2-2574-9720",
            "website": "www.mts.gov.eg"
        },
        "South Africa": {
            "authority": "South African Maritime Safety Authority",
            "emergency": "10177 (Emergency)",
            "non_emergency": "+27-12-366-2600",
            "website": "www.samsa.org.za"
        }
    }
    
    return contacts.get(country, {
        "authority": "Local Maritime Authority",
        "emergency": "Local emergency services (911/112/999)",
        "non_emergency": "Contact local port authority",
        "website": "Consult local maritime regulations"
    })

def get_risk_level(probability):
    """Determine risk level"""
    if probability >= 0.70:
        return "🔴 CRITICAL RISK", "🚨 IMMEDIATE ACTION REQUIRED"
    elif probability >= 0.50:
        return "🟠 HIGH RISK", "⚠️ ENHANCED MONITORING NEEDED"
    elif probability >= 0.30:
        return "🟡 MEDIUM RISK", "⚠️ INCREASED VIGILANCE"
    elif probability >= 0.15:
        return "🟢 LOW RISK", "✅ STANDARD MONITORING"
    else:
        return "✅ VERY LOW RISK", "✅ MINIMAL CONCERN"

def main():
    """Main advisory system"""
    
    print("🌊 SIMPLE MARINE ADVISORY SYSTEM")
    print("="*50)
    print("Input coordinates → Get location + risk + contacts")
    print()
    
    # Initialize ML system
    print("🔧 Initializing...")
    try:
        engine = PredictionEngine()
        engine.initialize()
        print("✅ Ready!")
    except Exception as e:
        print(f"❌ Failed: {e}")
        return
    
    while True:
        try:
            print("\n" + "="*50)
            print("📍 ENTER COORDINATES")
            print("="*50)
            
            # Get coordinates
            lat_input = input("🌐 Latitude (-90 to 90°): ").strip()
            lon_input = input("🌐 Longitude (-180 to 180°): ").strip()
            
            if not lat_input or not lon_input:
                print("❌ Please enter both coordinates")
                continue
            
            try:
                lat = float(lat_input)
                lon = float(lon_input)
            except ValueError:
                print("❌ Invalid coordinates")
                continue
            
            if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
                print("❌ Coordinates out of range")
                continue
            
            print(f"\n🔍 Analyzing location {lat:.4f}°, {lon:.4f}°...")
            
            # Find nearest place
            nearest_place, country, distance = find_nearest_place(lat, lon)
            
            # Get ML prediction
            probability = engine.predict_hotspot(lat, lon)
            risk_level, action = get_risk_level(probability)
            
            # Get marine authority contact
            contact_info = get_marine_authority_contact(country)
            
            # Generate simple advisory
            print(f"""
🌊 MARINE ADVISORY
{'='*50}
📍 LOCATION: {lat:.4f}°, {lon:.4f}°
🗺️  NEAREST PLACE: This location seems to be near {nearest_place}
🏴 COUNTRY: {country}
📏 DISTANCE: {distance:.1f} km from {nearest_place}

📊 RISK ASSESSMENT
{'='*50}
🎯 HOTSPOT PROBABILITY: {probability:.1%}
📈 RISK LEVEL: {risk_level}
⚡ RECOMMENDED ACTION: {action}

📞 MARINE AUTHORITY CONTACT
{'='*50}
🏢 AUTHORITY: {contact_info['authority']}
🚨 EMERGENCY: {contact_info['emergency']}
📞 NON-EMERGENCY: {contact_info['non_emergency']}
🌐 WEBSITE: {contact_info['website']}

⚠️  For immediate marine emergencies, always contact local emergency services first!
{'='*50}""")
            
            # Continue?
            choice = input("\n❓ Check another location? (y/n): ").strip().lower()
            if choice not in ['y', 'yes']:
                break
                
        except KeyboardInterrupt:
            print("\n\n👋 Thank you for using Marine Advisory System!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()