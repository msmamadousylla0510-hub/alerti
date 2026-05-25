"""
Neighborhood Service - Extracts neighborhoods/quarters from OpenStreetMap
Specifically for Mali cities
"""
import requests
import json
import os
from typing import List, Dict, Optional
from datetime import datetime

from utils.bamako_communes import BAMAKO_COMMUNES
from utils.bamako_neighborhoods import (
    BAMAKO_COMMUNE_NEIGHBORHOODS,
    get_commune_from_neighborhood,
)

class NeighborhoodService:
    """Service for fetching neighborhood data from OpenStreetMap"""
    
    def __init__(self, cache_file=None):
        self.overpass_urls = [
            "https://overpass-api.de/api/interpreter",
            "https://overpass.kumi.systems/api/interpreter",
            "https://overpass.openstreetmap.ru/api/interpreter",
        ]
        self._overpass_index = 0
        self.overpass_timeout = 25
        self.overpass_headers = {
            "Accept": "application/json",
            "User-Agent": "Alerti-FloodSense/1.0 (Mali flood alerts)",
        }
        
        # Cache file for neighborhoods
        if cache_file is None:
            cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
            os.makedirs(cache_dir, exist_ok=True)
            cache_file = os.path.join(cache_dir, 'mali_neighborhoods.json')
        
        self.cache_file = cache_file
        self.cache = self._load_cache()
    
    def _load_cache(self) -> Dict:
        """Load neighborhood cache from file"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading cache: {e}")
                return {}
        return {}
    
    def _save_cache(self):
        """Save neighborhood cache to file"""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving cache: {e}")

    def _fetch_overpass(self, query: str) -> Optional[Dict]:
        """POST Overpass avec en-têtes JSON et bascule entre miroirs."""
        last_error = None
        for _ in range(len(self.overpass_urls)):
            url = self.overpass_urls[self._overpass_index]
            try:
                response = requests.post(
                    url,
                    data=query,
                    headers=self.overpass_headers,
                    timeout=self.overpass_timeout,
                )
                if response.status_code in (429, 503, 504):
                    self._overpass_index = (self._overpass_index + 1) % len(self.overpass_urls)
                    continue
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                last_error = e
                self._overpass_index = (self._overpass_index + 1) % len(self.overpass_urls)
        if last_error:
            print(f"Error fetching neighborhoods from Overpass API: {last_error}")
        return None
    
    def get_mali_neighborhoods(self, city: str) -> List[Dict]:
        """
        Get neighborhoods for a Mali city using Overpass API
        Searches for administrative boundaries and residential areas
        """
        city_lower = city.lower()
        cache_key = f"mali_{city_lower}"

        # Mali city coordinates (approximate)
        city_coords = {
            'bamako': {'lat': 12.6392, 'lon': -8.0029},
            'kayes': {'lat': 14.4469, 'lon': -11.4445},
            'sikasso': {'lat': 11.3167, 'lon': -5.6667},
            'mopti': {'lat': 14.4869, 'lon': -4.2000},
            'segou': {'lat': 13.4400, 'lon': -6.2600},
            'tombouctou': {'lat': 16.7733, 'lon': -3.0074},
            'gao': {'lat': 16.2667, 'lon': -0.0500},
            'kidal': {'lat': 18.4411, 'lon': 1.4078},
            'koulikoro': {'lat': 12.8667, 'lon': -7.5667},
            'koutiala': {'lat': 12.3833, 'lon': -5.4667}
        }
        
        if city_lower not in city_coords:
            print(f"City {city} not found in Mali cities list")
            return []
        
        coords = city_coords[city_lower]

        # Check cache after we know coords (needed for annotation)
        cached_data = self.cache.get(cache_key)
        if cached_data and 'last_updated' in cached_data:
            last_update = datetime.fromisoformat(cached_data['last_updated'])
            if (datetime.now() - last_update).days < 30:
                neighborhoods = cached_data.get('neighborhoods', [])
                if city_lower == 'bamako':
                    neighborhoods = self._annotate_bamako_neighborhoods(neighborhoods, coords)
                    cached_data['neighborhoods'] = neighborhoods
                    self._save_cache()
                return neighborhoods
        
        # Overpass query to find neighborhoods with full geometry (polygons)
        # We search for ways and relations with admin_level or residential tags
        # Using out geom to get full polygon coordinates
        query = f"""
        [out:json][timeout:{self.overpass_timeout}];
        (
          way["place"~"neighbourhood|suburb"]["name"](around:5000,{coords['lat']},{coords['lon']});
          node["place"~"neighbourhood|suburb"]["name"](around:5000,{coords['lat']},{coords['lon']});
        );
        out center;
        """

        data = self._fetch_overpass(query)
        try:
            if data is None:
                raise requests.RequestException("Overpass returned no data")
            
            neighborhoods = []
            seen_names = set()
            
            for element in data.get('elements', []):
                if element.get('type') in ['way', 'relation', 'node']:
                    name = element.get('tags', {}).get('name', '')
                    if name and name.lower() not in seen_names:
                        # Get center coordinates
                        if 'center' in element:
                            lat = element['center']['lat']
                            lon = element['center']['lon']
                        elif 'lat' in element and 'lon' in element:
                            lat = element['lat']
                            lon = element['lon']
                        else:
                            continue
                        
                        # Extract polygon geometry if available
                        polygon = None
                        bbox = None
                        
                        if element.get('type') == 'way' and 'geometry' in element:
                            # Way geometry: array of {lat, lon}
                            geometry = element.get('geometry', [])
                            if len(geometry) >= 3:  # At least 3 points for a polygon
                                polygon = [[point.get('lat'), point.get('lon')] for point in geometry if point.get('lat') and point.get('lon')]
                                # Close the polygon if not already closed
                                if len(polygon) >= 3:
                                    if polygon[0] != polygon[-1]:
                                        polygon.append(polygon[0])
                                    
                                    # Calculate bounding box
                                    lats = [p[0] for p in polygon]
                                    lons = [p[1] for p in polygon]
                                    bbox = {
                                        'min_lat': min(lats),
                                        'max_lat': max(lats),
                                        'min_lon': min(lons),
                                        'max_lon': max(lons)
                                    }
                        elif element.get('type') == 'relation':
                            # For relations, try to get outer way geometry
                            # This is simplified - full implementation would fetch relation members
                            if 'members' in element:
                                # Use center point for now, could be enhanced
                                pass
                        
                        neighborhood_data = {
                            'name': name,
                            'city': city,
                            'coordinates': {'lat': lat, 'lon': lon},
                            'type': element.get('tags', {}).get('place', 'neighbourhood')
                        }
                        
                        # Add polygon and bbox if available
                        if polygon and len(polygon) >= 3:
                            neighborhood_data['polygon'] = polygon
                            if bbox:
                                neighborhood_data['bbox'] = bbox
                        
                        neighborhoods.append(neighborhood_data)
                        seen_names.add(name.lower())
            
            # If no neighborhoods found via Overpass, use fallback data
            if not neighborhoods:
                neighborhoods = self._get_fallback_neighborhoods(city_lower, coords)
            
            if city_lower == 'bamako':
                neighborhoods = self._annotate_bamako_neighborhoods(neighborhoods, coords)

            # Update cache
            self.cache[cache_key] = {
                'neighborhoods': neighborhoods,
                'last_updated': datetime.now().isoformat(),
                'city': city,
                'count': len(neighborhoods)
            }
            self._save_cache()
            
            return neighborhoods
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching neighborhoods from Overpass API: {e}")
            # Return fallback data
            neighborhoods = self._get_fallback_neighborhoods(city_lower, coords)
            if city_lower == 'bamako':
                neighborhoods = self._annotate_bamako_neighborhoods(neighborhoods, coords)
            return neighborhoods
        except Exception as e:
            print(f"Error processing neighborhoods: {e}")
            neighborhoods = self._get_fallback_neighborhoods(city_lower, coords)
            if city_lower == 'bamako':
                neighborhoods = self._annotate_bamako_neighborhoods(neighborhoods, coords)
            return neighborhoods
    
    def _get_fallback_neighborhoods(self, city: str, coords: Dict) -> List[Dict]:
        """Fallback neighborhoods for major Mali cities when API fails"""
        fallback_data = {
            'bamako': [
                {'name': 'Commune I', 'coordinates': {'lat': 12.65, 'lon': -8.0}, 'type': 'commune'},
                {'name': 'Commune II', 'coordinates': {'lat': 12.64, 'lon': -8.0}, 'type': 'commune'},
                {'name': 'Commune III', 'coordinates': {'lat': 12.63, 'lon': -8.0}, 'type': 'commune'},
                {'name': 'Commune IV', 'coordinates': {'lat': 12.62, 'lon': -8.0}, 'type': 'commune'},
                {'name': 'Commune V', 'coordinates': {'lat': 12.61, 'lon': -8.0}, 'type': 'commune'},
                {'name': 'Commune VI', 'coordinates': {'lat': 12.60, 'lon': -8.0}, 'type': 'commune'},
                {'name': 'Badalabougou', 'coordinates': {'lat': 12.65, 'lon': -7.98}, 'type': 'neighbourhood'},
                {'name': 'Aci 2000', 'coordinates': {'lat': 12.64, 'lon': -7.99}, 'type': 'neighbourhood'},
                {'name': 'Hamdallaye', 'coordinates': {'lat': 12.63, 'lon': -8.01}, 'type': 'neighbourhood'},
                {'name': 'Niaréla', 'coordinates': {'lat': 12.62, 'lon': -8.02}, 'type': 'neighbourhood'},
                {'name': 'Quartier du Fleuve', 'coordinates': {'lat': 12.61, 'lon': -8.03}, 'type': 'neighbourhood'},
            ],
            'kayes': [
                {'name': 'Centre-ville', 'coordinates': {'lat': 14.45, 'lon': -11.44}, 'type': 'neighbourhood'},
                {'name': 'Sabouciré', 'coordinates': {'lat': 14.46, 'lon': -11.43}, 'type': 'neighbourhood'},
            ],
            'sikasso': [
                {'name': 'Centre-ville', 'coordinates': {'lat': 11.32, 'lon': -5.67}, 'type': 'neighbourhood'},
                {'name': 'Bougoula', 'coordinates': {'lat': 11.31, 'lon': -5.66}, 'type': 'neighbourhood'},
            ],
            'mopti': [
                {'name': 'Centre-ville', 'coordinates': {'lat': 14.49, 'lon': -4.20}, 'type': 'neighbourhood'},
                {'name': 'Sévaré', 'coordinates': {'lat': 14.50, 'lon': -4.19}, 'type': 'neighbourhood'},
            ]
        }
        
        neighborhoods = fallback_data.get(city, [])
        # Add city name to each neighborhood
        for n in neighborhoods:
            n['city'] = city.title()
        
        return neighborhoods

    def _annotate_bamako_neighborhoods(self, neighborhoods: List[Dict], city_coords: Dict) -> List[Dict]:
        """Ajoute les informations de commune pour les quartiers de Bamako et complète la liste à partir des PDF."""
        neighborhoods = neighborhoods or []
        existing_by_name = {n['name'].lower(): n for n in neighborhoods if 'name' in n}
        
        # Annotate existing entries
        for n in neighborhoods:
            commune = get_commune_from_neighborhood(n.get('name', ''))
            if commune:
                n['commune'] = commune
            elif n.get('type') == 'commune':
                n['commune'] = n['name']
        
        # Add missing risk neighborhoods from static mapping
        for commune_name, names in BAMAKO_COMMUNE_NEIGHBORHOODS.items():
            commune_info = BAMAKO_COMMUNES.get(commune_name, {})
            lat = commune_info.get('lat', city_coords['lat'])
            lon = commune_info.get('lon', city_coords['lon'])
            for name in names:
                key = name.lower()
                if key in existing_by_name:
                    existing_by_name[key].setdefault('commune', commune_name)
                    continue
                new_entry = {
                    'name': name,
                    'city': 'Bamako',
                    'coordinates': {'lat': lat, 'lon': lon},
                    'type': 'neighbourhood',
                    'commune': commune_name
                }
                neighborhoods.append(new_entry)
                existing_by_name[key] = new_entry
        
        return neighborhoods
    
    def get_neighborhood_coordinates(self, neighborhood_name: str, city: str) -> Optional[Dict]:
        """Get coordinates for a specific neighborhood"""
        neighborhoods = self.get_mali_neighborhoods(city)
        
        neighborhood_lower = neighborhood_name.lower()
        for n in neighborhoods:
            if n['name'].lower() == neighborhood_lower:
                return n['coordinates']
        
        return None
    
    def get_all_mali_neighborhoods(self) -> Dict[str, List[Dict]]:
        """Get neighborhoods for all major Mali cities"""
        cities = ['bamako', 'kayes', 'sikasso', 'mopti', 'segou']
        all_neighborhoods = {}
        
        for city in cities:
            neighborhoods = self.get_mali_neighborhoods(city)
            all_neighborhoods[city] = neighborhoods
        
        return all_neighborhoods
    
    def search_neighborhood(self, query: str, city: str = None) -> List[Dict]:
        """Search for neighborhoods by name"""
        if city:
            neighborhoods = self.get_mali_neighborhoods(city)
        else:
            # Search all cities
            all_neighborhoods = self.get_all_mali_neighborhoods()
            neighborhoods = []
            for city_neighborhoods in all_neighborhoods.values():
                neighborhoods.extend(city_neighborhoods)
        
        query_lower = query.lower()
        results = [
            n for n in neighborhoods
            if query_lower in n['name'].lower()
        ]
        
        return results

