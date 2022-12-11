import functions_framework

from math import *
import numpy as np
import pandas as pd
from datetime import timedelta, datetime
import json

import skyfield
from skyfield.api import load
from skyfield.api import N, W, S, E, wgs84
from skyfield.api import Star
from skyfield.data import hipparcos
from skyfield.almanac import find_discrete, risings_and_settings
from skyfield.magnitudelib import planetary_magnitude
from skyfield import almanac

ts = load.timescale()
eph = load('de421.bsp')

sun = eph['SUN']
earth = eph['EARTH']
moon = eph['MOON']

with load.open(hipparcos.URL) as f:
    stars_df = hipparcos.load_dataframe(f)

planet_dictionary = {"Venus":"VENUS", "Mars":"MARS", "Jupiter":"JUPITER BARYCENTER", "Saturn":"SATURN BARYCENTER"}
    
star_dictionary = {"Alpheratz":677, "Ankaa":2081, "Schedar":3179, "Diphda":3419, "Achernar":7588, "Hamal":9884, "Polaris":11767, "Acamar":13847, "Menkar":14135, "Mirfak":15863, "Aldebaran":21421, "Rigel":24436, "Capella":24608, "Bellatrix":25336, "Elnath":25428, "Alnilam":26311, "Betelgeuse":27989, "Canopus":30438, "Sirius":32349, "Adhara":33579, "Procyon":37279, "Pollux":37826, "Avior":41037, "Suhail":44816, "Miaplacidus":45238, "Alphard":46390, "Regulus":49669, "Dubhe":54061, "Denebola":57632, "Gienah":59803, "Acrux":60718, "Gacrux":61084, "Alioth":62956, "Spica":65474, "Alkaid":67301, "Hadar":68702, "Menkent":68933, "Arcturus":69673, "Rigil Kent.":71683, "Kochab":72607, "Zuben'ubi":72622, "Alphecca":76267, "Antares":80763, "Atria":82273, "Sabik":84012, "Shaula":85927, "Rasalhague":86032, "Eltanin":87833, "Kaus Aust.":90185, "Vega":91262, "Nunki":92855, "Altair":97649, "Peacock":100751, "Deneb":102098, "Enif":107315, "Al Na'ir":109268, "Fomalhaut":113368, "Scheat":113881, "Markab":113963}

@functions_framework.http
def getGeographicalPosition(request):
    """HTTP Cloud Function.
        Args:
            request (flask.Request): The request object.
            <https://flask.palletsprojects.com/en/1.1.x/api/#incoming-request-data>
        Returns:
            The response text, or any set of values that can be turned into a
            Response object using `make_response`
            <https://flask.palletsprojects.com/en/1.1.x/api/#flask.make_response>.
    """
    request_args = request.args

    object_name = None
    dt = None

    if request_args and 'body' in request_args:
        object_name = request_args['body']

    if request_args and 'utc' in request_args:
        utc_arg = request_args['utc']
        utc_ts = float(utc_arg)
        dt = datetime.fromtimestamp(utc_ts)

    display = dt.strftime('%Y/%m/%d %H:%M:%S')
    t = ts.ut1(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)
    celestial_object = getCelestialObject(object_name)
    
    apparent = earth.at(t).observe(celestial_object).apparent()
    radec = apparent.radec()
    ra = radec[0].hours
    dec = radec[1].degrees
    distance = apparent.distance().km
    
    gha = (t.gast-ra)*15
    
    if (gha < 0):
        gha += 360
        
    response = {
        "body" : object_name,
        "utc" : display,
        "GHA" : gha,
        "ra" : ra,
        "dec" : dec,
        "distance": distance,
    }
    return (json.dumps(response))

@functions_framework.http
def getCelestialObjectData(request):
    request_args = request.args

    lat = None 
    if request_args and 'lat' in request_args:
        lat_arg = request_args['lat']
        lat=float(lat_arg)

    lon = None
    if request_args and 'lon' in request_args:
        lon_arg = request_args['lon']
        lon=float(lon_arg)

    rise_start = None
    if request_args and 'riseStart' in request_args:
        rise_start_arg = request_args['riseStart']
        dt = datetime.fromtimestamp(float(rise_start_arg))
        rise_start = ts.ut1(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)

    rise_days = None
    if request_args and 'riseDays' in request_args:
        rise_days_arg = request_args['riseDays']
        rise_days = float(rise_days_arg)

    objects = []
    order = 1
    
    location = wgs84.latlon(lat, lon)
    earth_at = (earth + location).at(rise_start)
    
    # Sun
    radec = earth_at.observe(sun).apparent().radec()
    rise_set = getRiseSetTimes(location, sun, rise_start, rise_days)
    objects.append(CelestialObject(order, "Sun", "Sun", 0.0, radec[0].hours, radec[1].degrees, radec[2].au, rise_set))
    order+=1
    
    # Moon
    radec = earth_at.observe(moon).apparent().radec()
    rise_set = getRiseSetTimes(location, moon, rise_start, rise_days)
    objects.append(CelestialObject(order, "Moon", "Moon", 0.0, radec[0].hours, radec[1].degrees, radec[2].au, rise_set))
    order+=1
    
    # Planets
    for planet_name in planet_dictionary:
        planet = eph[planet_dictionary[planet_name]]
        astrometric = earth_at.observe(planet)
        magnitude = planetary_magnitude(astrometric)
        radec = astrometric.apparent().radec()
        rise_set = getRiseSetTimes(location, planet, rise_start, rise_days)
        objects.append(CelestialObject(order, planet_name, "Planet", float(magnitude), radec[0].hours, radec[1].degrees, radec[2].au, rise_set))
        order+=1
        
    # Stars
    for star_name in star_dictionary:
        # Load dataframe separately to get magnitude
        star_df = stars_df.loc[star_dictionary[star_name]]
        star = Star.from_dataframe(star_df)
        radec = earth_at.observe(star).apparent().radec()
        rise_set = getRiseSetTimes(location, star, rise_start, rise_days)
        objects.append(CelestialObject(order, star_name, "Star", star_df.magnitude, radec[0].hours, radec[1].degrees, radec[2].au, rise_set))
        order+=1
        
    return json.dumps([ob.__dict__ for ob in objects], cls=CelestialObjectEncoder)


class CelestialObject():
    def __init__(self, order, name, objtype, magnitude, ra, dec, distance, riseset=None):
        self.order = order
        self.name = name
        self.objtype = objtype
        self.magnitude = magnitude
        self.ra = ra
        self.dec = dec
        self.distance = distance
        self.riseset = riseset
    def reprJSON(self):
        return dict(name=self.name, 
                    objtype=self.objtype, 
                    magnitude=self.magnitude, 
                    ra=self.ra, 
                    distance=self.distance, 
                    riseset=self.riseset)
        
class RiseSetTime():
    def __init__(self, utc, riseset):
        self.utc = utc
        self.riseset = riseset
    def reprJSON(self):
        return dict(utc=self.utc, riseset=self.riseset)
    
class CelestialObjectEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj,'reprJSON'):
            return obj.reprJSON()
        else:
            return json.JSONEncoder.default(self, obj)  

def getRiseSetTimes(location, celestial_obj, rise_start, rise_days):
    rise_set_times = []
    
    rise_end = rise_start + timedelta(days = rise_days)
    
    rs = risings_and_settings(eph, celestial_obj, location)
    
    for time, riseset in zip(*find_discrete(rise_start, rise_end, rs)):
        rise_set_times.append(RiseSetTime(time.utc_datetime().timestamp(), int(riseset)))  
        
    return rise_set_times

def getCelestialObject(object_name):
    celestial_obj = None
    if (object_name == "Sun"):
        celestial_obj = eph['Sun']
    elif (object_name == "Moon"):
        celestial_obj = eph['Moon']
    elif (object_name == "Venus"):
        celestial_obj = eph['Venus']    
    elif (object_name == "Mars"):
        celestial_obj = eph['Mars']
    elif (object_name == "Jupiter"):
        celestial_obj = eph['JUPITER BARYCENTER']
    elif (object_name == "Saturn"):
        celestial_obj = eph['SATURN BARYCENTER']
    else:
        celestial_obj = Star.from_dataframe(stars_df.loc[star_dictionary[object_name]])
    return celestial_obj

