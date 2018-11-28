from __future__ import absolute_import
from __future__ import division
from past.utils import old_div
from shapely import geometry
from rasterio import features
import numpy as np
from skimage import filters
import os
import requests
import sys
from builtins import zip
import jinja2
import pandas as pd
import folium
import matplotlib.pyplot as plt
import json


# FUNCTIONS
def to_geojson(l):
    g = {'crs': {u'properties': {u'name': u'urn:ogc:def:crs:OGC:1.3:CRS84'}, 'type': 'name'},
         'features': [{'geometry': d['geometry'].__geo_interface__, 'properties': d['properties'], 'type': 'Feature'}
                      for d in l],
         'type': u'FeatureCollection'}

    if sys.version_info[0] == 3:
        serializer = np_serializer
    else:
        serializer = None

    gj = json.dumps(g, default=serializer)

    return gj

def np_serializer(i):
    if type(i).__module__ == np.__name__:
        return np.asscalar(i)
    raise TypeError(repr(i) + " is not JSON serializable")

def from_geojson(source):
    if source.startswith('http'):
        response = requests.get(source)
        geojson = json.loads(response.content)
    else:
        if os.path.exists(source):
            with open(source, 'r') as f:
                geojson = json.loads(f.read())
        else:
            raise ValueError("File does not exist: {}".format(source))

    geometries = []
    feats = []
    for f in geojson['features']:
        geom = geometry.shape(f['geometry'])
        feats.append({'geometry': geom, 'properties': {}})
        geometries.append(geom)

    return geometries, feats


def geom_to_array(geom, img, geom_val=1, fill_value=0, all_touched=True, exterior_only=False):
    if exterior_only is True:
        geom_to_rasterize = geom.exterior
    else:
        geom_to_rasterize = geom

    geom_array = features.rasterize([(geom_to_rasterize, geom_val)],
                                    out_shape=(img.shape[1], img.shape[2]),
                                    transform=img.affine,
                                    fill=fill_value,
                                    all_touched=all_touched,
                                    dtype=np.uint8)

    return geom_array


def get_median_rgb(geom, image, geom_buffer_size=0.0003, return_chips=False):
    bounds = geom.buffer(geom_buffer_size).bounds
    chip = image.aoi(bbox=bounds, pansharpen=True)
    bldg_array = geom_to_array(geom, chip)
    rgb = chip.rgb(quiet=True)
    bldg_mask = np.zeros(rgb.shape)
    bldg_mask[:, :, :] = (bldg_array == 0)[:, :, None]
    rgb_masked = np.ma.masked_where(bldg_mask, rgb)
    red_vals = rgb_masked[:, :, 0].compressed()
    green_vals = rgb_masked[:, :, 1].compressed()
    blue_vals = rgb_masked[:, :, 2].compressed()
    pixels = np.column_stack([red_vals, green_vals, blue_vals])
    medians = np.ma.median(pixels, axis=0)
    dist = np.linalg.norm(pixels - medians, axis=1)
    psuedo_medoid = pixels[np.argmin(dist), :]
    rgb_vals = list(np.round(pixels[np.argmin(dist), :] * 255., 0).astype('int'))

    if return_chips is False:
        return rgb_vals
    else:
        # create the masked color image
        bldg_colored = np.zeros(rgb.shape)
        bldg_colored[:, :, 0] = psuedo_medoid[0]
        bldg_colored[:, :, 1] = psuedo_medoid[1]
        bldg_colored[:, :, 2] = psuedo_medoid[2]
        bldg_colored *= (bldg_mask == 0)

        return rgb_vals, rgb, bldg_colored


def convert_response_to_binary(response):
    if response == 'Yes':
        binary = 1
    elif response == 'No':
        binary = 0

    return binary


def calc_overhanging_veg(geom, image, geom_buffer_size=0.0003, return_chips=False):
    bounds = geom.buffer(geom_buffer_size).bounds
    chip = image.aoi(bbox=bounds, pansharpen=True)
    ndvi = chip.ndvi(quiet=True)
    # First, clean up any nan values
    ndvi[np.isnan(ndvi)] = 0
    # Next, run a gaussian smoothing kernel over the image. This will smooth out localized noise in the water index
    # by use a focal moving window.
    ndvi_smoothed = filters.gaussian(ndvi, preserve_range=True)
    # find the vegetation
    veg = ndvi_smoothed >= filters.threshold_otsu(ndvi_smoothed)
    bldg_array = geom_to_array(geom, chip)
    overhanging_trees = bldg_array * veg

    # calc pct overhnging
    pct_overhanging_trees = old_div(float(overhanging_trees.sum()), bldg_array.sum())

    if return_chips is False:
        return pct_overhanging_trees
    else:
        overhanging_trees_with_bldg = overhanging_trees + bldg_array
        chip_rgb = chip.rgb()
        bldg_colored = np.zeros(chip_rgb.shape)
        bldg_colored[:, :, 0][overhanging_trees_with_bldg == 1] = old_div(180, 255.)
        bldg_colored[:, :, 1][overhanging_trees_with_bldg == 1] = old_div(180, 255.)
        bldg_colored[:, :, 2][overhanging_trees_with_bldg == 1] = old_div(180, 255.)
        bldg_colored[:, :, 0][overhanging_trees_with_bldg == 2] = old_div(0, 255.)
        bldg_colored[:, :, 1][overhanging_trees_with_bldg == 2] = old_div(255, 255.)
        bldg_colored[:, :, 2][overhanging_trees_with_bldg == 2] = old_div(23, 255.)

        return pct_overhanging_trees, chip_rgb, bldg_colored


# PLOTTING FUNCTIONS AND CONSTANTS
# CONSTANTS
FOOTPRINTS_SAMPLE_SMALL = 'https://s3.amazonaws.com/gbdx-training/ecopia_sample/ecopia_footprints_sample_small.geojson'
IDAHO_TMS_1030010067C11400 = 'https://idaho.geobigdata.io/v1/tile/idaho-images/1ee4d8dc-3bc1-4c28-8914-30d79f149467/{{z}}/{{x}}/{{y}}?bands=4,2,1&panId=8e46348d-3d4b-4552-ae6e-bc1c7d9c0e6f&token={token}'
TMS_1030010067C11400 = 'https://s3.amazonaws.com/notebooks-small-tms/1030010067C11400/{z}/{x}/{y}.png'
ESRI_TILES = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}.jpg'
COLORS = {'gray'       : '#8F8E8E',
          'white'      : '#FFFFFF',
          'brightgreen': '#00FF17',
          'red'        : '#FF0000',
          'cyan'       : '#1FFCFF'}


# FUNCTIONS
def footprints_outline_styler(x):
    return {'fillOpacity': .25,
            'color'      : COLORS['brightgreen'],
            'fillColor'  : COLORS['gray'],
            'weight'     : 1}


def footprints_color_styler(x):
    return {'fillColor'  : rgb2hex(x['properties']['rgbcolor']),
            'color'      : COLORS['white'],
            'fillOpacity': 1,
            'weight'     : 0.5}


def footprints_trees_styler(x):
    return {'fillColor'  : COLORS['gray'],
            'color'      : COLORS['red'] if x['properties']['sig_overhanging_trees'].lower() == 'true' else COLORS[
                'white'],
            'fillOpacity': 0,
            'weight'     : 1}


def footprints_label_styler(x):
    if 'manual_label' not in list(x['properties'].keys()):
        color = COLORS['gray']
    else:
        if x['properties']['manual_label'] == 1:
            color = COLORS['cyan']
        else:
            color = COLORS['white']

    return {'fillColor'  : COLORS['white'],
            'color'      : color,
            'fillOpacity': 0,
            'weight'     : 1}


def footprints_combined_styler(x):
    return {'fillColor'  : rgb2hex(x['properties']['rgbcolor']),
            'color'      : COLORS['red'] if x['properties']['sig_overhanging_trees'].lower() == 'true' else COLORS[
                'white'],
            'fillOpacity': 1,
            'weight'     : 1}


def folium_map(geojson_to_overlay, layer_name, location, style_function=None, tiles='Stamen Terrain', zoom_start=16,
               show_layer_control=True, width='100%', height='75%', attr=None, map_zoom=18, max_zoom=20, tms=False,
               zoom_beyond_max=None, base_tiles='OpenStreetMap', opacity=1,
               tooltip_props=None, tooltip_aliases=None):
    m = folium.Map(location=location, zoom_start=zoom_start, width=width, height=height, max_zoom=map_zoom,
                   tiles=base_tiles)
    tiles = folium.TileLayer(tiles=tiles, attr=attr, name=attr, max_zoom=max_zoom,
                             overlay=True, show=True)
    if tms is True:
        options = json.loads(tiles.options)
        options.update({'tms': True})
        tiles.options = json.dumps(options, sort_keys=True, indent=2)
        tiles._template = jinja2.Template(u"""
        {% macro script(this, kwargs) %}
            var {{this.get_name()}} = L.tileLayer(
                '{{this.tiles}}',
                {{ this.options }}
                ).addTo({{this._parent.get_name()}});
        {% endmacro %}
        """)
    if zoom_beyond_max is not None:
        options = json.loads(tiles.options)
        options.update({'maxNativeZoom': zoom_beyond_max, 'maxZoom': max_zoom})
        tiles.options = json.dumps(options, sort_keys=True, indent=2)
        tiles._template = jinja2.Template(u"""
        {% macro script(this, kwargs) %}
            var {{this.get_name()}} = L.tileLayer(
                '{{this.tiles}}',
                {{ this.options }}
                ).addTo({{this._parent.get_name()}});
        {% endmacro %}
        """)
    if opacity < 1:
        options = json.loads(tiles.options)
        options.update({'opacity': opacity})
        tiles.options = json.dumps(options, sort_keys=True, indent=2)
        tiles._template = jinja2.Template(u"""
        {% macro script(this, kwargs) %}
            var {{this.get_name()}} = L.tileLayer(
                '{{this.tiles}}',
                {{ this.options }}
                ).addTo({{this._parent.get_name()}});
        {% endmacro %}
        """)

    tiles.add_to(m)
    if style_function is not None:
        gj = folium.GeoJson(geojson_to_overlay, overlay=True, name=layer_name, style_function=style_function)
    else:
        gj = folium.GeoJson(geojson_to_overlay, overlay=True, name=layer_name)
    if tooltip_props is not None:
        folium.features.GeoJsonTooltip(tooltip_props, aliases=tooltip_aliases).add_to(gj)
    gj.add_to(m)

    if show_layer_control is True:
        folium.LayerControl().add_to(m)

    return m


def rgb2hex(rgb):
    hex_val = "#{:02x}{:02x}{:02x}".format(*rgb)
    return hex_val


def plot_array(array, subplot_ijk, title="", font_size=18, cmap=None):
    sp = plt.subplot(*subplot_ijk)
    sp.set_title(title, fontsize=font_size)
    plt.axis('off')
    plt.imshow(array, cmap=cmap)


def imprint_geom(chip, geom):
    outlines = geom_to_array(geom, chip, all_touched=False, exterior_only=True)
    rgb_w_outlines = chip.rgb().copy()

    r = rgb_w_outlines[:, :, 0]
    r[outlines == 1] = 255

    g = rgb_w_outlines[:, :, 1]
    g[outlines == 1] = 13

    b = rgb_w_outlines[:, :, 2]
    b[outlines == 1] = 255

    return rgb_w_outlines


def display_chip(feat, img, geom_buffer_size=0.0003, title=""):
    geom = feat['geometry']
    bounds = geom.buffer(geom_buffer_size).bounds
    chip = img.aoi(bbox=bounds, pansharpen=True)
    rgb_imprinted = imprint_geom(chip, geom)
    sp = plt.figure(figsize=(5, 5))
    plot_array(rgb_imprinted, (1, 1, 1), title=title)
    _ = plt.plot()

