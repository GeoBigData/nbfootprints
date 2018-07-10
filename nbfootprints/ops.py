from __future__ import absolute_import
from __future__ import division
from past.utils import old_div
from shapely import geometry
from rasterio import features
import numpy as np
from skimage import filters
from ipywidgets import widgets, HBox
from IPython import display
import os
import json
import requests
import sys
if sys.version_info[0] == 3:
    from . import plots
else:
    import plots

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


def _catch_vote_and_advance(b, img, vote_question, buttons_widget, feature_list, result_list, geom_buffer_size=0.0003):
    # record the vote to the existing feature
    vote = convert_response_to_binary(b.description)
    result_list.append(vote)
    votes_tally = len(result_list)
    current_feature = feature_list[votes_tally - 1]
    current_feature['properties']['manual_label'] = vote

    # exit if no features remain
    if votes_tally == len(feature_list):
        b.close_all()

        # otherwise, advance to the next feature
    display.clear_output(wait=True)
    next_feature = feature_list[votes_tally]
    plots.display_chip(next_feature, img, geom_buffer_size, vote_question)
    display.display(buttons_widget)


def initialize_voting(feature_list, img, buttons_widget, vote_question, geom_buffer_size=0.0003):
    display.clear_output(wait=True)
    feat = feature_list[0]
    plots.display_chip(feat, img, geom_buffer_size, vote_question)
    display.display(buttons_widget)


def create_vote_buttons():
    button_yes = widgets.Button(description='Yes')
    button_no = widgets.Button(description='No')
    buttons = HBox([button_yes, button_no])

    return buttons


def add_button_callback(buttons, callback):
    for b in buttons.children:
        b.on_click(callback)

    return buttons
