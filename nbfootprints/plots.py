from . import ops.geom_to_array as geom_to_array
import jinja2
from gbdxtools import CatalogImage
import pandas as pd
import folium
import matplotlib.pyplot as plt
import json


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
TABLE_CSS = """<style>
.dataframe .ps__rail-y {
  width: 9px;
  background-color: transparent;
  opacity: 1 !important;
  right: 5px;
}

.dataframe .ps__rail-y::before {
  content: "";
  display: block;
  position: absolute;
  background-color: #ebebeb;
  border-radius: 5px;
  width: 100%;
  height: calc(100% - 30px);
  left: 0;
  top: 15px;
}

.dataframe .ps__rail-y .ps__thumb-y {
  width: 100%;
  right: 0;
  background-color: transparent;
  opacity: 1 !important;
}

.dataframe .ps__rail-y .ps__thumb-y::before {
  content: "";
  display: block;
  position: absolute;
  background-color: #cccccc;
  border-radius: 5px;
  width: 100%;
  height: calc(100% - 30px);
  left: 0;
  top: 15px;
}




/*//////////////////////////////////////////////////////////////////
[ Table ]*/
.dataframe {
  background-color: #fff;
	font-size: 14px;
	font-style: normal;
	font-variant: normal;
	font-weight: 300;
}

table {
  width: 100%;
}

th, td {
  font-weight: unset;
  padding-right: 10px;
}


.dataframe-head th {
  padding-top: 18px;
  padding-bottom: 18px;
}

.dataframe-body td {
  padding-top: 16px;
  padding-bottom: 16px;
	font-style: normal;
	font-variant: normal;
	font-weight: 300;
}

/*==================================================================
[ Fix header ]*/
.dataframe {
  position: relative;
  padding-top: 60px;
}

.dataframe-head {
  position: absolute;
  width: 100%;
  top: 0;
  left: 0;
}

.dataframe-body {
  max-height: 585px;
  overflow: auto;
}


/*==================================================================
[ Ver5 ]*/
.dataframe {
  margin-right: -30px;
}

.dataframe .dataframe-head {
  padding-right: 30px;
}

.dataframe th {
  color: #555555;
  line-height: 1.4;
  text-transform: uppercase;
  background-color: transparent;
}

.dataframe td {
  line-height: 1.4;

  background-color: #f7f7f7;

}

.dataframe .dataframe-body tr {
  overflow: hidden; 
  border-bottom: 10px solid #fff;
  border-radius: 10px;
}

.dataframe .dataframe-body table { 
  border-collapse: separate; 
  border-spacing: 0 10px; 
}
.dataframe .dataframe-body td {
    border: solid 1px transparent;
    border-style: solid none;
    padding-top: 10px;
    padding-bottom: 10px;
}
.dataframe .dataframe-body td:first-child {
    border-left-style: solid;
    border-top-left-radius: 10px; 
    border-bottom-left-radius: 10px;
}
.dataframe .dataframe-body td:last-child {
    border-right-style: solid;
    border-bottom-right-radius: 10px; 
    border-top-right-radius: 10px; 
}

.dataframe tr:hover td {
  background-color: #ebebeb;
  cursor: pointer;
}

.dataframe .dataframe-head th {
  padding-top: 25px;
  padding-bottom: 25px;
}

/*---------------------------------------------*/

.dataframe {
  overflow: hidden;
}

.dataframe .dataframe-body{
  padding-right: 30px;
}

.dataframe .ps__rail-y {
  right: 0px;
}

.dataframe .ps__rail-y::before {
  background-color: #ebebeb;
}

.dataframe .ps__rail-y .ps__thumb-y::before {
  background-color: #cccccc;
}
</style>"""


# FUNCTIONS
def footprints_outline_styler(x):
    return {'fillOpacity': .25,
            'color'      : COLORS['brightgreen'],
            'fillColor'  : COLORS['gray'],
            'weight'     : 1}


def footprints_color_styler(x):
    return {'fillColor'  : x['properties']['hexcolor'],
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
    if 'manual_label' not in x['properties'].keys():
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


def folium_map(geojson_to_overlay, layer_name, location, style_function=None, tiles='Stamen Terrain', zoom_start=19,
               show_layer_control=True, width='100%', height='75%', attr=None, map_zoom=18, max_zoom=20, tms=False,
               zoom_beyond_max=None):
    m = folium.Map(location=location, zoom_start=zoom_start, width=width, height=height, max_zoom=map_zoom)
    tiles = folium.TileLayer(tiles=tiles, attr=attr, name=attr, max_zoom=max_zoom)
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

    tiles.add_to(m)
    if style_function is not None:
        gj = folium.GeoJson(geojson_to_overlay, overlay=True, name=layer_name, style_function=style_function)
    else:
        gj = folium.GeoJson(geojson_to_overlay, overlay=True, name=layer_name)
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


def add_popups(features, m):
    for feature in features:
        lngs, lats = zip(*list(feature['geometry'].exterior.coords))
        locations = zip(lats, lngs)
        df = pd.DataFrame(zip(*[(k, v) for k, v in feature['properties'].iteritems()])).transpose()
        df.columns = ['attribute', 'value']
        df.set_index('attribute', inplace=True)
        html = df.to_html(header=False, float_format='{:,.2f}'.format, border=False)
        html = TABLE_CSS + html.replace('<tbody>', '<tbody class="dataframe-body">').replace('th', 'td')
        popup = folium.map.Popup(html=html, max_width=500)
        marker = folium.features.PolygonMarker(locations, color='white', weight=0, fill_color='white', fill_opacity=0,
                                               popup=popup)
        marker.add_to(m)

    return m


def get_idaho_tms_ids(image):
    ms_parts = {str(p['properties']['attributes']['idahoImageId']): str(
            p['properties']['attributes']['vendorDatasetIdentifier'].split(':')[1])
                for p in image._find_parts(image.cat_id, 'MS')}

    pan_parts = {str(p['properties']['attributes']['vendorDatasetIdentifier'].split(':')[1]): str(
            p['properties']['attributes']['idahoImageId'])
                 for p in image._find_parts(image.cat_id, 'pan')}

    ms_idaho_id = str(image.ipe.metadata['image']['imageId'])
    pan_idaho_id = pan_parts[ms_parts[ms_idaho_id]]

    idaho_ids = {'ms_id' : ms_idaho_id,
                 'pan_id': pan_idaho_id}
    return idaho_ids


def get_idaho_tms_url(source_catid_or_image, gbdx):
    if type(source_catid_or_image) == str:
        image = CatalogImage(source_catid_or_image)
    elif '_ipe_op' in source_catid_or_image.__dict__.keys():
        image = source_catid_or_image
    else:
        err = "Invalid type for source_catid_or_image. Must be either a Catalog ID (string) or CatalogImage object"
        raise TypeError(err)

    url_params = get_idaho_tms_ids(image)
    url_params['token'] = str(gbdx.gbdx_connection.access_token)
    url_params['z'] = '{z}'
    url_params['x'] = '{x}'
    url_params['y'] = '{y}'
    url_template = 'https://idaho.geobigdata.io/v1/tile/idaho-images/{ms_id}/{z}/{x}/{y}?bands=4,2,1&token={token}&panId={pan_id}'
    url = url_template.format(**url_params)

    return url

