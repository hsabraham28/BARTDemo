#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
import pandas as pd
import numpy as np


# In[3]:

st.title('OCC Dashboard')
#st.write(stationlocs)
st.header('Upload a log:')
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
     # Can be used wherever a "file-like" object is accepted:
     dataframe = pd.read_csv(uploaded_file).head(60)
     #st.write(dataframe)
    
def swimlanevega(testlog):
    stationlocs = pd.read_csv("assets/BARTLocationPercentages.csv")
    station_list = (stationlocs['Station_Name']).tolist()
    s_abbrv = [station[:3] for station in station_list]
    def lookfornames(c):
        for s in range(len(s_abbrv)):
            if s_abbrv[s] in c:
                return station_list[s]
        
    testlog['Location'] = testlog["Log"].apply(lookfornames)
    justxandy = stationlocs.drop(columns=['Rain_Critical', 'Asset_Location'])
    merged = testlog.merge(justxandy, left_on='Location', right_on='Station_Name')
    merged['dupped'] = merged.duplicated(subset=['Log'])
    merged = merged[merged.dupped != True]
    vizdata = merged.drop(columns=['dupped'])
    from tensorflow import keras
    model = keras.models.load_model('occ-log-classification/model')
    class_legend = {
    0:"Misc",
    1:'Other BPD',
    2:'Homeless',
    3:'Medical',
    4:'Patron interference',
    5:"Vehicle failure",
    6:"Wayside equipment",
    7:"Software related failures",
    8:"Human Error",
    9:"Weather",
    10:"Info (no error)",
    11:"Delays",
    12:"Track obstruction",
    13:"Schedule maintenance"
    }
    vizdata['Cat_Pred'] = vizdata['Log'].apply(lambda log: np.argmax(model.predict([log]), axis=1)[0])
    vizdata['Class_Pred'] = vizdata['Cat_Pred'].apply(lambda code: class_legend[code])
    catvizdata = vizdata.drop(columns=['X_Percentage', 'Y_Percentage'])
    refs = []
    refs = np.random.choice(np.arange(len(catvizdata['Location'])), len(catvizdata['Location']), replace=False)
    catvizdata['Reference'] = refs
    import datetime
    tim = []
    catvizdata['TimeOG'] = catvizdata['Time']
    for t in catvizdata['Time']:
        t = int(t)
        #temp = datetime.time(t//100, t%100)
        tim.append(str(t//100) + ":" + str(t%100))
    catvizdata['Time Str'] = tim



    timarr = []
    for t in catvizdata['Time']:
        t = int(t)
        temp = datetime.time(t//100, t%100)
        timarr.append(temp)
    catvizdata['Time'] = timarr
   
    import altair as alt
    import json
    from json import dumps
    timarray = []
    for t in catvizdata['Time']:
        timarray.append(json.dumps(t, indent=4, sort_keys=True, default=str))
    for x in range(len(timarray)):
        timarray[x] = timarray[x][1:9]
    catvizdata['Time J'] = timarray
    catvizdatav = catvizdata.drop('Time', axis=1)
    ugh = catvizdatav.groupby('Time J').count()
    catvizdatav['counted'] = ugh[['Log']].reset_index()["Log"]
    catvizdatav['Timestamps'] = pd.to_datetime(timarray)
    catvizdatav['Times'] = catvizdata['Time J']
    class_icons = {
    "Misc":'icons/misc.png',
    'Other BPD':'icons/police.png',
    'Homeless':'icons/homeless.png',
    'Medical':'icons/medical.png',
    'Patron interference':'icons/patronInt.png',
    "Vehicle failure":'icons/vehicleFail.png',
    "Wayside equipment":'icons/waysideEquip.png',
    "Software related failures":'icons/software.png',
    "Human Error":'icons/humanErr.png',
    "Weather":'icons/weather.png',
    "Info (no error)": 'icons/info.png',
    "Delays": 'icons/delays.png',
    "Track obstruction":'icons/obstruction.png',
    "Schedule maintenance":'icons/maintenance.png'
    }
    catvizdatav['img'] = catvizdatav['Class_Pred'].apply(lambda code: class_icons[code])
    
    sourcetwo = catvizdatav
    swimchart = alt.Chart(sourcetwo).mark_circle().encode(
    alt.X('Times:O',
        scale=alt.Scale(zero=False), title='Timestamps'
    ),
    y=alt.Y('Class_Pred:O', axis=alt.Axis(title='Incident Type')),
    tooltip= ['Log', 'Location', 'Times'],
    #url="img"
    color='Class_Pred'
    
    )
    return swimchart



def geoviz(testlog):
    import plotly.graph_objects as go
    import base64
    import plotly.express as px
    stationlocs = pd.read_csv("assets/BARTLocationPercentages.csv")
    station_list = (stationlocs['Station_Name']).tolist()
    s_abbrv = [station[:3] for station in station_list]
    def lookfornames(c):
        for s in range(len(s_abbrv)):
            if s_abbrv[s] in c:
                return station_list[s]
        
    testlog['Location'] = testlog["Log"].apply(lookfornames)
    justxandy = stationlocs.drop(columns=['Rain_Critical', 'Asset_Location'])
    merged = testlog.merge(justxandy, left_on='Location', right_on='Station_Name')
    merged['dupped'] = merged.duplicated(subset=['Log'])
    merged = merged[merged.dupped != True]
    vizdata = merged.drop(columns=['dupped'])
    from tensorflow import keras
    model = keras.models.load_model('occ-log-classification/model')
    class_legend = {
    0:"Misc",
    1:'Other BPD',
    2:'Homeless',
    3:'Medical',
    4:'Patron interference',
    5:"Vehicle failure",
    6:"Wayside equipment",
    7:"Software related failures",
    8:"Human Error",
    9:"Weather",
    10:"Info (no error)",
    11:"Delays",
    12:"Track obstruction",
    13:"Schedule maintenance"
    }
    vizdata['Cat_Pred'] = vizdata['Log'].apply(lambda log: np.argmax(model.predict([log]), axis=1)[0])
    vizdata['Class_Pred'] = vizdata['Cat_Pred'].apply(lambda code: class_legend[code])
    maptable = vizdata
    maptable["xplot"] = (1200* (maptable['X_Percentage']/100)) + np.random.normal(0, 10, size=len(maptable['X_Percentage']))
    maptable["yplot"] = (1080* (maptable['Y_Percentage']/100)) + np.random.normal(0, 10, size=len(maptable['X_Percentage']))
    frametable = maptable[["xplot", "yplot"]]
    xvals = maptable['xplot']
    yvals = maptable['yplot']

    mapfig = go.Figure()
    img_width = 1200
    img_height = 1080
    mapfig.update_layout(
        autosize=False
    )

    bartimage = base64.b64encode(open("assets/BARTtracksmap.png", 'rb').read())
    mapfig.add_layout_image(
            dict(
                source='data:image/png;base64,{}'.format(bartimage.decode()), #"./assets/BARTtracksmap.png",
                xref="x",
                yref="y",
                x=0,
                y=0,
                sizex=img_width,
                sizey=img_height,
                #sizing="stretch",
                opacity=1,
                sizing='contain',
                layer="below",xanchor="left", yanchor="top")
    )
    mapfig.update_xaxes(showgrid=False, range=(0, img_width))
    mapfig.update_yaxes(showgrid=False, scaleanchor='x', range=(img_height, 0))

    mapfig.add_trace(
        go.Scatter(x=maptable["xplot"], y=maptable["yplot"], mode="markers", text=maptable['Log'])

    )

    
    return mapfig



def vegageo():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from shapely.geometry import Point # to compute projection onto line
    from shapely.geometry import LineString
    import json
    real_df = pd.read_csv('BART_Data/AerialStructuresAndTrainControl.csv')
    trail_map = plt.imread('BART_Data/BART-tracks-dashboard-map.png')
    bart_json = []

    with open('BART_Data/bart_json.txt', 'r') as json_data:
        #make a loop to read file
        line = json_data.readline()
        
        while line:
            status = json.loads(line)
            
            # extract variable 
            station_name = status['name']
            stattion_latitude = status['gtfs_latitude']
            stattion_longitude = status['gtfs_longitude']
            
            # make a dictionary
            json_file = {'STATION_NAME': station_name, 
                         'world_latitude': stattion_latitude, 
                         'world_longitude': stattion_longitude
                        }
            bart_json.append(json_file)
            
            # read next line
            line = json_data.readline()
            
    #convert the dictionary list to a df
    df_json = pd.DataFrame(bart_json, columns = ['STATION_NAME', 'world_latitude', 'world_longitude'])

    df_json['world_latitude'] = df_json['world_latitude'].astype(float)
    df_json['world_longitude'] = df_json['world_longitude'].astype(float)
    import gmaps
    import gmaps.datasets
    google_key = "AIzaSyCOjkkBKCKVRt94C1wHef0I4fnoh_-CXvA"
    gmaps.configure(api_key=google_key) 
    real_location = real_df[['Latitude', 'Longitude']]
    bart_location = df_json[['world_latitude', 'world_longitude']]
    trail_layer = gmaps.symbol_layer(
        real_location, stroke_color="yellow", scale=2
    )
    bart_layer = gmaps.symbol_layer(
        bart_location, fill_color="purple", stroke_color="purple", scale=2
    )
    fig = gmaps.figure()
    fig.add_layer(trail_layer)
    fig.add_layer(bart_layer)
    import matplotlib.pyplot as plt
    import numpy as np
    import mplcursors
    #%matplotlib inline
    #%matplotlib nbagg

    fig = plt.imshow(trail_map)
    mplcursors.cursor()  

    plt.title("Collect the station coordinate on the graph.")
    graph_df = pd.DataFrame({'STATION': ['A10', 'A20', 'A30', 'A40', 'A50', 'A60', 'A70', 'A80', 'A90',
       'C10', 'C20', 'C30', 'C40', 'C50', 'C60', 'C70', 'C80',
       'K10', 'K20', 'K30', 'L10', 'L20', 'L30', 'M10', 'M16', 'M20',
       'M30', 'M40', 'M50', 'M60', 'M70', 'M80', 'M90', 'R10', 'R20',
       'R30', 'R40', 'R50', 'R60', 'S20', 'S40', 'S50', 'W10', 'W20',
       'W30', 'W40', 'Y10', 'E20', 'E30'],
                   'x_coordinate': [509, 541, 571, 604, 636, 698, 739, 785, 828, #A10-A90
                                    509, 538, 568, 597, 624, 653, 716, 801, #C10-C80
                                    472, 472, 472, #K10-K30
                                    762, 883, 966, #L10-L30
                                    430, 308, 288, 265, 246, 226, 226, 226, 226, 246, #M10-M90
                                    436, 407, 377, 347, 321, 292, #R10-R60
                                    860, 860, 860, #S20-S50
                                    275, 305, 334, 370, #W10-W40
                                    416, #Y10
                                    888, 957 #E20-E30 (NOT IN station_names_BART.csv)
                                   ],
                   'y_coordinate': [500, 535, 566, 597, 627, 688, 731, 778, 821, #A10-A90
                                    295, 265, 236, 208, 178, 148, 125, 127, #C10-C80
                                    420, 389, 362, #K10-K30
                                    648, 648, 648, #L10-L30
                                    464, 490, 509, 530, 553, 607, 638, 666, 696, 749, #M10-M90
                                    295, 265, 239, 208, 178, 148, #R10-R60
                                    887, 931, 974, #S20-S50
                                    777, 806, 836, 867, #W10-W40
                                    846, #Y10
                                    127, 127 #E20-E30 (NOT IN station_names_BART.csv)
                                   ]})
    station_df = pd.read_csv("BART_Data/station_names_BART.csv")
    station_df_updated = station_df.copy()
    station_df_updated.loc['48',:]=['E20', 'Pittsburg Center'] 
    station_df_updated.loc['49',:]=['E30', 'Antioch'] 
    station_df_updated.query("STATION == 'C88'")
    station_df_updated = station_df_updated.drop([17])
    station_df_updated = station_df_updated.reset_index().drop(['index'], 1)
    draw_points = []
    for i in range(len(graph_df)):
        draw_points.append([graph_df['x_coordinate'][i], graph_df['y_coordinate'][i]])

    import matplotlib.image as mpimg
    # %matplotlib widget
    #%matplotlib inline
    #%matplotlib nbagg

    image = mpimg.imread("BART_Data/BART-tracks-dashboard-map.png")

    pts = np.array(draw_points)
    combine_df = pd.merge(graph_df, station_df_updated, how="left", on="STATION")
    diagram_world_df = pd.merge(combine_df, df_json, how="left", on="STATION_NAME")
    diagram_world_df = diagram_world_df.rename(columns = {'STATION' : 'station', 'x_coordinate' : 'diagram_x', 'y_coordinate' : 'diagram_y',
                                    'STATION_NAME' : 'description'})

    # plt.imshow(image)
    # plt.scatter(pts[:, 0], pts[:, 1], marker="o", color="purple", s=8)


    graph_df = graph_df.append({'STATION': 'TEMP', 'x_coordinate': 366, 'y_coordinate':466 }, ignore_index=True)

    groups = {
    'groupA': ['A10',
     'A20',
     'A30',
     'A40',
     'A50',
     'A60',
     'A70',
     'A80',
     'A90'],
    'groupC': ['C10',
     'C20',
     'C30',
     'C40',
     'C50',
     'C60'],
    'groupK': ['K10',
     'K20',
     'K30',],
    'groupL': ['L10',
     'L20',
     'L30',],
    'groupM1': ['M10',
     'TEMP'],
    'groupM2': ['M16',
     'M20',
     'M30',
     'M40'],
    'groupM3': ['M50',
     'M60',
     'M70',
     'M80'],
    'groupR': ['R10',
     'R20',
     'R30',
     'R40',
     'R50',
     'R60',],
    'groupS': ['S20',
     'S40',
     'S50',],
    'groupW': ['M90',
     'W10',
     'W20',
     'W30'],
    'groupE': [
     'C70',
     'C80',
     'E20',
     'E30'],
    'groupother': ['Y10', 'W40']}

    world_df = diagram_world_df[['station', 'world_latitude', 'world_longitude']]


    def find_two_closest_bart(input_point, world_df):
        in_lat, in_lon = input_point
        dists = []
        for idx, row in world_df.iterrows():
            station_name, lat, lon = row
            dists.append(((in_lat-lat)**2 + (in_lon-lon)**2) ** (1/2))
        temp_arr = np.array(dists)
        closest_idx = temp_arr.argmin()  # return the indices of the minimum values
        
        closest_dist = temp_arr.min()
        temp_arr = np.delete(temp_arr, closest_idx)  
        sec_closest_idx = temp_arr.argmin()
        temp_df = world_df.copy().drop(closest_idx).reset_index(drop=True)

        return (world_df.loc[closest_idx].station, temp_df.loc[sec_closest_idx].station), closest_dist


    def dist_from_close_diagram(two_close_point, world_df, real_dist, graph_df):
    # dist on real world between two station
        id_1, id_2 = two_close_point
        station_name_1, lat_1, lon_1 = world_df.query(f"station == '{id_1}'").values.tolist()[0]
        station_name_2, lat_2, lon_2 = world_df.query(f"station == '{id_2}'").values.tolist()[0]
        real_whole_dist = (((lat_1-lat_2)**2 + (lon_1-lon_2)**2) ** (1/2))
        
        # dist on diagram between two station
        station_name_3, x_3, y_3 = graph_df.query(f"STATION == '{id_1}'").values.tolist()[0]
        station_name_4, x_4, y_4 = graph_df.query(f"STATION == '{id_2}'").values.tolist()[0]
        diagram_whole_dist = (((x_3-x_4)**2 + (y_3-y_4)**2) ** (1/2))
        diagram_dist = (real_dist * diagram_whole_dist) / real_whole_dist
        
        # Math explanation: @https://math.stackexchange.com/questions/175896/finding-a-point-along-a-line-a-certain-distance-away-from-another-point
        # get the projected point location
        ratio = diagram_dist / diagram_whole_dist 
        x = (1 - ratio) * x_3 + ratio * x_4
        y = (1 - ratio) * y_3 + ratio * y_4
        return (x, y)


    def get_projected_loc(input_point):
        two_point, real_dist = find_two_closest_bart(input_point, world_df)
        loc = dist_from_close_diagram(two_point, world_df, real_dist, graph_df)
        return loc



    def draw_projected_point(input_point):
        projected_loc = get_projected_loc(input_point)
        print(projected_loc)
        
        #%matplotlib inline
        #%matplotlib nbagg

        image = mpimg.imread("BART-tracks-dashboard-map.png")

        plt.figure(figsize=(12,12))
        # trial_pts = np.array(trial_point)

        plt.imshow(image)
        # plt.scatter(trial_pts[0][0], trial_pts[0][1], marker="v", color="green", s=20)
        plt.scatter(projected_loc[0], projected_loc[1], marker="v", color="green", s=100)
        plt.scatter(pts[:, 0], pts[:, 1], marker="o", color="purple", s=8)
        plt.show()

    project_points = []

    for i in range(len(real_df)):
        lat = real_df[['Latitude', 'Longitude']].values.tolist()[i][0]
        lon = real_df[['Latitude', 'Longitude']].values.tolist()[i][1]
        project_loc = get_projected_loc((lat, lon))
        
        project_points.append([project_loc[0], project_loc[1]])

    import matplotlib.image as mpimg
    #%matplotlib inline
    #%matplotlib nbagg

    image = mpimg.imread("BART-tracks-dashboard-map.png")

    plt.figure(figsize=(12,12))
    proj_pts = np.array(project_points)


    def find_group(station_id, groups):
        for key, value in groups.items():
            if station_id in value:
                return key
        return 'Not found'
    def find_group_startend(graph_df, group_name):
        start_point = groups[group_name][0] # C70
        end_point = groups[group_name][-1] # E30
        sp = graph_df.query(f"STATION=='{start_point}'")
        ep = graph_df.query(f"STATION=='{end_point}'")
        start_point = (sp.x_coordinate.item(), sp.y_coordinate.item())
        end_point = (ep.x_coordinate.item(), ep.y_coordinate.item())
        return start_point, end_point

    def project_to_line(start_point, end_point, target_point):
        point = Point(target_point)
        line = LineString([start_point, end_point])
        x = np.array(point.coords[0])
        u = np.array(line.coords[0])
        v = np.array(line.coords[len(line.coords)-1])
        n = v - u
        n /= np.linalg.norm(n, 2)
        P = u + n*np.dot(x - u, n)
        return P

    def project_target_to_map(input_point, target_point):
        c_station = find_two_closest_bart(input_point, world_df)[0][0]
        group_name = find_group(c_station, groups)
        start_point, end_point = find_group_startend(graph_df, group_name)
        p_p = project_to_line(start_point, end_point, target_point)
        return p_p
    project_points = []

    for i in range(len(real_df)):
        lat = real_df[['Latitude', 'Longitude']].values.tolist()[i][0]
        lon = real_df[['Latitude', 'Longitude']].values.tolist()[i][1]
        predict_loc = get_projected_loc((lat, lon))
        projected_loc = project_target_to_map((lat, lon), (predict_loc[0], predict_loc[1]))
        project_points.append([projected_loc[0], projected_loc[1]])

    import matplotlib.image as mpimg
    # %matplotlib inline
    # %matplotlib nbagg

    image = mpimg.imread("BART-tracks-dashboard-map.png")

    plt.figure(figsize=(12,12))
    proj_pts = np.array(project_points)





    output_df = pd.DataFrame(data={'x': proj_pts[:, 0], 'y': proj_pts[:, 1], 'Station_Name': real_df['Station_Name']})
    import plotly.graph_objects as go
    import plotly.express as px
    from skimage import io

    # load image
    img = io.imread('BART-tracks-dashboard-map.png')
    fig = px.imshow(img)

    # Add scatter
    fig.add_trace(
        go.Scatter(x=proj_pts[:, 0], y=proj_pts[:, 1], mode="markers"), 
    )
    jittered = proj_pts + np.random.normal(0, 10, size=proj_pts.shape)
    # load image
    img = io.imread('BART-tracks-dashboard-map.png')
    fig = px.imshow(img)

    # Add scatter
    fig.add_trace(
        go.Scatter(x=jittered[:, 0], y=jittered[:, 1], mode="markers"), 
    )
    # idea: fuse points that are within distance d of each other, O(n^2) code
    def dist2(p1, p2):
        ''' return squared dist between two points '''
        return (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2

    def fuse(points, d):
        ''' naive method to combine points that are within d distance by going through points
        and fusing the points. takes O(n^2) time'''
        ret = []
        d2 = d * d
        n = len(points)
        taken = [False] * n
        takenby = {} # keep track of points that are averaged in case we need later
        # dictionary is keys: averaged pt, values is list of points that were used
        for i in range(n):
            if not taken[i]:
                count = 1
                point = [points[i][0], points[i][1]]
                taken[i] = True

                pts_within_dist = []
                for j in range(i+1, n):
                    if dist2(points[i], points[j]) < d2:
                        point[0] += points[j][0]
                        point[1] += points[j][1]
                        count+=1
                        taken[j] = True
                        pts_within_dist.append(points[j])
                point[0] /= count
                point[1] /= count

                takenby[(point[0], point[1])] = pts_within_dist
                ret.append((point[0], point[1]))
        return np.array(ret), takenby

        fused_pts, _ = fuse(proj_pts, 5)

    # read in image
    # img = io.imread('BART-tracks-dashboard-map.png')
    # fig = px.imshow(img)

    # Add scatter
    fig.add_trace(
        go.Scatter(x=fused_pts[:, 0], y=fused_pts[:, 1], mode="markers"), 
    )

    fig.show()

st.markdown('<style>#vg-tooltip-element{z-index: 1000051}</style>',
             unsafe_allow_html=True)

st.header("Visualize by Category")
st.write(swimlanevega(dataframe))


st.header("Visualize by Location")
st.write(geoviz(dataframe))

#st.write(vegageo())

    # st.header("Visualize by Location - Improved")
    # st.write(vegageo(dataframe))





    # In[ ]:




