import cv2
import cmapy
import numpy as np
import pandas as pd
import altair as alt
from PIL import Image
import ISO_Parameters
import streamlit as st
from scipy.ndimage import gaussian_filter

# Running Streamlit app from local host
# cmd --> streamlit run XXX.py

# ---- Platforms -----------------#
# https://dashboard.heroku.com/apps
# https://github.com/Jesse-Redford/gps-surface-texture
# https://www.streamlit.io/sharing-sign-up

#----- Deploy app using heroku -------#
# cd C:\Users\Jesse\Desktop
# git clone https://github.com/Jesse-Redford/gps-surface-texture.git
# cd C:\Users\Jesse\Desktop\gps-surface-texture
# heroku login

#---------- OPTION ONE------------------#
# heroku create (add name here, otherwise one will be automatically generated)
# git add .
# git commit -m "first commit"
# git push heroku main <or> git push heroku HEAD:master

#-------- OPTION TWO -------------------#
# heroku git:remote -a iso-app
# heroku git:remote -a iso-app
# git add .
# git commit -am "commit message"
# heroku ps:scale web=1
# heroku open

# ------- ISO refrences --------------#
# https://www.iso.org/obp/ui/#iso:std:iso:25178:-2:ed-1:v1:en
# https://guide.digitalsurf.com/en/guide-metrology-videos.html
# https://guide.digitalsurf.com/en/guide-filtration-techniques.html

#----- Tutorials Resources ------------#
# https://medium.com/analytics-vidhya/deploying-a-streamlit-and-opencv-based-web-application-to-heroku-456691d28c41 <-- opencv
# https://www.youtube.com/watch?v=mQ7rGcE766k
# https://towardsdatascience.com/from-streamlit-to-heroku-62a655b7319
# https://www.youtube.com/watch?v=skpiLtEN3yk
# https://stackoverflow.com/questions/26595874/i-want-make-push-and-get-error-src-refspec-master-does-not-match-any

#-------- Misc Troubleshooting --------------#
# https://stackoverflow.com/questions/20003290/output-different-precision-by-column-with-pandas-dataframe-to-csv

st.set_page_config(layout="wide")
st.title('Surface Measurment App')
st.subheader('Author: Jesse Redford')

uploaded_file = st.file_uploader("Upload a surface image", type=([".png",".jpg"]))
col1, col2, col3,col4, col5 = st.beta_columns(5)

if uploaded_file is not None:
    uploaded_image = Image.open(uploaded_file).convert('LA')
    #st.image(uploaded_image, caption='Uploaded Image.', use_column_width=True)

    st.sidebar.header('Inspection Parameters')
    surface = np.asarray(uploaded_image)[:, :, 0]
    prof_pos = st.sidebar.slider('1D Profile Position', 0, surface.shape[0])


    st.sidebar.header('Processing Parameters')
    s1 = st.sidebar.slider('S Filter (Low Pass) with s1 Nesting Index', 0, 30)
    poly_deg = st.sidebar.slider('F Operator (poly degree)',0,10)
    s2 = st.sidebar.slider('S Filter (gaussian low pass, s2 nesting index)',0,30)
    l = st.sidebar.slider('L Filter (gaussian high pass, nesting index)',0,30)



    primary_surface = gaussian_filter(surface, sigma=s1); primary_surface = primary_surface - np.mean(surface)
    surface_form = ISO_Parameters.form_fit(primary_surface,degree=poly_deg)
    SF_surface = primary_surface - surface_form
    SF_wave_surface = gaussian_filter(SF_surface, sigma=s2)
    SL_surface = SF_wave_surface - gaussian_filter(SF_wave_surface, sigma=l)
    #SL_surface = gaussian_filter(SF_surface - SF_wave_surface, sigma=l)


    df = ISO_Parameters.Height_Parameters(SF_surface, SF_wave_surface, SL_surface)
    df.index = [""] * len(df)


    df['SF Surface'] = df['SF Surface'].map(lambda x: '{0:.1}'.format(x))
    df['SF Surface (waveiness)'] = df['SF Surface (waveiness)'].map(lambda x: '{0:.1}'.format(x))
    df['SL Surface (roughness)'] = df['SL Surface (roughness)'].map(lambda x: '{0:.1}'.format(x))



    wide_form = pd.DataFrame({'Sample Length': np.arange(surface.shape[1]),
                              #'surface': surface[50,:],
                              'Primary Surface Form': surface_form[prof_pos,:],
                              'Primary Surface': primary_surface[prof_pos,:],
                              'SF Surface (Form Removed)': SF_surface[prof_pos, :],
                              'SF Surface (Waviness)': SF_wave_surface[prof_pos, :],
                              'SL Surface (Roughness)': SL_surface[prof_pos, :]
                              })
    data = wide_form.melt('Sample Length', var_name='type', value_name='height')

    input_dropdown = alt.binding_select(options=['Primary Surface', 'Primary Surface Form', 'SF Surface (Form Removed)', 'SF Surface (Waviness)','SL Surface (Roughness)',
                                                 ['Primary Surface', 'SF Surface (Form Removed)', 'SF Surface (Waviness)','SL Surface (Roughness)']])
    selection = alt.selection_single(fields=['type'], bind=input_dropdown, name='SurfaceType')
    color = alt.condition(selection,alt.Color('type:N', legend=None),alt.value('lightgray'))


    a = alt.Chart(data).mark_area(opacity=0.3).encode(x='Sample Length', y='height',color='type:N', tooltip='type:N').add_selection(selection).transform_filter(selection)

    #c = alt.layer(a, b)

    st.header('1D Profile')
    st.altair_chart(a, use_container_width=True)

    cmap = 'seismic_r'

    #surface = cv2.normalize(surface, surface, 0, 255, norm_type=cv2.NORM_MINMAX)
    primary_surface =  cv2.normalize(primary_surface, primary_surface, 0, 255, norm_type=cv2.NORM_MINMAX)
    SF_surface = cv2.normalize(SF_surface, SF_surface, 0, 255, norm_type=cv2.NORM_MINMAX)
    SF_wave_surface = cv2.normalize(SF_wave_surface, SF_wave_surface, 0, 255, norm_type=cv2.NORM_MINMAX)
    SL_surface = cv2.normalize(SL_surface, SL_surface, 0, 255, norm_type=cv2.NORM_MINMAX)



    surface = cv2.applyColorMap(surface.astype('uint8'), cmapy.cmap(cmap))
    primary_surface = cv2.applyColorMap(primary_surface.astype('uint8'), cmapy.cmap(cmap))
    SF_surface = cv2.applyColorMap(SF_surface.astype('uint8'), cmapy.cmap(cmap))
    SF_wave_surface = cv2.applyColorMap(SF_wave_surface.astype('uint8'), cmapy.cmap(cmap))
    SL_surface = cv2.applyColorMap(SL_surface.astype('uint8'), cmapy.cmap(cmap))

    surface= cv2.line(surface.astype('uint8'), (0,prof_pos), (surface.shape[1],prof_pos), (0,0,0), 3)
    #surface = cv2.putText(surface, '1D Profile', (int(surface.shape[1]/4), prof_pos), cv2.FONT_HERSHEY_SIMPLEX ,surface.shape[1]/500, (0,0,0), int(surface.shape[1]/500), cv2.LINE_AA)


    col1.image(surface, caption='Extracted Surface = S', use_column_width=True)
    col2.image(primary_surface, caption='Primary Surface = lowpass(S)-mean(lowpass(S))', use_column_width=True)
    col3.image(SF_surface, caption='SF Surface (Primary - Surface Form) ', use_column_width=True)
    col4.image(SF_wave_surface, caption='SF Surface = waviness = lowpass(SF)', use_column_width=True)
    col5.image(SL_surface, caption='SL Surface = roughness = SF - lowpass(SF)', use_column_width=True)


    st.header('Areal Height Parameters ISO 25178-2-2012')
    st.table(df)
    #st.header('Spatial Parameters')
    #st.write(ISO_Parameters.Spatial_Parameters(SF_surface))
    #st.header('Hybrid Parameters')
    #st.write(ISO_Parameters.Hybrid_Parameters(SF_surface))
    #st.header('Functions and Related Parameters')



    #ref = primary_surface
    #ref[x:x+5,:] = 0
    #col3.image(ref, caption='Secondary Surface.', use_column_width=True)
    #value = st.sidebar.selectbox("Hello", ["one", "two", "three"])
    #st.sidebar.area_chart(surface[x, :])





