import numpy as np
import pandas as pd
import numpy.polynomial.polynomial as poly



def form_fit(surface,degree=2):
    form = np.zeros(surface.shape)
    x = np.arange(0,surface.shape[1])
    y = np.arange(0,surface.shape[0])
    for j in range(surface.shape[1]):
        coefs = poly.polyfit(y, surface[:, j], degree)
        form[:, j] = poly.polyval(y, coefs)
    for i in range(surface.shape[0]):
        coefs = poly.polyfit(x, surface[i,:], degree)
        form[i,:] =  poly.polyval(x, coefs)
    return form


def Height_Parameters(surface,Wsurface,Rsurface):

    """ Compute Height Parameters Defined by ISO for Scale_limited_surface """

    #surface = scale_limited_surface

    A = surface.shape[0] * surface.shape[1]

    Sa = np.sqrt((1 / A) * np.abs(np.sum(surface[:, :])))
    WSa = np.sqrt((1 / A) * np.abs(np.sum(Wsurface[:, :])))
    RSa = np.sqrt((1 / A) * np.abs(np.sum(Rsurface[:, :])))
    Sa_info = 'arithmetical mean height'

    Sq = np.sqrt((1 / A) * np.sum(surface[:, :] ** 2))
    WSq = np.sqrt((1 / A) * np.sum(Wsurface[:, :] ** 2))
    RSq = np.sqrt((1 / A) * np.sum(Rsurface[:, :] ** 2))
    Sq_info = 'root mean square height'

    Ssk = 1 / Sq ** 3 * ((1 / A) * np.sum(surface[:, :] ** 3))
    WSsk = 1 / Sq ** 3 * ((1 / A) * np.sum(Wsurface[:, :] ** 3))
    RSsk = 1 / Sq ** 3 * ((1 / A) * np.sum(Rsurface[:, :] ** 3))
    Ssk_info = 'skewness'

    Sku = 1 / Sq ** 4 * ((1 / A) * np.sum(surface[:, :] ** 4))
    WSku = 1 / Sq ** 4 * ((1 / A) * np.sum(Wsurface[:, :] ** 4))
    RSku = 1 / Sq ** 4 * ((1 / A) * np.sum(Rsurface[:, :] ** 4))
    Sku_info = 'kurtosis'

    Sp = np.max(surface)
    WSp = np.max(Wsurface)
    RSp = np.max(Rsurface)
    Sp_info = 'maximum peak height'

    Sv = abs(np.min(surface))
    WSv = abs(np.min(surface))
    RSv = abs(np.min(surface))
    Sv_info = 'maximum pit height'

    Sz = Sp + Sv
    WSz = WSp + WSv
    RSz = RSp + RSv
    Sz_info = 'maximum height'


    Height_data = pd.DataFrame({'Parameters': ['Sa','Sq','Ssk','Sku','Sp','Sv','Sz'],
                               'SF Surface': [Sa, Sq, Ssk, Sku, Sp, Sv, Sz],
                                'SF Surface (waveiness)': [WSa, WSq, WSsk, WSku, WSp, WSv, WSz],
                                'SL Surface (roughness)': [RSa, RSq, RSsk, RSku, RSp, RSv, RSz],
                               'Description': [Sa_info, Sq_info, Ssk_info, Sku_info, Sp_info, Sv_info, Sz_info]
                               })


    return Height_data


def Spatial_Parameters(scale_limited_surface):

    """ Compute Areal Spatial Parameters Defined by ISO """

    surface = scale_limited_surface

    A = surface.shape[0] * surface.shape[1]

    Sal = 'To do'
    Sal_info = 'autocorrelation length'

    Str = 'To do'
    Str_info = 'texture aspect ratio'

    Spatial_data = pd.DataFrame({'Areal Spatial Parameters': ['Sal','Str'],
                               'Values': [Sal,Str],
                               'Description': [Sal_info, Str_info]
                               })

    return Spatial_data


def Hybrid_Parameters(scale_limited_surface):
    """ Compute Areal Spatial Parameters Defined by ISO """

    surface = scale_limited_surface

    A = surface.shape[0] * surface.shape[1]

    Sdq = 'To do'
    Sdq_info = 'root mean square gradient'

    Sdr = 'To do'
    Sdr_info = 'developed interfacial area ratio'

    Hybrid_data = pd.DataFrame({'Areal Hybrid Parameters': ['Sdq', 'Sdr'],
                                 'Values': [Sdq, Sdr],
                                 'Description': [Sdq_info, Sdr_info]
                                 })

    return Hybrid_data