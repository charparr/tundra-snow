#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 09:40:52 2016

@author: cparr
"""

import itertools
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import numpy as np
import pandas as pd

fr_wind11 = pd.read_csv('/home/cparr/Snow_Patterns/wind_data/fr2011m.csv', header = 25)
rng = pd.date_range('1/1/2011', periods=8760, freq='H')
fr_wind11 = fr_wind11.set_index(rng)
fr_wind11['3m Wind Speed'] = fr_wind11['3 meter wind speed (m/s)']
fr_wind11['Wind Direction'] = fr_wind11['Wind direction TN']
fr_wind11['3m RH %'] = fr_wind11['3 meter relative humidity (%)']
fr_wind11 = fr_wind11[['Wind Direction','3m Wind Speed','3m RH %']]
fr_wind11_daily = pd.DataFrame()
fr_wind11_daily = fr_wind11.resample('D').mean()

fr_wind12 = pd.read_csv('/home/cparr/Snow_Patterns/wind_data/fr2012m.csv', header = 7)
fr_wind12 = fr_wind12[2::]
rng = pd.date_range('1/1/2012', periods=8784, freq='H')
fr_wind12 = fr_wind12.set_index(rng)
fr_wind12['3m Wind Speed'] = fr_wind12['WS_3m_Avg']
fr_wind12['Wind Direction'] = fr_wind12['WS_10m_WVc(2)']
fr_wind12 = fr_wind12[['Wind Direction','3m Wind Speed']]
fr_wind12['Wind Direction'] = fr_wind12['Wind Direction'].astype(float)
fr_wind12['3m Wind Speed'] = fr_wind12['3m Wind Speed'].astype(float)
fr_wind12_daily = pd.DataFrame()
fr_wind12_daily = fr_wind12.resample('D').mean()

fr_wind13 = pd.read_csv('/home/cparr/Snow_Patterns/wind_data/fr2013m.csv', header = 5)
fr_wind13 = fr_wind13[2::]
rng = pd.date_range('1/1/2013', periods=8760, freq='H')
fr_wind13 = fr_wind13.set_index(rng)
fr_wind13['3m Wind Speed'] = fr_wind13['WS_3m_Avg']
fr_wind13['Wind Direction'] = fr_wind13['WS_10m_WVc(2)']
fr_wind13 = fr_wind13[['Wind Direction','3m Wind Speed']]
fr_wind13['Wind Direction'] = fr_wind13['Wind Direction'].astype(float)
fr_wind13['3m Wind Speed'] = fr_wind13['3m Wind Speed'].astype(float)
fr_wind13_daily = pd.DataFrame()
fr_wind13_daily = fr_wind13.resample('D').mean()

fr_wind14 = pd.read_csv('/home/cparr/Snow_Patterns/wind_data/fr2014m.csv', header = 7)
fr_wind14 = fr_wind14[2::]
rng = pd.date_range('1/1/2014', periods=8760, freq='H')
fr_wind14 = fr_wind14.set_index(rng)
fr_wind14['3m Wind Speed'] = fr_wind14['WS_3m_Avg']
fr_wind14['Wind Direction'] = fr_wind14['WS_10m_WVc(2)']
fr_wind14 = fr_wind14[['Wind Direction','3m Wind Speed']]
fr_wind14['Wind Direction'] = fr_wind14['Wind Direction'].astype(float)
fr_wind14['3m Wind Speed'] = fr_wind14['3m Wind Speed'].astype(float)
fr_wind14_daily = pd.DataFrame()
fr_wind14_daily = fr_wind14.resample('D').mean()

fr_wind15 = pd.read_csv('/home/cparr/Snow_Patterns/wind_data/fr2015m.csv', header = 19)
rng = pd.date_range('1/1/2015', periods=14126, freq='H')
fr_wind15 = fr_wind15.set_index(rng)
fr_wind15['3m Wind Speed'] = fr_wind15['windSpeed_3m']
fr_wind15['Wind Direction'] = fr_wind15['winddir_10m']
fr_wind15 = fr_wind15[['Wind Direction','3m Wind Speed']]
fr_wind15['Wind Direction'] = fr_wind15['Wind Direction'].astype(float)
fr_wind15['3m Wind Speed'] = fr_wind15['3m Wind Speed'].astype(float)
fr_wind15.replace('6999.0', np.nan, inplace=True)
fr_wind15_daily = pd.DataFrame()
fr_wind15_daily = fr_wind15.resample('D').mean()


winter_11_12 = fr_wind11_daily.ix['9/1/2011'::].append(fr_wind12_daily.ix['1/1/2012':'5/1/2012'])
winter_12_13 = fr_wind12_daily.ix['9/1/2012'::].append(fr_wind13_daily.ix['1/1/2013':'5/1/2013'])
winter_13_14 = fr_wind13_daily.ix['9/1/2013'::].append(fr_wind14_daily.ix['1/1/2014':'5/1/2014'])
winter_14_15 = fr_wind14_daily.ix['9/1/2014'::].append(fr_wind15_daily.ix['1/1/2015':'5/1/2015'])

############
fig = plt.figure()
plt.title('Franklin Bluffs Winter 2011/2012 Daily Average Wind')
ax1 = fig.add_subplot(111)
ax1.plot(winter_11_12['Wind Direction'], 'r', lw = 2, alpha = 0.5)
plt.axhline(y=180, color='k', linestyle='--', alpha = 0.5)
plt.axhline(y=270, color='k', linestyle='--', alpha = 0.5)
plt.axhline(y=90, color='k', linestyle='--', alpha = 0.5)
ax1.set_ylabel('Wind Direction (deg. N.)', color='r')
ax1.set_ylim([0,360])
ax1.set_yticks([0,90,180,270,360])

ax2 = ax1.twinx()
ax2.set_xlabel([])
ax2.plot(winter_11_12['3m Wind Speed'], lw = 2)
ax2.set_xticklabels(['Sept', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May'])
ax2.set_ylabel('3m Wind Speed (m/s)', color='b')
ax2.set_ylim([5,13])

plt.savefig('/home/cparr/Snow_Patterns/figures/wind/winter_11_12.png',dpi = 300)

############

fig = plt.figure()
plt.title('Franklin Bluffs Winter 2012/2013 Daily Average Wind')
ax1 = fig.add_subplot(111)
ax1.plot(winter_12_13['Wind Direction'], 'r', lw = 2, alpha = 0.5)
plt.axhline(y=180, color='k', linestyle='--', alpha = 0.5)
plt.axhline(y=270, color='k', linestyle='--', alpha = 0.5)
plt.axhline(y=90, color='k', linestyle='--', alpha = 0.5)
ax1.set_ylabel('Wind Direction (deg. N.)', color='r')
ax1.set_ylim([0,360])
ax1.set_yticks([0,90,180,270,360])

ax2 = ax1.twinx()
ax2.set_xlabel([])
ax2.plot(winter_12_13['3m Wind Speed'], lw = 2)
ax2.set_xticklabels(['Sept', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May'])
ax2.set_ylabel('3m Wind Speed (m/s)', color='b')
ax2.set_ylim([5,19])
plt.savefig('/home/cparr/Snow_Patterns/figures/wind/winter_12_13.png',dpi = 300)

############

fig = plt.figure()
plt.title('Franklin Bluffs Winter 2013/2014 Daily Average Wind')
ax1 = fig.add_subplot(111)
ax1.plot(winter_13_14['Wind Direction'], 'r', lw = 2, alpha = 0.5)
plt.axhline(y=180, color='k', linestyle='--', alpha = 0.5)
plt.axhline(y=270, color='k', linestyle='--', alpha = 0.5)
plt.axhline(y=90, color='k', linestyle='--', alpha = 0.5)
ax1.set_ylabel('Wind Direction (deg. N.)', color='r')
ax1.set_ylim([0,360])
ax1.set_yticks([0,90,180,270,360])

ax2 = ax1.twinx()
ax2.set_xlabel([])
ax2.plot(winter_13_14['3m Wind Speed'], lw = 2)
ax2.set_xticklabels(['Sept', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May'])
ax2.set_ylabel('3m Wind Speed (m/s)', color='b')
ax2.set_ylim([5,19])
plt.savefig('/home/cparr/Snow_Patterns/figures/wind/winter_13_14.png',dpi = 300)

      
############

fig = plt.figure()
plt.title('Franklin Bluffs Winter 2014/2015 Daily Average Wind')
ax1 = fig.add_subplot(111)
ax1.plot(winter_14_15['Wind Direction'], 'r', lw = 2, alpha = 0.3)
plt.axhline(y=180, color='k', linestyle='--', alpha = 0.5)
plt.axhline(y=270, color='k', linestyle='--', alpha = 0.5)
plt.axhline(y=90, color='k', linestyle='--', alpha = 0.5)
ax1.set_ylabel('Wind Direction (deg. N.)', color='r')
ax1.set_ylim([0,360])
ax1.set_yticks([0,90,180,270,360])

ax2 = ax1.twinx()
ax2.set_xlabel([])
ax2.plot(winter_14_15['3m Wind Speed'], lw = 2)
ax2.set_xticklabels(['Sept', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May'])
ax2.set_ylabel('3m Wind Speed (m/s)', color='b')
ax2.set_ylim([5,18])
plt.savefig('/home/cparr/Snow_Patterns/figures/wind/winter_14.png',dpi = 300)


def main():
    azi = wd_11_12
    z = ws_11_12

    plt.figure(figsize=(5,6))
    plt.subplot(111, projection='polar')
    coll = rose(azi, z=z, bidirectional=True)
    plt.xticks(np.radians(range(0, 360, 45)), 
               ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
    plt.colorbar(coll, orientation='horizontal')
    plt.xlabel('2011 - 2012 3m Wind rose colored by mean wind speed')
    plt.rgrids(range(5, 20, 5), angle=290)

    plt.savefig('/home/cparr/Snow_Patterns/figures/wind/winter_11_12.png',dpi = 300)

    plt.show()
    
def rose(azimuths, z=None, ax=None, bins=36, bidirectional=False, 
         color_by=np.mean, **kwargs):
    """Create a "rose" diagram (a.k.a. circular histogram).  

    Parameters:
    -----------
        azimuths: sequence of numbers
            The observed azimuths in degrees.
        z: sequence of numbers (optional)
            A second, co-located variable to color the plotted rectangles by.
        ax: a matplotlib Axes (optional)
            The axes to plot on. Defaults to the current axes.
        bins: int or sequence of numbers (optional)
            The number of bins or a sequence of bin edges to use.
        bidirectional: boolean (optional)
            Whether or not to treat the observed azimuths as bi-directional
            measurements (i.e. if True, 0 and 180 are identical).
        color_by: function or string (optional)
            A function to reduce the binned z values with. Alternately, if the
            string "count" is passed in, the displayed bars will be colored by
            their y-value (the number of azimuths measurements in that bin).
        Additional keyword arguments are passed on to PatchCollection.

    Returns:
    --------
        A matplotlib PatchCollection
    """
    azimuths = np.asanyarray(azimuths)
    if color_by == 'count':
        z = np.ones_like(azimuths)
        color_by = np.sum
    if ax is None:
        ax = plt.gca()
    ax.set_theta_direction(-1)
    ax.set_theta_offset(np.radians(90))
    if bidirectional:
        other = azimuths + 180
        azimuths = np.concatenate([azimuths, other])
        if z is not None:
            z = np.concatenate([z, z])
    # Convert to 0-360, in case negative or >360 azimuths are passed in.
    azimuths[azimuths > 360] -= 360
    azimuths[azimuths < 0] += 360
    counts, edges = np.histogram(azimuths, range=[0, 360], bins=bins)
    if z is not None:
        idx = np.digitize(azimuths, edges)
        z = np.array([color_by(z[idx == i]) for i in range(1, idx.max() + 1)])
        z = np.ma.masked_invalid(z)
    edges = np.radians(edges)
    coll = colored_bar(edges[:-1], counts, z=z, width=np.diff(edges), 
                       ax=ax, **kwargs)
    return coll

def colored_bar(left, height, z=None, width=0.8, bottom=0, ax=None, **kwargs):
    """A bar plot colored by a scalar sequence."""
    if ax is None:
        ax = plt.gca()
    width = itertools.cycle(np.atleast_1d(width))
    bottom = itertools.cycle(np.atleast_1d(bottom))
    rects = []
    for x, y, h, w in zip(left, bottom, height, width):
        rects.append(Rectangle((x,y), w, h))
    coll = PatchCollection(rects, array=z, **kwargs)
    ax.add_collection(coll)
    ax.autoscale()
    return coll

if __name__ == '__main__':
    main()
    
