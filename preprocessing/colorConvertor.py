# small file to convert colours from rgb to hsv
import numpy as np
import pandas as pd
import colorsys

def main():

    # load data
    dataset = '/home/markg/Documents/TCD/ML/ML1819--task-107--team-11/dataset/gender-classifier-DFE-791531.csv'
    data = pd.read_csv(dataset, encoding='latin-1')

    H, S, V = [], [], []
    linkColorList = data['link_color']
    for color in linkColorList:
        try:
            color = list(int(color[i:i+2], 16) for i in (0, 2 ,4))
        except ValueError:
            color = [0, 0, 0]
        # must divide by 255 as co-ordinates are in range 0 to 1
        hsv = colorsys.rgb_to_hsv(color[0]/255, color[1]/255, color[2]/255)

        # rescale to hsv
        x = round(hsv[0]*360, 1)
        y = round(hsv[1]*100, 1)
        z = round(hsv[2]*100, 1)
        H.append(x)
        S.append(y)
        V.append(z)

    data['link_hue'] = H
    data['link_sat'] = S
    data['link_vue'] = V

if __name__=='__main__':
    main()
