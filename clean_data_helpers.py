import numpy as np


def fill(image, threshold_dist=30):
    rows, cols = len(image), len(image[0])
    for u in range(rows):  # Iterate through rows
        for v in range(cols):  # Iterate through cols
            ltr_color, gtr_color, ltc_color, gtc_color = False, False, False, False
            for ltr in range(u, max(0, u-threshold_dist), -1):
                if image[ltr, v] != 0: 
                    ltr_color = image[ltr, v]
                    break
            for gtr in range(u, min(rows, u+threshold_dist)):
                if image[gtr, v] != 0: 
                    gtr_color = image[gtr, v]
                    break
            for ltc in range(v, max(0, v-threshold_dist), -1):
                if image[u, ltc] != 0: 
                    ltc_color = image[u, ltc]
                    break
            for gtc in range(v, min(cols, v+threshold_dist)):
                if image[u, gtc] != 0: 
                    gtc_color = image[u, gtc]
                    break
#             print([ltr_color, gtr_color, ltc_color, gtc_color])
            if np.all([ltr_color, gtr_color, ltc_color, gtc_color]):
                if len(set([ltr_color, gtr_color, ltc_color, gtc_color])) == 1:
                    image[u, v] = ltr_color
#               np.mean([ltr_color, gtr_color, ltc_color, gtc_color])
    return image