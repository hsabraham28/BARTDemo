import matplotlib.pyplot as plt
import numpy as np

IMG_PATH = './BART-tracks-dashboard-map.png'
NUM_POINTS = 26 # number of points to create lines

def getpoints(img):
    plt.figure()
    plt.imshow(img)
    pts = plt.ginput(NUM_POINTS, timeout=500)
    pts = np.array(pts)
    line_pts = []
    for i in range(0,len(pts)-1,2):
        line_pts.append(pts[i:i+2])
    plt.close()
    plt.figure()
    plt.imshow(img)
    plt.plot(pts[:,0], pts[:,1], "r+")
    plt.show()
    return line_pts

if __name__ == "__main__":
    # 1. load the image
    aaron = plt.imread(IMG_PATH)
    # 2. create lines
    pts1 = getpoints(aaron)
    import pickle
    # with open("cheez.txt", "wb") as fp:
    #     pickle.dump(pts1, fp)
    print(pts1)
    with open("lines.txt", "wb") as fp:   #Pickling
        pickle.dump(pts1, fp)
