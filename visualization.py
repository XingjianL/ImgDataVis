
from mpl_toolkits import mplot3d
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import time

import ImagePrep
class Visualization:
    def __init__(self):
        self.img_prep = ImagePrep.ImagePrep()
        self.fig = plt.figure()
        self.total_plots = 4
        self.subplot_size = math.ceil(math.sqrt(self.total_plots))
        self.current_fig = 1
        self.subplots = []
        self.all_plot_added = False
        self.frame_counts = 0

        self.main_colors = np.array([[0,0,0]])

        pass

    def vis_color_distribution_3d(self, img, num_colors = 3):
        #ax = self.fig.add_subplot(self.subplot_size, self.subplot_size, self.current_fig, projection = '3d')
        img, colors = self.img_prep.localized_color_segmentation(img, final_k = num_colors)
        #ax.scatter(colors[:,0], colors[:,1], colors[:,2], facecolors=colors/255.)
        
        if self.all_plot_added is True: return img
        #self.subplots.append(ax)
        print(self.main_colors.shape, colors.shape)
        self.main_colors = np.concatenate((self.main_colors, colors), axis=0)
        ax = self.fig.add_subplot(self.subplot_size, self.subplot_size, self.current_fig, projection = '3d')
        ax.scatter(self.main_colors[:,0], self.main_colors[:,1],self.main_colors[:,1], facecolors = self.main_colors/255.,)
        ax.set_xlim(0,200)
        ax.set_ylim(0,200)
        ax.set_zlim(0,200)
        self.subplots.append(ax)
        self.current_fig += 1
        return img


    def vis_color_distribution_ch(self, img, channel = 0):
        ch_color, ch_count = np.unique(img[:,:,channel], return_counts=True)       
        ax = self.fig.add_subplot(self.subplot_size, self.subplot_size, self.current_fig)
        ax.plot(ch_color, np.log(ch_count))
        str_ch = str(channel)
        ax.title.set_text(str_ch)
        self.current_fig += 1
    
    def vis_reduced_color_distribution_3d(self, img):
        return
    
    def show_img(self, img):
        ax = self.fig.add_subplot(self.subplot_size, self.subplot_size, self.current_fig)
        ax.imshow(img)

    def clear_figs(self, iterations = 20):
        self.current_fig = 1
        self.frame_counts += 1
        if self.frame_counts > iterations:
            self.frame_counts = 0
            plt.clf()
        if self.main_colors.shape[0] > 50:
            self.main_colors = np.delete(self.main_colors, np.random.randint(0, 40, size=25), axis = 0)

        #plt.clf()
    
    def new_figure(self):
        self.fig = plt.figure()
        
    def show_plot(self):
        plt.pause(0.1)
    


source_directory = "/home/xing/TesterCodes/OpenCV/PathProject/Data/manual_path_edited.mp4"

img_prep = ImagePrep.ImagePrep()
img_vis = Visualization()

cap = cv2.VideoCapture(source_directory)
while(cap.isOpened()):
    _, frame = cap.read()
    height, width = frame.shape[:2]
    height, width = int(height * .2), int(width * .2)
    frame = cv2.resize(frame, (width, height))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    local_seg = img_vis.vis_color_distribution_3d(frame)
    #img_vis.vis_color_distribution_ch(frame, channel=0)
    #img_vis.vis_color_distribution_ch(frame, channel=1)
    #img_vis.vis_color_distribution_ch(frame, channel=2)
    img_vis.show_img(local_seg)
    print("new frame")
    img_vis.show_plot()
    img_vis.clear_figs()
plt.show()