
from turtle import color, width
from mpl_toolkits import mplot3d
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import time
class ImagePrep:
    KMEANSFILTER = [3,  # num of clusters
                4,  # num of iterations
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0), # criteria
                cv2.KMEANS_PP_CENTERS]  # flag

    def __init__(self, slice_size = 10, kmeans_filter = KMEANSFILTER):
        self.slice_size = slice_size
        self.k, self.iter_num, self.criteria, self.flag = kmeans_filter

    def slice(self, image):
        arr_size = tuple(int(element / self.slice_size) for element in image.shape)
        col_array = np.array_split(image, arr_size[0], axis=0)
        img_array = []
        for col in col_array:
            img_array.append(np.array_split(col,arr_size[1],axis=1))
        return img_array

    def combineRow(self, imgs):
        combined_img = imgs[0]
        for img in imgs[1:]:
            combined_img = np.concatenate((combined_img,img),axis=1)
        return combined_img

    def combineCol(self, imgs):
        combined_img = imgs[0]
        for img in imgs[1:]:
            combined_img = np.concatenate((combined_img,img),axis=0)
        return combined_img

    def reduce_image_color(self, image, ncluster = None):
        img_kmean = image.reshape(-1,3)
        img_kmean = np.float32(img_kmean)
        if ncluster is not None:
            ret,label,center = cv2.kmeans(img_kmean,ncluster,None,self.criteria,self.iter_num,self.flag)
        else:
            ret,label,center = cv2.kmeans(img_kmean,self.k,None,self.criteria,self.iter_num,self.flag)
        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((image.shape))
        return res2, center
    
    def localized_color_segmentation(self, img, inter_k = 2, final_k = 20):
        slice_imgs = self.slice(frame)
        kmeans = slice_imgs.copy()
        comb_row = [i for i in range(len(slice_imgs))]
        for i,row in enumerate(slice_imgs):
            for j,block in enumerate(row):
                kmeans[i][j], _ = self.reduce_image_color(block, inter_k)
            comb_row[i] = (self.combineRow(kmeans[i]))
        combined_filter = self.combineCol(comb_row)
        combined_filter, colors = self.reduce_image_color(combined_filter, final_k)
        return combined_filter, colors

class Visualization:
    def __init__(self):
        self.img_prep = ImagePrep()
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
        ax.set_xlim(0,255)
        ax.set_ylim(0,255)
        ax.set_zlim(0,255)
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
        if self.main_colors.shape[0] > 100:
            self.main_colors = np.delete(self.main_colors, np.random.randint(0, 100, size=80), axis = 0)

        #plt.clf()
    
    def new_figure(self):
        self.fig = plt.figure()
        
    def show_plot(self):
        plt.pause(0.1)
    


source_directory = "/home/xing/TesterCodes/OpenCV/PathProject/Data/manual_path_edited.mp4"

img_prep = ImagePrep()
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