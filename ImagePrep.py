import cv2
import numpy as np
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
        slice_imgs = self.slice(img)
        kmeans = slice_imgs.copy()
        comb_row = [i for i in range(len(slice_imgs))]
        for i,row in enumerate(slice_imgs):
            for j,block in enumerate(row):
                kmeans[i][j], _ = self.reduce_image_color(block, inter_k)
            comb_row[i] = (self.combineRow(kmeans[i]))
        combined_filter = self.combineCol(comb_row)
        combined_filter, colors = self.reduce_image_color(combined_filter, final_k)
        return combined_filter, colors

    # input: binary image
    # output: mean and PCA vector, values of the white (non-zero) region
    # useful for orientation, variance, and location of the white region
    def binary_PCA(image):                       
        pca_vector = []
        #image = cv2.resize(image,IMAGE_SIZE)
        coords_data = np.array(cv2.findNonZero(image)).T.reshape((2,-1))            # 2 x n matrix of coords [[x1,x2,...],[y1,y2,...]]
        mean = np.mean(coords_data,axis=1,keepdims=True)                         # center of coords
        cov_mat = np.cov(coords_data - mean, ddof = 1)              # find covariance
        pca_val, pca_vector = np.linalg.eig(cov_mat)                # find eigen vectors (also PCA first and second component)
        return mean, pca_vector, pca_val

    # input: colored image
    # output: mean and PCA vector, values of the colors in the image
    # for transforming and compressing the colors, which may be useful for analysis as demonstrated in 3d visualization
    def color_3d_PCA(self, image):
        pca_vector =[]
        coords_data = image.reshape((-1,3)).T            # 3 x n matrix of coords [[b1,b2,...],[g1,g2,...],[r1,r2,...]]
        mean = np.mean(coords_data, axis=1, keepdims=True)                            # center of each color
        cov_mat = np.cov(coords_data - mean, ddof = 1)                              # find covariance
        pca_val, pca_vector = np.linalg.eig(cov_mat)                                # find eigen vectors (also PCA first and second component)
        return mean, pca_vector, pca_val

    # project the image colors onto vectors (can select between 1 or 2 vectors)
    # pca_output: from color_3d_PCA()
    # dim: number of most significant vectors
    # output: image with PCA color reduction
    def simplify_by_pca(self, image, pca_output, dim = 2):
        # prepare the color information from the image
        pixel_colors = np.transpose(image,(2,0,1)).reshape(3,-1) # 3 x n, n = width * height
        mean, pca_vector, pca_val = pca_output
        # find the most significant vectors
        sorted_pca = np.argsort(pca_val)
        # filter out the unimportant ones
        selected_vectors = pca_vector[:,sorted_pca[-dim:]] # 3 x k, k = dim

        # sometimes the vectors go in the negative direction, this loop makes the vectors more consistent
        for i in range(dim):
            if selected_vectors[1,i] < 0:
                selected_vectors[:,i] = -selected_vectors[:,i]

        # compute locations in the reduced dimension
        reduced_data_points = np.dot(selected_vectors.T, pixel_colors) # k x 3 * 3 x n -> k x n

        # shift the color above 0
        reduced_data_points = np.where(np.min(reduced_data_points, axis=0) < 0, reduced_data_points - np.min(reduced_data_points, axis=0), reduced_data_points)
        
        # add pixels to keep the output with 3 colors, using the minimum of the existing ones
        # k x n -> 3 x n
        for i in range(3-dim):
            reduced_data_points = np.vstack((np.min(reduced_data_points, axis=0) * np.ones(reduced_data_points.shape[1]).reshape(1,-1),reduced_data_points))

        # convert the 3 x n matrix to original image shape
        return reduced_data_points.T.reshape((image.shape)).astype(int)
        
        
