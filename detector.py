from math import degrees
from collections import Counter
import numpy as np

import rasterio

from sklearn.cluster import KMeans

from skimage.feature import canny
from skimage.color import rgb2gray
from skimage.filters import gaussian
from skimage.measure import regionprops
from skimage.segmentation import watershed, expand_labels
from skimage.transform import rotate, probabilistic_hough_line

from scipy import ndimage as ndi
from scipy.signal import find_peaks


class Loader:
    def load_image(tif_path: str) -> np.ndarray:
        dataset = rasterio.open(tif_path)
        image = dataset.read().transpose(1, 2, 0)

        return image[:, :, [1, 2, 3]]

class Preprocessor:
    def __init__(self, enhance_range: int) -> None:
        self.enhance_range = enhance_range
    
    @staticmethod
    def get_ndvi(image: np.ndarray) -> np.ndarray:
        nir = image[:, :, 3]
        red = image[:, :, 1]

        ndvi = (nir - red) / (nir + red)
        return np.nan_to_num(ndvi, nan=0.0)
    
    @staticmethod
    def enhance(band: np.ndarray, p: int):
        aux = np.sort(band.flatten())

        imin = int(len(aux) * p / 100)
        imax = int(len(aux) * (100 - p) / 100)

        vmin = float(aux[imin])
        vmax = float(aux[imax])
        
        rimag = (band - vmin) / (vmax - vmin)
        rimag[rimag < 0] = 0
        rimag[rimag > 1] = 1
        
        return rimag
    
    def preprocesss(self, image: np.ndarray) -> np.ndarray:
        ndvi = self.get_ndvi(image)
        image = np.dstack([image, ndvi])

        p = self.enhance_range
        enhanced_bands = [self.enhance(image[..., i], p) for i in range(image.shape[-1])]
        enhanced_image = np.stack(enhanced_bands, axis=-1)
        
        return enhanced_image[:, :, [2, 3, 4]]


class Segmentator:
    def __init__(self, sigma: float, low_threshold: float, high_threshold: float, label_expansion_distance: int, markers: int) -> None:
        self.sigma = sigma
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.label_expansion_distance = label_expansion_distance
        self.markers = markers

    @staticmethod
    def get_edges(image: np.ndarray, sigma: float, low_threshold: float, high_threshold: float) -> np.ndarray:
        gray_img = rgb2gray(image)
        if low_threshold == 0 and high_threshold == 1:
            edges = canny(gray_img, sigma=sigma)
        else:
            edges = canny(gray_img, sigma=sigma, low_threshold=low_threshold, high_threshold=high_threshold, use_quantiles=True)
        
        return edges
    
    def segmentate(self, image: np.ndarray) -> np.ndarray:
        edges = self.get_edges(image, self.sigma, self.low_threshold, self.high_threshold)
        expanded = 1 - expand_labels(edges, distance=self.label_expansion_distance)
        distance = ndi.distance_transform_edt(expanded)
        regions = watershed(-distance, self.markers, mask=expanded)
        regions_with_props = regionprops(regions)

        return regions_with_props
    

class RegionClassifier:
    def __init__(self, mean_size_percentage_threshold: int) -> None:
        self.mean_size_percentile_threshold = mean_size_percentage_threshold

    def classify(self, regions: list) -> list:
        mean_size = np.mean([region.area for region in regions])
        size_threshold = mean_size * (self.mean_size_percentage_threshold / 100)
        return [region for region in regions if region.area > size_threshold]


class Rotator:
    def __init__(self, sigma:float, hough_threshold:int, hough_gap:int) -> None:
        self.sigma = sigma
        self.hough_threshold = hough_threshold
        self.hough_gap = hough_gap

    @staticmethod
    def get_rows_angle(lines: list) -> float:
        angles = []
        for line in lines:
            p0, p1 = line
            angles.append(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))

        if len(angles) > 6:
            km = KMeans(3, random_state=0, n_init=10).fit(np.array(angles).reshape(-1, 1))
            counter = Counter(km.labels_)
            most_common_label = max(counter, key=counter.get)
            angles = np.array(angles)[(np.where(km.labels_ == most_common_label))]

        angles = sorted(angles)
        extremes_len = len(angles) // 20
        if len(angles) <= 2 * extremes_len:
            extremes_len = 0
        angles = angles[extremes_len:len(angles)-extremes_len]
        
        return np.mean(angles)

    def rotate_region(self, region_props, source_image) -> np.ndarray:
        image_bbox = source_image[region_props.bbox[0]:region_props.bbox[2], region_props.bbox[1]:region_props.bbox[3], :]
        region_image = image_bbox.copy()
        region_image[~region_props.image_filled] = [0, 0, 0]

        edges = canny(rgb2gray(region_image), sigma=self.sigma)
        lines = probabilistic_hough_line(
            edges,
            threshold=self.hough_threshold,
            line_length=int(region_props.major_axis_length * 0.1),
            line_gap=self.hough_gap
        )

        rotation_angle = 180 + degrees(self.lines_mean_angle(lines))
        region_image = rotate(region_image, rotation_angle, resize=True)
        return region_image        


class Cropper:
    def __init__(self, row_ratio: float, column_ratio: float) -> None:
        self.row_ratio = row_ratio
        self.column_ratio = column_ratio

    def crop(self, image: np.ndarray) -> np.ndarray:
        img = image.copy()
        if len(img.shape) == 3:
            img = img[:, :, 2] # NDVI BAND

        active_pixels = (img != 0.0) * 1
        sum_active_pixels = np.sum(active_pixels, axis=1)

        total_row_len = img.shape[1]

        lower_bound, upper_bound = np.where(sum_active_pixels/total_row_len >= self.row_ratio)[0][[0, -1]]
        crop = image[lower_bound:upper_bound + 1, :]

        sum_active_pixels = np.sum(active_pixels, axis=0)
        total_column_len = img.shape[0]
        left_bound, right_bound = np.where(sum_active_pixels/total_column_len >= self.column_ratio)[0][[0, -1]]
        crop = crop[:, left_bound:right_bound + 1]

        return crop


class RowDetector:
    def __init__(self, width: int, reverse: bool) -> None:
        self.width = width
        self.reverse = reverse

    def get_crop_rows_indexes(self, image: np.ndarray) -> list:
        ndvi_cumulative_sum = np.sum(gaussian(image[:, :, 2]), axis=1)

        if self.reverse:
            ndvi_cumulative_sum = np.array(list(reversed(ndvi_cumulative_sum))) 

        peaks = find_peaks(ndvi_cumulative_sum, width=self.width)[0]

        return peaks
    

def run():
    ms_path = '../data/DRON_Gilesky_2019-09-18/2Stack_Gilesky2_18-09-19_tif.tif'

    raw_image = Loader().load_image(ms_path)
    preprocessed = Preprocessor(enhance_range=4).preprocess(raw_image)
    
    segmentator = Segmentator(
        sigma=5,
        low_threshold=0.1,
        high_threshold=0.9,
        label_expansion_distance=3,
        markers=50,
    )
    regions_with_props = segmentator.segmentate(preprocessed)

    classifier = RegionClassifier(mean_size_percentage_threshold=25)
    regions = classifier.classify(regions_with_props)

    rotator = Rotator(sigma=1, hough_threshold=1, hough_gap=5)
    rotated_regions = [rotator.rotate_region(region, preprocessed) for region in regions]
    
    cropper = Cropper(row_ratio=0.35, column_ratio=0.05)
    cropped_regions = [cropper.crop(region) for region in rotated_regions]

    row_detector = RowDetector(width=10, reverse=True)
    regions_with_detected_rows = [
        {
            'region': region,
            'crop_rows': row_detector.get_crop_rows_indexes(region)
        }
        for region in cropped_regions
    ]




