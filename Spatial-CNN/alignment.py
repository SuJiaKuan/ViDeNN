import cv2
import numpy as np


class ImageRegistrationAligner:
    def __init__(self):
        pass

    def computeSrcToDstMap(src_img, dst_img):
        raise NotImplementedError


# Method 1.
# See: https://docs.opencv.org/3.4/d1/d89/tutorial_py_orb.html
# See: https://stackoverflow.com/questions/46520123
class OrbFeatureAligner(ImageRegistrationAligner):
    def __init__(self):
        self.orb = cv2.ORB_create(
            edgeThreshold=15,
            patchSize=31,
            nlevels=8,
            fastThreshold=20,
            scaleFactor=1.2,
            WTA_K=2,
            scoreType=cv2.ORB_HARRIS_SCORE,
            firstLevel=0,
            nfeatures=1500,
        )
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.flann = cv2.FlannBasedMatcher_create()

    def computeSrcToDstMap(self, src_img, dst_img):
        src_keypts, src_feats = self.orb.detectAndCompute(src_img, mask=None)
        dst_keypts, dst_feats = self.orb.detectAndCompute(dst_img, mask=None)
        matches = self.bf.match(src_feats, dst_feats)
        # matches = self.flann.knnMatch(src_feats, dst_feats, k=2)

        # Store all the good matches as per Lowe's ratio test.
        # map_pairs = []
        # for m, n in matches:
        #     if m.distance < 0.7 * n.distance:
        #         map_pairs.append(m)
        map_pairs, num_pairs = matches, len(matches)

        if num_pairs < 4:
            raise ValueError("less than 4 pairs, unable to estimate homography")

        # Extract matched keypoints for later homography estimation
        src_good_pts = np.float32(
            [src_keypts[m.queryIdx].pt for m in map_pairs]
        ).reshape(-1, 1, 2)
        dst_good_pts = np.float32(
            [dst_keypts[m.trainIdx].pt for m in map_pairs]
        ).reshape(-1, 1, 2)
        dst_to_src_H, mask = cv2.findHomography(
            dst_good_pts, src_good_pts, cv2.RANSAC, ransacReprojThreshold=3.0
        )
        if dst_to_src_H is None:
            raise ValueError("fail to reach consensus homography via random sampling")

        # Draw matching images
        # draw_params = dict(
        #     matchColor=(0, 255, 0),  # draw matches in green color
        #     singlePointColor=None,
        #     matchesMask=mask.ravel().tolist(),  # draw only inliers
        #     flags=2,
        # )
        # viz_matches = cv2.drawMatches(
        #     src_img, src_keypts, dst_img, dst_keypts, map_pairs, None, **draw_params
        # )
        viz_matches = None

        # Compute entire warping lookup table from homography matrix
        # See: https://stackoverflow.com/questions/46520123
        # create indices of the destination image and linearize them
        h, w = dst_img.shape[:2]
        indy, indx = np.indices((h, w), dtype=np.float32)
        lin_homg_ind = np.array(
            [indx.ravel(), indy.ravel(), np.ones_like(indx).ravel()]
        )

        # warp the coordinates of src to those of true_dst
        map_ind = dst_to_src_H.dot(lin_homg_ind)
        map_x, map_y = map_ind[:-1] / map_ind[-1]  # ensure homogeneity
        map_x = map_x.reshape(h, w).astype(np.float32)
        map_y = map_y.reshape(h, w).astype(np.float32)

        return map_x, map_y, viz_matches


def align_images(src_img, dst_img):
    img_reg = OrbFeatureAligner()

    map_x, map_y, viz_matches = img_reg.computeSrcToDstMap(src_img, dst_img)
    map_img = cv2.remap(src_img, map_x, map_y, cv2.INTER_LANCZOS4)

    diff_no_comp = cv2.absdiff(dst_img, src_img)
    diff_comp_on = cv2.absdiff(dst_img, map_img)
    sad_no_comp, sad_comp_on = np.sum(diff_no_comp), np.sum(diff_comp_on)

    return sad_comp_on < sad_no_comp, map_img
