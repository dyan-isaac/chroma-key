import cv2
import numpy as np
import sys


class ColorSpaceChromaProcessor:

    def getCentroid(self, img):
        gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        _, binary = cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY_INV)

        # find contours in the binary image
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            moments = cv2.moments(largest_contour)
            if moments["m00"] != 0:
                cX = int(moments["m10"] / moments["m00"])

                return cX

        return None

    def resizeImage(self, image, min_size=(1280, 960), max_size=(1280, 960), row_col_num=2):
        # Just set the above minmax size for the whole window to 1280 after the requirement change and did not touch
        # below, just added the compute for ratio to apply interpolation as needed.
        target_width = min(max_size[0] // row_col_num, max(min_size[0] // row_col_num, image.shape[1]))
        target_height = min(max_size[1] // row_col_num, max(min_size[1] // row_col_num, image.shape[0]))

        aspect_ratio = image.shape[0] / image.shape[1]

        # Get the ratio of change from the original size to target width
        ratio = image.shape[1] / target_width

        if aspect_ratio > 1:
            # Wider image
            target_height = min(target_height, int(target_width * aspect_ratio))
            target_width = int(target_height / aspect_ratio)
        else:
            # Taller image
            target_width = min(target_width, int(target_height / aspect_ratio))
            target_height = int(target_width * aspect_ratio)

        # Only apply interpolation for significant change in size. Just setting it to a threshold of 1.5
        if ratio > 1.5:
            resized_image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)
        else:
            resized_image = cv2.resize(image, (target_width, target_height))

        return resized_image

    def displayViewingWindow(self, image1, image2, image3, image4, mode):
        if mode == 1:
            image00 = self.resizeImage(image1)
            image01 = cv2.cvtColor(self.resizeImage(image2), cv2.COLOR_GRAY2RGB)
            image10 = cv2.cvtColor(self.resizeImage(image3), cv2.COLOR_GRAY2RGB)
            image11 = cv2.cvtColor(self.resizeImage(image4), cv2.COLOR_GRAY2RGB)
        else:
            image00 = cv2.cvtColor(self.resizeImage(image1), cv2.COLOR_BGR2RGB)
            image01 = cv2.cvtColor(self.resizeImage(image2), cv2.COLOR_BGR2RGB)
            image10 = self.resizeImage(image3)
            image11 = cv2.cvtColor(self.resizeImage(image4), cv2.COLOR_BGR2RGB)

        col1 = cv2.vconcat([image00, image10])
        col2 = cv2.vconcat([image01, image11])
        grid = cv2.hconcat([col1, col2])

        cv2.imshow(f"Task {str(mode)} {grid.shape}", grid)
        cv2.waitKey()
        cv2.destroyAllWindows()

    def returnSplitChannel(self, colorspace, image):
        image = cv2.cvtColor(image, colorspace)
        image_1, image_2, image_3 = cv2.split(image)

        return image_1, image_2, image_3

    def loadTaskOne(self, image_file, option):
        original_image = cv2.imread(image_file)
        if original_image is None:
            print(f"Error: Unable to load image file {image_file}")
            return

        image = original_image.copy()

        if option == "-XYZ":
            image_x, image_y, image_z = self.returnSplitChannel(cv2.COLOR_BGR2XYZ, image)

            self.displayViewingWindow(original_image, image_x, image_y, image_z, 1)
        elif option == "-Lab":
            # Lightness, a- color component ranging from Green to Magenta,
            # b- color component ranging from Blue to Yellow
            image_l, image_a, image_b = self.returnSplitChannel(cv2.COLOR_RGB2Lab, image)

            self.displayViewingWindow(original_image, image_l, image_a, image_b, 1)
        elif option == "-YCrCb":
            # Y- Luminance or Luma component obtained from RGB gamma correction,
            # Cr = R-Y (how far is the red component from Luma), Cb = B-Y (how far is the blue component from Luma)
            image_y, image_cr, image_cb = self.returnSplitChannel(cv2.COLOR_RGB2YCrCb, image)

            self.displayViewingWindow(original_image, image_y, image_cr, image_cb, 1)
        elif option == "-HSB":
            # H-Hue, -Saturation, B-Brightness
            image_h, image_s, image_b = self.returnSplitChannel(cv2.COLOR_RGB2HSV, image)

            self.displayViewingWindow(original_image, image_h, image_s, image_b, 1)

    def loadTaskTwo(self, scenicImage, greenScreenImage):
        org_screen_image = cv2.imread(greenScreenImage)
        scenicImage = cv2.imread(scenicImage, cv2.COLOR_BGR2RGB)

        if org_screen_image is None:
            print(f"Error: Unable to load green screen file {greenScreenImage}")
            return

        if scenicImage is None:
            print(f"Error: Unable to load scenic image file {scenicImage}")
            return

        org_screen_image = self.resizeImage(org_screen_image, (scenicImage.shape[1], scenicImage.shape[0]),
                                            (scenicImage.shape[1], scenicImage.shape[0]), 1)
        orig_green_img_copy = cv2.cvtColor(org_screen_image.copy(), cv2.COLOR_BGR2RGB)

        # a- channel : green to magenta component since we want to isolate the green screen area
        a_channel = self.returnSplitChannel(cv2.COLOR_BGR2LAB, org_screen_image.copy())[1]

        # Green area converted to black
        th = cv2.threshold(a_channel, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        masked = cv2.bitwise_and(org_screen_image, org_screen_image, mask=th)

        # Convert black area to white (initial)
        mask_green_bg = masked.copy()
        mask_green_bg[th == 0] = (255, 255, 255)

        # Retrieve the green border around the person to improve the edge around the person
        green_border = cv2.cvtColor(masked, cv2.COLOR_BGR2LAB)
        green_border_a_channel = cv2.normalize(green_border[:, :, 1], dst=None, alpha=0, beta=255,
                                               norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # Segment green shaded areas
        threshold_value = 100
        dst_th = cv2.threshold(green_border_a_channel, threshold_value, 255, cv2.THRESH_BINARY_INV)[1]

        green_border[:, :, 1][dst_th == 255] = 127

        # Convert the image to BGR and set to white where its 0 in the threshold border image (th)
        img2 = cv2.cvtColor(green_border, cv2.COLOR_LAB2BGR)
        img2[th == 0] = (255, 255, 255)

        final_person_white_bg = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        # Contains black for the non-green pixels
        inv_mask = 255 - th

        # Processing of overlaying the green screen image on top of the background scenic image starts here.

        overlay = cv2.cvtColor(final_person_white_bg.copy(), cv2.COLOR_RGB2BGRA)

        # Convert white areas to transparent
        overlay[np.all(overlay[:, :, :3] == [255, 255, 255], axis=-1)] = [0, 0, 0, 0]

        bg_height, bg_width, _ = scenicImage.shape
        overlay_height, overlay_width = overlay.shape[:2]

        # Get the center of the black portion of this inverted mask that represents the
        # person/object we want to overlay on top of the background
        cX_overlay = self.getCentroid(cv2.cvtColor(inv_mask, cv2.COLOR_GRAY2RGB))
        cX_scenic = bg_width // 2

        # Calculate by how much we need to adjust the overlay,
        # here the x_offset is the distance between the center of the background
        # and the center of the inverted mask
        x_offset = cX_scenic - cX_overlay

        # Align the overlay to the bottom of the background scenic image
        y_offset = bg_height - overlay_height

        if scenicImage.shape[2] == 3:
            scenicImage_RGBA = cv2.cvtColor(scenicImage, cv2.COLOR_BGR2BGRA)
        else:
            scenicImage_RGBA = scenicImage

        # Apply offsets and blend the overlay on the scenic image
        for y in range(overlay_height):
            for x in range(overlay_width):
                # Only select those that are not transparent from the overlay
                if overlay[y, x, 3] != 0:
                    new_x = x + x_offset
                    new_y = y + y_offset
                    if 0 <= new_x < bg_width and 0 <= new_y < bg_height:
                        alpha = overlay[y, x, 3] / 255.0

                        scenicImage_RGBA[new_y, new_x, :3] = (alpha * overlay[y, x, :3] +
                                                              (1 - alpha) * scenicImage_RGBA[new_y, new_x, :3]).astype(
                            np.uint8)

                        scenicImage_RGBA[new_y, new_x, 3] = 255

        self.displayViewingWindow(orig_green_img_copy, final_person_white_bg, scenicImage,
                                  cv2.cvtColor(scenicImage_RGBA, cv2.COLOR_BGRA2RGB), 2)

    def run(self):
        first_arg = sys.argv[1]
        second_arg = sys.argv[2]

        if first_arg in ['-XYZ', '-Lab', '-YCrCb', '-HSB']:
            # Color space conversion
            if second_arg:
                self.loadTaskOne(second_arg, first_arg)
            else:
                print("Error: Missing image file for color space conversion.")
                self.parser.print_help()
        elif second_arg:
            # Chroma keying - 1st scenic , 2nd green screen image
            self.loadTaskTwo(first_arg, second_arg)
        else:
            print("Error: Invalid argument combination.")
            self.parser.print_help()


if __name__ == "__main__":
    processor = ColorSpaceChromaProcessor()
    processor.run()
# %%
