import numpy as np
import cv2
import pytesseract
from tqdm import tqdm
from sudoku import Sudoku

SIZE = 450
DIM = SIZE // 9


def do_preprocessing(image):
    # Grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("pre.gray", image)

    # Blur
    image = cv2.GaussianBlur(image, (15, 15), 0)
    # cv2.imshow("pre.blur", image)

    # Threshold
    # -  Binary Thresholing
    # image = cv2.threshold(image, 170, 255, cv2.THRESH_BINARY)

    # -  Adaptive Thresholding
    image = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 5
    )
    # cv2.imshow("pre.threshold", image)

    # Erode
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    image = cv2.erode(image, kernel, iterations=1)
    # cv2.imshow("pre.erode", image)

    return image


def get_contours(image):
    # Detect image edges
    image = cv2.Canny(image, 30, 200)
    # cv2.imshow("contours.edge", image)

    # Find contours
    contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # # Uncomment to see all contours
    # all_contours = cv2.drawContours(
    #     cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), contours, -1, (255, 0, 0), 4
    # )
    # cv2.imshow("contours.all", all_contours)

    # Select largest contour
    best_contour = max(contours, key=cv2.contourArea)

    # # Uncomment to see best contour
    # image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    # cv2.drawContours(image, [best_contour], -1, (255, 0, 0), 8)
    # cv2.imshow("contours.best", image)

    return best_contour


def get_points(image, contour):
    sums = []
    diffs = []

    for point in contour:
        for x, y in point:
            sums.append(x + y)
            diffs.append(x - y)

    top_left = contour[np.argmin(sums)].squeeze()
    bottom_right = contour[np.argmax(sums)].squeeze()
    top_right = contour[np.argmax(diffs)].squeeze()
    bottom_left = contour[np.argmin(diffs)].squeeze()

    # # Uncomment to see corners
    # image = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)
    # cv2.circle(image, top_left, 10, (0, 0, 255), -1)
    # cv2.circle(image, bottom_right, 10, (0, 0, 255), -1)
    # cv2.circle(image, top_right, 10, (0, 0, 255), -1)
    # cv2.circle(image, bottom_left, 10, (0, 0, 255), -1)
    # cv2.imshow("points", image)

    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


def do_warp(image, pts):
    target = np.array([[0, 0], [SIZE, 0], [SIZE, SIZE], [0, SIZE]], dtype=np.float32)

    # Compute warp matrix
    warp_mat = cv2.getPerspectiveTransform(pts, target)

    # Apply warp
    warped = cv2.warpPerspective(image, warp_mat, (SIZE, SIZE))

    return warped


def remove_gridlines(image):
    # Invert image
    image = ~image

    # Find horizontal lines
    horizontal_k = cv2.getStructuringElement(cv2.MORPH_RECT, (45, 1))
    horizontal_mask = cv2.dilate(
        cv2.erode(image, horizontal_k, iterations=1), horizontal_k, iterations=1
    )
    # cv2.imshow("grid.horizontal", horizontal)

    # Find vertical lines
    vertical_k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 45))
    vertical_mask = cv2.dilate(
        cv2.erode(image, vertical_k, iterations=1), vertical_k, iterations=1
    )
    # cv2.imshow("grid.vertical", vertical)

    # Combine to find all gridlines
    combined = horizontal_mask | vertical_mask
    # cv2.imshow("grid.combined", combined)

    # Expand selection to match more
    gridlines = cv2.dilate(
        combined,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    # cv2.imshow("grid.dilated", gridlines)

    # Uninvert image and apply mask to remove grid lines
    image = ~image | gridlines
    # cv2.imshow("without_grid", image)

    return image


def make_slices(image):
    slices = []
    for y in range(9):
        for x in range(9):
            slices.append(image[y * DIM : (y + 1) * DIM, x * DIM : (x + 1) * DIM])
    return slices


def do_ocr(slices):
    # 9 x 9 array, 0 = unfilled.
    board = [[0] * 9 for _ in range(9)]
    for i, slice in tqdm(enumerate(slices), total=81, desc="OCR"):
        x = i % 9
        y = i // 9

        # Use Tesseract to detect the number
        num = pytesseract.image_to_string(
            slice,
            config="-c tessedit_char_whitelist='123456789' --psm 6",
        )

        if num:
            board[y][x] = int(num)

    return board


def make_overlay(unsolved, solved):
    # Create a blank image to store the overlay
    overlay = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)

    for y in range(9):
        for x in range(9):
            # Identify if this cell was modified by the solver
            if unsolved[y][x] != solved[y][x]:
                # Write the number onto the overlay image
                cv2.putText(
                    overlay,
                    str(solved[y][x]),
                    (x * DIM + DIM // 2 - 10, y * DIM + DIM // 2 + 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 0),
                    2,
                )
    # cv2.imshow("overlay", overlay)
    return overlay


def do_unwarp(image, pts, original):
    source = np.array([[0, 0], [SIZE, 0], [SIZE, SIZE], [0, SIZE]], dtype=np.float32)

    # Compute unwarp matrix
    unwarp_mat = cv2.getPerspectiveTransform(source, pts)

    # Apply unwarp
    image = cv2.warpPerspective(
        image, unwarp_mat, (original.shape[1], original.shape[0])
    )
    # cv2.imshow("unwarped", image)
    return image


def do_composite(image, overlay):
    # Invert overlay
    overlay = 255 - overlay

    # Apply overlay onto original image
    composite = cv2.bitwise_and(image, overlay)
    # cv2.imshow("composite", composite)
    return composite


def print_board(board):
    """Pretty-print sudoku board"""
    print("╔═══╤═══╤═══╦═══╤═══╤═══╦═══╤═══╤═══╗")
    for y, row in enumerate(board):
        print("║", end="")
        for x, val in enumerate(row):
            print(f" {val or ' '} ", end="║" if x % 3 == 2 else "│")
        print()
        if y % 3 == 2:
            if y == 8:
                print("╚═══╧═══╧═══╩═══╧═══╧═══╩═══╧═══╧═══╝")
            else:
                print("╠═══╪═══╪═══╬═══╪═══╪═══╬═══╪═══╪═══╣")
        else:
            print("╟───┼───┼───╫───┼───┼───╫───┼───┼───╢")


def main():
    original = cv2.imread("sudoku_assets/puzzle.jpg")

    # Step 1: Clean up source image
    image = do_preprocessing(original)

    # Step 2: Fix perspective warp
    contour = get_contours(image)
    pts = get_points(image, contour)
    warped = do_warp(image, pts)

    # Step 3: Remove gridlines and slice for OCR
    grid_free = remove_gridlines(warped)
    slices = make_slices(grid_free)

    # Step 4: Do OCR
    board = do_ocr(slices)
    print("Unsolved:")
    print_board(board)

    # Step 4a: Solve board (using external library)
    solved = Sudoku(3, 3, board).solve().board
    print("Solved:")
    print_board(solved)

    # Step 5: Construct solution image
    overlay = make_overlay(board, solved)

    # Step 6: Warp and composite solution onto original image
    unwarped = do_unwarp(overlay, pts, original)
    composite = do_composite(original, unwarped)

    # Step 7: Write image
    cv2.imwrite("solved.png", composite)


if __name__ == "__main__":
    main()
