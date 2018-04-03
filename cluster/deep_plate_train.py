import sys
import deeplate.platesegmenter as ps

datafolder = sys.argv[1]
img_rows = int(sys.argv[2])
img_cols = int(sys.argv[3])
dims = int(sys.argv[4])

ps.plate_deeptrain(folder = datafolder, img_rows=img_rows, img_cols=img_cols, dims=dims)