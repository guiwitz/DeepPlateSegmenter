import sys
import deeplate.platesegmenter as ps

datafolder = sys.argv[1]
img_rows = int(sys.argv[2])
img_cols = int(sys.argv[3])
dims = int(sys.argv[4])
if len(sys.argv)==6:
    ps.plate_deeptrain_batches(folder = datafolder, img_rows=img_rows, img_cols=img_cols, dims=dims, weights = sys.argv[5])
else:
    ps.plate_deeptrain_batches(folder = datafolder, img_rows=img_rows, img_cols=img_cols, dims=dims)